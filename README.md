# Arbitrage statistique sur paires DeFi : extensions de l'OTT

Pipeline complet d'une stratégie de *pairs trading* sur crypto-actifs : acquisition des
données, cointégration, modélisation Ornstein-Uhlenbeck, signal à bandes dynamiques,
allocation rendement/CVaR et backtest avec frictions DeFi réalistes.

**Univers** : stETH/ETH (LST), USDC/USDT (stablecoins), AAVE/ETH (DeFi).
**Données** : API CoinGecko, horaire, sept. 2025 → sept. 2026 (8 736 points par actif).

## Résultat principal

> Les deux paires cointégrées montrent un fort retour à la moyenne **sur le papier**
> (Sharpe brut de 2 à 4), mais **aucune ne dégage de rendement net exploitable hors
> échantillon** une fois pris en compte un délai d'exécution réaliste et les frictions.

Trois constats, dans l'ordre où ils détruisent la performance :

1. **Le délai d'exécution.** La demi-vie du spread stETH/ETH est d'environ 2 heures :
   une heure après le signal, une bonne partie de l'écart s'est déjà résorbée. Exécuter
   au prix même qui a déclenché le signal (hypothèse implicite de beaucoup de backtests)
   gonfle fortement le résultat. Sur la période d'apprentissage, stETH/ETH passe de
   +17,3 % à +5,4 % de PnL brut avec 1 h de délai.
2. **Les frictions.** Frais de pool, slippage et gas absorbent le reste. Plus le seuil
   d'entrée est bas, plus on trade et plus le Sharpe net s'effondre (figure ci-dessous).
3. **Le hors-échantillon.** Le seuil choisi sur l'apprentissage (|z| > 4) ne génère que
   1 à 2 trades sur la période de test : aucune conclusion statistique n'est possible.

AAVE/ETH n'est pas cointégrée (ADF ≈ −1,7) et est exclue dès l'étape 2.

![PnL cumulé](outputs/figures/equity_curves.png)

![Sensibilité au seuil](outputs/figures/sensitivity.png)

### Chiffres clés (seuil |z| > 4, capital de 100 000 $ équipondéré)

| Paire | Délai | Période | PnL brut | PnL net | Sharpe net | Trades |
|---|---|---|---|---|---|---|
| stETH/ETH | 0 h | apprentissage | +17,3 % | +11,1 % | 2,77 | 16 |
| stETH/ETH | 1 h | apprentissage | +5,4 % | −0,8 % | −0,28 | 16 |
| stETH/ETH | 1 h | **test** | +0,0 % | **−0,4 %** | −2,49 | 1 |
| USDC/USDT | 0 h | apprentissage | +1,5 % | +0,6 % | 1,48 | 7 |
| USDC/USDT | 1 h | apprentissage | +0,4 % | −0,4 % | −1,45 | 7 |
| USDC/USDT | 1 h | **test** | +0,3 % | **+0,1 %** | 0,30 | 2 |

Rendements en % du notionnel de la ligne, non annualisés. Détail complet dans
`outputs/backtest_results.csv`.

## Méthodologie

| Étape | Choix | Par rapport à l'article original |
|---|---|---|
| Découpage | 2/3 apprentissage (sept. 2025 → mai 2026), 1/3 test (mai → sept. 2026) | tout paramètre est estimé sur l'apprentissage uniquement |
| Hedge ratio | β = 1 imposé pour les paires ancrées 1:1, OLS sinon | — |
| Cointégration | ADF codé à la main (numpy) + processus d'Ornstein-Uhlenbeck | remplace Engle-Granger statique ; donne la demi-vie |
| Signal | z-score sur moyenne/écart-type EWMA (fenêtre = 10 × demi-vie OU) | remplace des seuils fixes calculés sur tout l'historique |
| Exécution | signal à t, trade à t + 1 h | l'original exécute au prix du signal |
| Seuil d'entrée | choisi sur l'apprentissage parmi {1,5 … 4} | — |
| Allocation | max rendement/CVaR 5 % (repli sur CVaR min. si tous les rendements sont négatifs) | remplace rendement/variance |
| Frictions | palier de frais AMM, slippage x·y=k, gas log-normal en Gwei | remplace un frais fixe de 0,1 % |

## Limites

- **Données de prix** : les prix CoinGecko sont des agrégats horaires multi-plateformes,
  pas des prix exécutables. Une partie du « retour à la moyenne » mesuré à l'heure est
  probablement du bruit d'agrégation.
- **Liquidité** : faute de données on-chain, la profondeur des pools est approchée par
  10 × le volume 24h CoinGecko (toutes places confondues). C'est une hypothèse forte ;
  le slippage est vraisemblablement sous-estimé pour stETH et surestimé pour USDC.
- **Gas** : simulé (log-normale centrée sur 25 Gwei), pas historique.
- **Petit échantillon de test** : 1 à 2 trades par paire sur 4 mois au seuil retenu.
- **Allocation** : le gas étant un coût fixe en USD, les rendements nets dépendent du
  notionnel ; l'optimisation les calcule au notionnel équipondéré (approximation).

## Structure

```
.
├── data/                       # CSV bruts CoinGecko (timestamp_ms, price_usd, volume_usd)
├── outputs/                    # résultats (CSV) et figures, régénérés par run_all.py
└── src/
    ├── config.py               # découpage apprentissage/test, délai d'exécution, capital
    ├── fetch_coingecko.py      # 1. acquisition des données
    ├── cointegration_ou.py     # 2. hedge ratio, ADF, Ornstein-Uhlenbeck
    ├── signal_generation.py    # 3. bandes EWMA et machine à états
    ├── sensitivity_frequency.py#    choix du seuil d'entrée (apprentissage)
    ├── backtest_engine.py      # 4. backtest avec frictions, métriques par période
    ├── portfolio_optimization.py # 5. allocation rendement/CVaR
    ├── make_figures.py         #    figures du README
    └── run_all.py              #    exécute tout dans l'ordre
```

## Reproduire

```bash
pip install -r requirements.txt
python src/run_all.py            # ~1 min ; les données sont fournies dans data/
python src/fetch_coingecko.py    # optionnel : retélécharger les données (365 derniers jours)
```

Pour voir l'effet du délai d'exécution, changer `EXECUTION_LAG_HOURS` dans `src/config.py`.
