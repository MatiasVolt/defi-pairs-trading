"""
README figures:
  1. gross/net equity curves per pair, with the training/test split;
  2. net Sharpe ratio as a function of the entry threshold, with and without execution lag.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import EXECUTION_LAG_HOURS, OUTPUT_DIR, ensure_output_dir, split_timestamp
from backtest_engine import run_backtests
from sensitivity_frequency import run_sensitivity

COLORS = {"gross": "#2a6f97", "net": "#c8553d"}


def equity_figure():
    res = run_backtests()
    fig, axes = plt.subplots(len(res), 1, figsize=(10, 3.4 * len(res)), sharex=True)
    for ax, (pair, r) in zip(axes, res.items()):
        for mode, key in [("gross", "raw_pnl_usd"), ("net", "net_pnl_usd")]:
            pnl = r["bt"][key].cumsum() / r["notional"] * 100
            ax.plot(pnl.index, pnl.values, color=COLORS[mode], lw=1.4,
                    label=f"{mode} ({'without' if mode == 'gross' else 'with'} frictions)")
        split = split_timestamp(r["bt"]["position"].index)
        ax.axvline(split, color="#555", ls="--", lw=1)
        ax.axvspan(split, r["bt"]["position"].index[-1], color="#f2f2f2", zorder=0)
        ax.axhline(0, color="#999", lw=0.6)
        ax.set_title(pair.replace("_", " / "), loc="left", fontsize=11)
        ax.set_ylabel("Cumulative PnL (% of notional)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9, loc="upper left")
    for ax in axes:
        ax.text(0.99, 0.04, "shaded area = test period", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color="#555")
    fig.suptitle(f"Cumulative PnL, threshold chosen on the training period, execution lag {EXECUTION_LAG_HOURS} h",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figures" / "equity_curves.png", dpi=150)


def sensitivity_figure():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, lag in zip(axes, [0, EXECUTION_LAG_HOURS]):
        df = run_sensitivity(lag=lag)
        df = df[(df["mode"] == "net") & (df.period == "test")]
        for pair, g in df.groupby("pair"):
            ax.plot(g.entry_z, g.sharpe, marker="o", lw=1.4, label=pair.replace("_", " / "))
        ax.axhline(0, color="#999", lw=0.6)
        ax.set_title(f"Execution lag: {lag} h", loc="left", fontsize=11)
        ax.set_xlabel("Entry threshold (|z|)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Annualized net Sharpe ratio (test)")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Out-of-sample net Sharpe ratio by entry threshold", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figures" / "sensitivity.png", dpi=150)


if __name__ == "__main__":
    ensure_output_dir()
    equity_figure()
    sensitivity_figure()
    print(f"Figures saved in {OUTPUT_DIR.name}/figures/")
