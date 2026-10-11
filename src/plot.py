"""
plot.py - equity curve chart (matplotlib). Optional; only needed for --plot.
"""
def plot_equity_curve(result, out_path: str = "equity.png", title: str = "Equity Curve"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(result.equity_curve.values, color="#4ade80", linewidth=1.3)
    ax.axhline(result.initial_capital, color="#888", linewidth=0.8, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Bar")
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
