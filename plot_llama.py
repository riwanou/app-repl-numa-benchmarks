import os
import config
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import numpy as np
import pandas as pd


RESULT_DIR = config.RESULT_DIR

TESTS = [("pp512", "pp512"), ("tg128", "tg128")]
TEST_NAMES = [label for _, label in TESTS]
N_TESTS = len(TESTS)

CSVS = ["llama", "llama-repl"]

# csv tags exactly as bench_llama writes them. Every arm runs --numa
# distribute and is preloaded; the labels drop both, the text explains them.
BASELINE = "distribute-balancing-warmup"  # the 0 line
VALUE_TAG = "repl-distribute-warmup"  # the bar carrying absolute tokens/s
TAGS = [
    "distribute-balancing-warmup",
    "distribute-warmup",
    "interleaved-distribute-warmup",
    "repl-distribute-warmup",
]
TAG_LABELS = {
    "distribute-balancing-warmup": "Linux Vanilla",
    "distribute-warmup": "First Touch",
    "interleaved-distribute-warmup": "Interleave",
    "repl-distribute-warmup": "SPARe",
}
RAMP_STEPS = 17
TAG_SHADES = {
    "distribute-warmup": (config.LINUX_COLOR, 2),
    "interleaved-distribute-warmup": (config.LINUX_COLOR, 8),
    "distribute-balancing-warmup": (config.LINUX_COLOR, 11),
    "repl-distribute-warmup": (config.SPARE_COLOR, 13),
}

ARCH_ORDER = ["silver", "gold", "plat", "gold5320"]
SILVER_GOLD = [("silver", "Silver x2"), ("gold", "Gold x4")]

YLABEL_SIZE = 6.5
XTICK_SIZE = 6
XTICK_PAD = 2.0
BAR_WIDTH = 0.07
BAR_GAP = 0.0
X_SPACING = 0.29

ABS_FIGSIZE = (3.3, 1.4)
ABS_BAR_WIDTH = 0.1
ABS_PCT_SIZE = 3.0
ABS_HEADROOM = 1.45

HALO = [pe.withStroke(linewidth=1.2, foreground="white")]
VALUE_SIZE = 4.2  # tokens/s printed over each bar, stood up


def load() -> pd.DataFrame:
    rows = []
    for arch in os.listdir(RESULT_DIR):
        arch_dir = os.path.join(RESULT_DIR, arch, "llama")
        if not os.path.isdir(arch_dir):
            continue

        for name in CSVS:
            csv_path = os.path.join(arch_dir, f"{name}.csv")
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                rows.append(
                    {
                        "arch": arch,
                        "tag": row["tag"],
                        "test": row["test"],
                        "avg_ts": row["avg_ts"],
                        "stddev_ts": row["stddev_ts"],
                    }
                )
    return pd.DataFrame(rows)


def palette(tag: str):
    ramp, shade = TAG_SHADES[tag]
    return sns.color_palette(ramp, n_colors=RAMP_STEPS)[shade]


def make_plot_llama():
    os.makedirs(config.PLOT_DIR_LLAMA, exist_ok=True)

    df_all = load()
    if df_all.empty:
        return

    sns.set_style(style="ticks")
    sns.set_context("paper")
    plt.rcParams.update({"font.family": "serif", "font.serif": "DejaVu Serif"})

    ylim = _pct_ylim(df_all)
    leftmost = _sorted_archs(df_all)[0]
    for arch in df_all["arch"].unique():
        arch_data = df_all[df_all["arch"] == arch]
        plot_pct(arch, arch_data, ylim, ylabel=(arch == leftmost))
        plot_abs(arch, arch_data)
    plot_silver_gold(df_all, ylim)
    plot_legend(df_all)


# --- Data ---


def _value(arch_data: pd.DataFrame, test: str, tag: str, col: str) -> float:
    row = arch_data[(arch_data["test"] == test) & (arch_data["tag"] == tag)]
    return row.iloc[0][col] if len(row) > 0 else 0


def _pct_values(arch_data: pd.DataFrame, tag: str) -> tuple[list, list]:
    means, stds = [], []
    for test, _ in TESTS:
        base = _value(arch_data, test, BASELINE, "avg_ts")
        value = _value(arch_data, test, tag, "avg_ts")
        std = _value(arch_data, test, tag, "stddev_ts")
        if not base or not value:
            means.append(0)
            stds.append(0)
            continue
        means.append(100 * (value - base) / base)
        stds.append(100 * std / base)
    return means, stds


def _tags_present(arch_data: pd.DataFrame) -> list:
    present = set(arch_data["tag"])
    return [t for t in TAGS if t in present]


def _plotted_tags(arch_data: pd.DataFrame) -> list:
    return [t for t in _tags_present(arch_data) if t != BASELINE]


def _sorted_archs(df_all: pd.DataFrame) -> list:
    return sorted(
        df_all["arch"].unique(),
        key=lambda a: ARCH_ORDER.index(config.ARCH_SUBNAMES[a]),
    )


def _pct_ends(arch_data: pd.DataFrame) -> list:
    """Every error bar end, so a scale can cover them all."""
    ends = [0.0]
    for tag in _plotted_tags(arch_data):
        means, stds = _pct_values(arch_data, tag)
        ends += [m + s for m, s in zip(means, stds)]
        ends += [m - s for m, s in zip(means, stds)]
    return ends


def _pct_ylim(df_all: pd.DataFrame) -> tuple[float, float]:
    ends = []
    for arch in df_all["arch"].unique():
        ends += _pct_ends(df_all[df_all["arch"] == arch])
    lo, hi = min(ends), max(ends)
    return lo - 0.08 * (hi - lo), hi * 1.18


# --- Drawing ---


def _bar_positions(x, bar_index: int, n_bars: int, width: float) -> list:
    group_width = n_bars * width + (n_bars - 1) * BAR_GAP
    return [
        pos - group_width / 2 + bar_index * (width + BAR_GAP) + width / 2
        for pos in x
    ]


def _format_ax(ax, x, labels, y_nbins: int = 8, tick_fs=XTICK_SIZE):
    ax.set_axisbelow(True)
    ax.grid(axis="y", ls=":", lw=0.4, color="0.85", zorder=0)
    sns.despine(ax=ax)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.tick_params(axis="y", labelsize=5, length=2, width=0.8)
    ax.tick_params(axis="x", labelsize=6, length=2, width=0.8, pad=XTICK_PAD)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=tick_fs)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=y_nbins))


def _save(fig, path: str):
    plt.savefig(path, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close(fig)


def _pct_bars(ax, arch_data: pd.DataFrame):
    tags = _plotted_tags(arch_data)
    x = np.arange(N_TESTS) * X_SPACING
    ends = _pct_ends(arch_data)
    span = max(ends) - min(ends)

    for i, tag in enumerate(tags):
        means, stds = _pct_values(arch_data, tag)
        positions = _bar_positions(x, i, len(tags), BAR_WIDTH)
        ax.bar(
            positions,
            means,
            yerr=stds,
            width=BAR_WIDTH,
            label=TAG_LABELS[tag],
            capsize=0.8,
            linewidth=0,
            error_kw=dict(lw=0.4, capthick=0.4),
            color=palette(tag),
        )
        if tag != VALUE_TAG:
            continue
        for pos, (test, _), mean, std in zip(positions, TESTS, means, stds):
            value = _value(arch_data, test, tag, "avg_ts")
            if not value:
                continue
            # the error bar's cap, which a slightly negative bar still
            # pushes above zero
            top = max(mean + std, 0)
            ax.text(
                pos,
                top + span * 0.015,
                f"{value:,.0f} t/s",
                ha="left",
                va="bottom",
                rotation=45,
                rotation_mode="anchor",
                fontsize=VALUE_SIZE,
                zorder=3,
                path_effects=HALO,
            )
    ax.axhline(0, linestyle="--", color="gray", linewidth=0.3, alpha=0.25)
    return x


def plot_pct(arch: str, arch_data: pd.DataFrame, ylim, ylabel: bool = True):
    """One machine, both tests on one axis."""
    fig, ax = plt.subplots(figsize=(2.2, 1.35))
    x = _pct_bars(ax, arch_data)
    _format_ax(ax, x, TEST_NAMES)
    ax.set_ylim(*ylim)
    if ylabel:
        ax.set_ylabel(
            "Improvement over \nLinux Vanilla (%)", fontsize=YLABEL_SIZE
        )
    path = os.path.join(
        config.PLOT_DIR_LLAMA, f"{config.ARCH_SUBNAMES[arch]}_llama.pdf"
    )
    _save(fig, path)


def plot_silver_gold(df_all: pd.DataFrame, ylim):
    """Both machines side by side, on one shared scale."""
    by_sub = {config.ARCH_SUBNAMES[a]: a for a in df_all["arch"].unique()}
    panels = [(t, by_sub[sub]) for sub, t in SILVER_GOLD if sub in by_sub]
    if len(panels) < 2:
        return

    fig, axes = plt.subplots(
        1, len(panels), figsize=(3.3, 1.0), gridspec_kw={"wspace": 0.12}
    )
    for ax, (title, arch) in zip(axes, panels):
        x = _pct_bars(ax, df_all[df_all["arch"] == arch])
        _format_ax(ax, x, TEST_NAMES, y_nbins=6, tick_fs=5)
        ax.set_ylim(*ylim)
        ax.text(
            sum(ax.get_xlim()) / 2,
            -0.19,  # just under the tick labels, which end at -0.155
            title,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6.5,
        )
    axes[0].set_ylabel("Improvement over \nLinux Vanilla (%)", fontsize=5.5)
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)

    _save(fig, os.path.join(config.PLOT_DIR_LLAMA, "silver_gold.pdf"))


def plot_abs(arch: str, arch_data: pd.DataFrame):
    """Absolute throughput, one panel per test: they differ by ~10x."""
    tags = _tags_present(arch_data)
    x = [0]
    group_width = len(tags) * ABS_BAR_WIDTH + (len(tags) - 1) * BAR_GAP

    fig, axes = plt.subplots(
        1, N_TESTS, figsize=ABS_FIGSIZE, gridspec_kw={"wspace": 0.35}
    )
    for ti, (test, test_label) in enumerate(TESTS):
        ax = axes[ti]
        means = [_value(arch_data, test, t, "avg_ts") for t in tags]
        pcts = [_pct_values(arch_data, t)[0][ti] for t in tags]
        offset = (max(means) if means else 1) * 0.03

        for i, tag in enumerate(tags):
            bars = ax.bar(
                _bar_positions(x, i, len(tags), ABS_BAR_WIDTH),
                [means[i]],
                yerr=[_value(arch_data, test, tag, "stddev_ts")],
                width=ABS_BAR_WIDTH,
                label=TAG_LABELS[tag],
                capsize=0.6,
                linewidth=0,
                error_kw=dict(lw=0.3, capthick=0.3),
                color=palette(tag),
            )
            if tag == BASELINE or means[i] == 0:
                continue
            rect = bars[0]
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + offset,
                f"{pcts[i]:+.1f}%",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=ABS_PCT_SIZE,
                color="green" if pcts[i] > 0 else "red",
            )

        _format_ax(ax, x, [test_label], y_nbins=6)
        ax.set_ylim(top=max(means) * ABS_HEADROOM if means else None)
        ax.set_xlim(-group_width / 2 - 0.02, group_width / 2 + 0.02)

    axes[0].set_ylabel("Throughput (tokens/s)", fontsize=YLABEL_SIZE)
    path = os.path.join(
        config.PLOT_DIR_LLAMA, f"{config.ARCH_SUBNAMES[arch]}_llama_abs.pdf"
    )
    _save(fig, path)


def plot_legend(df_all: pd.DataFrame):
    """One legend for both figures, on its own."""
    tags = _plotted_tags(df_all)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=palette(t), linewidth=0)
        for t in tags
    ]
    fig = plt.figure(figsize=(3.3, 0.12))
    fig.legend(
        handles,
        [TAG_LABELS[t] for t in tags],
        loc="upper center",
        ncol=len(tags),
        fontsize=6,
        frameon=False,
        handlelength=1.2,
        handleheight=0.7,
        columnspacing=1.0,
        borderpad=0,
        borderaxespad=0,
        handletextpad=0.4,
    )
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    _save(fig, os.path.join(config.PLOT_DIR_LLAMA, "legend.pdf"))
