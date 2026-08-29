import os
import config
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import numpy as np
import pandas as pd


RESULT_DIR = config.RESULT_DIR

# prompt processing then token generation, llama-bench's two default tests
TESTS = [("pp512", "pp512"), ("tg128", "tg128")]
TEST_NAMES = [label for _, label in TESTS]
N_TESTS = len(TESTS)

# bench_llama writes one csv per kernel variant
CSVS = ["llama", "llama-repl"]

# percentages are relative to this, first one present wins
BASELINES = ["baseline-balancing"]
# and always to its warmed up run: one divisor for the whole figure, so bar
# heights compare across the cold/warm split, and NUMA Balancing is measured
# at its best
BASELINE_WARMUP = True

# left to right, baseline first
BASELINE_TAG = "baseline"  # Vanilla, the bar the values sit on
TAGS = [
    "baseline-balancing",
    "baseline",
    "distribute",
    "interleaved-distribute",
    "repl-distribute",
]
TAG_LABELS = {
    "baseline-balancing": "NUMA Balancing",
    "baseline": "Vanilla",
    "distribute": "Distribute",
    "interleaved-distribute": "Interleaved (Distr.)",
    "repl-distribute": "SPARe (Distr.)",
}

# each tag comes in a warmed-up and a cold variant (tag vs tag + suffix);
# plotted as a pair of bars, cold on the left, warm on the right. The csv
# calls it warmup, the paper calls it preloading the model.
WARMUP_SUFFIX = "-warmup"
WARMUP_STATES = [False, True]
WARMUP_LABEL = "Model preload"  # the hatched bar; the plain one is without

# panels left to right
ARCH_ORDER = ["silver", "gold", "plat", "gold5320"]

YLABEL_SIZE = 6.5
TITLE_SIZE = 6.5
TITLE_PAD = 1.5
XTICK_SIZE = 6
XTICK_PAD = 2.0
# each tag is a pair of bars (no warmup, warmup) that all but touch, with the
# tests set well apart
BAR_WIDTH = 0.035
BAR_GAP = 0.0
GROUP_GAP = 0.005
X_SPACING = 0.35

# the absolute plot has one group per panel instead of two, so it spreads the
# bars out and stands its percentages up to keep them readable
ABS_FIGSIZE = (3.3, 1.4)
ABS_BAR_WIDTH = 0.05
ABS_GROUP_GAP = 0.01
ABS_PCT_SIZE = 4.5

# the warmup twin, marked the way plot_fio_pgtable marks its replicated bars:
# a light, sparse hatch over a deeper shade of the same hue
HATCH = "///"
HATCH_COLOR = "0.85"
HATCH_LINEWIDTH = 0.85
# a white halo, so a value sitting on an error bar stays readable
HALO = [pe.withStroke(linewidth=1.2, foreground="white")]
VALUE_SIZE = 3.5  # tokens/s printed at the end of each bar, stood up
ABS_HEADROOM = 1.45  # top of the y axis, as a share of the tallest bar


def load():
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


# shades of one 17 step ramp: the cold bar, then two steps deeper for its
# warmup twin. Indices 2/5/8/11 are ann/rocksdb's linux shades exactly.
RAMP_STEPS = 17
TAG_SHADES = {
    "baseline": (config.LINUX_COLOR, 2),
    "distribute": (config.LINUX_COLOR, 5),
    "interleaved-distribute": (config.LINUX_COLOR, 8),
    "baseline-balancing": (config.LINUX_COLOR, 11),
    "repl-distribute": (config.SPARE_COLOR, 13),
}
WARMUP_STEPS = 2


def palette(tag: str, warmup: bool = False):
    ramp, shade = TAG_SHADES[tag]
    if warmup:
        shade += WARMUP_STEPS
    return sns.color_palette(ramp, n_colors=RAMP_STEPS)[shade]


def make_plot_llama():
    os.makedirs(config.PLOT_DIR_LLAMA, exist_ok=True)

    df_all = load()
    if df_all.empty:
        return

    sns.set_style(style="ticks")
    sns.set_context("paper")
    plt.rcParams.update({"font.family": "serif", "font.serif": "DejaVu Serif"})
    plt.rcParams["hatch.linewidth"] = HATCH_LINEWIDTH

    # the per-arch figures go side by side in the paper, so they share a
    # scale and only the leftmost one carries the y label
    ylim = _pct_ylim(df_all)
    leftmost = _sorted_archs(df_all)[0]
    for arch in df_all["arch"].unique():
        arch_data = df_all[df_all["arch"] == arch]
        plot_pct(arch, arch_data, ylim, ylabel=(arch == leftmost))
        plot_abs(arch, arch_data)
    plot_silver_gold(df_all, ylim)
    plot_legend(df_all)


# --- Plot helpers ---


def _tags_present(arch_data: pd.DataFrame) -> list:
    """Plot order, minus what this machine did not run (either variant)."""
    present = set(arch_data["tag"])
    return [t for t in TAGS if t in present or t + WARMUP_SUFFIX in present]


def _plotted_tags(arch_data: pd.DataFrame) -> list:
    """The bars: everything present but the baseline, which is the 0 line."""
    return [t for t in _tags_present(arch_data) if t != _baseline(arch_data)]


def _sorted_archs(df_all: pd.DataFrame) -> list:
    return sorted(
        df_all["arch"].unique(),
        key=lambda a: ARCH_ORDER.index(config.ARCH_SUBNAMES[a]),
    )


def _pct_ylim(df_all: pd.DataFrame) -> tuple[float, float]:
    """One scale for every machine, error bars included."""
    values = [0.0]
    for arch in df_all["arch"].unique():
        arch_data = df_all[df_all["arch"] == arch]
        for tag in _plotted_tags(arch_data):
            for warmup in WARMUP_STATES:
                means, stds = _pct_values(arch_data, tag, warmup)
                values += [m + s for m, s in zip(means, stds)]
                values += [m - s for m, s in zip(means, stds)]
    lo, hi = min(values), max(values)
    # room under 0 for small negative bars, over the top for the value labels
    return lo - 0.08 * (hi - lo), hi * 1.18


def _pct_ylim_span(arch_data: pd.DataFrame) -> float:
    """Height of the plotted range, to place labels a fixed fraction above."""
    values = [0.0]
    for tag in _plotted_tags(arch_data):
        for warmup in WARMUP_STATES:
            means, stds = _pct_values(arch_data, tag, warmup)
            values += [m + s for m, s in zip(means, stds)]
            values += [m - s for m, s in zip(means, stds)]
    return max(values) - min(values)


def _baseline(arch_data: pd.DataFrame) -> str:
    present = set(arch_data["tag"])
    return next(
        (t for t in BASELINES if t in present or t + WARMUP_SUFFIX in present),
        BASELINES[-1],
    )


def _bar_positions(
    x, bar_index: int, n_bars: int, width=BAR_WIDTH, gap=BAR_GAP
) -> list:
    group_width = n_bars * width + (n_bars - 1) * gap
    return [
        pos - group_width / 2 + bar_index * (width + gap) + width / 2
        for pos in x
    ]


def _paired_positions(
    x, pair_index: int, n_pairs: int, warmup_index: int, width: float, group_gap: float
) -> list:
    """Position within a row of cold/warm pairs: no gap inside a pair,
    group_gap between pairs."""
    pair_width = 2 * width
    total_width = n_pairs * pair_width + (n_pairs - 1) * group_gap
    start = pair_index * (pair_width + group_gap) - total_width / 2
    offset = start + warmup_index * width + width / 2
    return [pos + offset for pos in x]


def _value(
    arch_data: pd.DataFrame, test: str, tag: str, col: str, warmup: bool
) -> float:
    """One measurement, 0 when missing."""
    csv_tag = tag + WARMUP_SUFFIX if warmup else tag
    row = arch_data[(arch_data["test"] == test) & (arch_data["tag"] == csv_tag)]
    return row.iloc[0][col] if len(row) > 0 else 0


def _pct_values(
    arch_data: pd.DataFrame, tag: str, warmup: bool
) -> tuple[list, list]:
    """Throughput and stddev vs the one baseline run, in percent."""
    means, stds = [], []
    baseline_tag = _baseline(arch_data)
    for test, _ in TESTS:
        base = _value(arch_data, test, baseline_tag, "avg_ts", BASELINE_WARMUP)
        value = _value(arch_data, test, tag, "avg_ts", warmup)
        std = _value(arch_data, test, tag, "stddev_ts", warmup)
        if not base or not value:
            means.append(0)
            stds.append(0)
            continue
        means.append(100 * (value - base) / base)
        stds.append(100 * std / base)
    return means, stds


def _format_ax(ax, x, labels, y_nbins: int = 8, tick_fs=XTICK_SIZE):
    # faint dotted rules at each tick, like the pressure plot
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


# --- Plot functions ---


def _short_value(value: float) -> str:
    """Whole units, with a thousands separator: "21,999"."""
    return f"{value:,.0f}"


def _pct_bars(ax, arch_data: pd.DataFrame):
    """Improvement over the baseline, one group per test, cold+warm pairs."""
    tags = _plotted_tags(arch_data)
    x = np.arange(N_TESTS) * X_SPACING
    n_tags = len(tags)

    span = _pct_ylim_span(arch_data)

    for i, tag in enumerate(tags):
        for w, warmup in enumerate(WARMUP_STATES):
            means, stds = _pct_values(arch_data, tag, warmup)
            positions = _paired_positions(x, i, n_tags, w, BAR_WIDTH, GROUP_GAP)
            ax.bar(
                positions,
                means,
                yerr=stds,
                width=BAR_WIDTH,
                label=TAG_LABELS[tag] if warmup else None,
                capsize=0.8,
                linewidth=0,
                hatch=HATCH if warmup else None,
                error_kw=dict(lw=0.4, capthick=0.4),
                color=palette(tag, warmup),
                edgecolor=HATCH_COLOR,
            )
            # one throughput per test, off the baseline tag's cold bar: every
            # bar divides by the same run, so one anchor reads the whole group
            if tag != BASELINE_TAG or warmup:
                continue
            for pos, (test, _), mean, std in zip(positions, TESTS, means, stds):
                value = _value(arch_data, test, tag, "avg_ts", warmup)
                if not value:
                    continue
                # over the bar's top, error bar included
                top = (mean + std) if mean >= 0 else 0
                ax.text(
                    pos,
                    top + span * 0.015,
                    _short_value(value),
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
    fig, ax = plt.subplots(figsize=(2.2, 1.5))
    x = _pct_bars(ax, arch_data)
    _format_ax(ax, x, TEST_NAMES)
    ax.set_ylim(*ylim)
    if ylabel:
        ax.set_ylabel(
            "Improvement over \nNUMA Balancing (%)", fontsize=YLABEL_SIZE
        )
    path = os.path.join(
        config.PLOT_DIR_LLAMA, f"{config.ARCH_SUBNAMES[arch]}_llama.pdf"
    )
    _save(fig, path)


# silver left, gold right, in one compact figure
SILVER_GOLD = [("silver", "Silver x2"), ("gold", "Gold x4")]


def plot_silver_gold(df_all: pd.DataFrame, ylim):
    """Both machines side by side, on one shared scale."""
    by_sub = {config.ARCH_SUBNAMES[a]: a for a in df_all["arch"].unique()}
    panels = [(t, by_sub[sub]) for sub, t in SILVER_GOLD if sub in by_sub]
    if len(panels) < 2:
        return

    fig, axes = plt.subplots(
        1, len(panels), figsize=(3.3, 1.05), gridspec_kw={"wspace": 0.12}
    )
    for ax, (title, arch) in zip(axes, panels):
        x = _pct_bars(ax, df_all[df_all["arch"] == arch])
        _format_ax(ax, x, TEST_NAMES, y_nbins=6, tick_fs=5)
        ax.set_ylim(*ylim)
        ax.text(
            sum(ax.get_xlim()) / 2,
            -0.18,
            title,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6.5,
        )
    axes[0].set_ylabel(
        "Improvement over \nNUMA Balancing (%)", fontsize=5.5
    )
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)

    _save(fig, os.path.join(config.PLOT_DIR_LLAMA, "silver_gold.pdf"))


def plot_abs(arch: str, arch_data: pd.DataFrame):
    """Absolute throughput, one panel per test: they differ by ~10x."""
    tags = _tags_present(arch_data)
    x = [0]
    n_tags = len(tags)
    group_width = (
        n_tags * 2 * ABS_BAR_WIDTH + (n_tags - 1) * ABS_GROUP_GAP
    )

    fig, axes = plt.subplots(
        1, N_TESTS, figsize=ABS_FIGSIZE, gridspec_kw={"wspace": 0.35}
    )
    for ti, (test, test_label) in enumerate(TESTS):
        ax = axes[ti]
        pairs = [(tag, warmup) for tag in tags for warmup in WARMUP_STATES]
        means = [_value(arch_data, test, t, "avg_ts", w) for t, w in pairs]
        pcts = [_pct_values(arch_data, t, w)[0][ti] for t, w in pairs]
        offset = (max(means) if means else 1) * 0.03

        for i, (tag, warmup) in enumerate(pairs):
            tag_index, w = divmod(i, len(WARMUP_STATES))
            bars = ax.bar(
                _paired_positions(
                    x, tag_index, n_tags, w, ABS_BAR_WIDTH, ABS_GROUP_GAP
                ),
                [means[i]],
                yerr=[_value(arch_data, test, tag, "stddev_ts", warmup)],
                width=ABS_BAR_WIDTH,
                label=TAG_LABELS[tag] if warmup else None,
                capsize=0.6,
                linewidth=0,
                hatch=HATCH if warmup else None,
                error_kw=dict(lw=0.3, capthick=0.3),
                color=palette(tag, warmup),
                edgecolor=HATCH_COLOR,
            )
            # everything but the divisor itself carries its percentage
            is_divisor = tag == _baseline(arch_data) and warmup == BASELINE_WARMUP
            if is_divisor or means[i] == 0:
                continue
            rect = bars[0]
            # stood up: side by side they run into each other
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
        # one group only, so frame it by hand
        ax.set_xlim(-group_width / 2 - 0.02, group_width / 2 + 0.02)

    axes[0].set_ylabel("Throughput (tokens/s)", fontsize=YLABEL_SIZE)
    path = os.path.join(
        config.PLOT_DIR_LLAMA, f"{config.ARCH_SUBNAMES[arch]}_llama_abs.pdf"
    )
    _save(fig, path)


def plot_legend(df_all: pd.DataFrame):
    """One legend for both figures, on its own: policy colors, then the
    plain/hatched key for cold vs warm."""
    tags = _plotted_tags(df_all)
    color_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=palette(t), linewidth=0)
        for t in tags
    ]
    # only the hatch needs saying: a plain bar is the run without it
    warmup_handle = plt.Rectangle(
        (0, 0),
        1,
        1,
        facecolor="gray",
        edgecolor=HATCH_COLOR,
        linewidth=0,
        hatch=HATCH + HATCH,
    )
    # the policies on the first line, the warmup key on the second: the figure
    # height is the gap between them, the paddings are all trimmed away
    fig = plt.figure(figsize=(3.3, 0.19))
    style = dict(
        fontsize=6,
        frameon=False,
        handlelength=1.2,
        handleheight=0.7,
        columnspacing=1.0,
        borderpad=0,
        borderaxespad=0,
        handletextpad=0.4,
    )
    fig.legend(
        color_handles,
        [TAG_LABELS[t] for t in tags],
        loc="upper center",
        ncol=len(tags),
        **style,
    )
    fig.legend(
        [warmup_handle],
        [WARMUP_LABEL],
        loc="lower center",
        ncol=1,
        **style,
    )
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    _save(fig, os.path.join(config.PLOT_DIR_LLAMA, "legend.pdf"))
