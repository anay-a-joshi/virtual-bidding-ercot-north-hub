# ==============================================================================
# Name    : Anay Abhijit Joshi
# Date    : November 22, 2026
# Course  : ISyE 6767 - Interim Project 2 (Virtual Bidding at ERCOT HB_NORTH)
# File    : neon_plots.py
# Purpose : One shared neon theme for every figure of the project, so that all
#           plots in the LaTeX report look like they belong to the same family.
# ==============================================================================

import matplotlib

matplotlib.use("Agg")  # So, figures are written to disk, no window pops up.
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from config import OUTPUT_FIGURE_DIR, ensure_output_folders

# Let's do now the palette. The order is FIXED: series 1 is always cyan, series 2 magenta, ...
# (checked with a colour-blind separation validator against the dark surface below).
NEON_CYAN = "#00E5FF"
NEON_MAGENTA = "#FF2BD6"
NEON_LIME = "#B6FF3B"
NEON_ORANGE = "#FF9F1C"
NEON_VIOLET = "#9D7BFF"
NEON_YELLOW = "#FFE14D"
NEON_SERIES_ORDER = (NEON_CYAN, NEON_MAGENTA, NEON_LIME, NEON_ORANGE, NEON_VIOLET, NEON_YELLOW)

# Surfaces and ink. In my opinion text must never wear a series colour, it stays light grey.
NEON_SURFACE = "#0B0F1A"
NEON_GRID_INK = "#24304A"
NEON_TEXT_INK = "#E6EDF7"
NEON_MUTED_INK = "#8A97B0"


def apply_neon_theme():
    # So, every rcParam that controls the look is set once, here.
    plt.rcParams.update({
        "figure.facecolor": NEON_SURFACE,
        "axes.facecolor": NEON_SURFACE,
        "savefig.facecolor": NEON_SURFACE,
        "axes.edgecolor": NEON_GRID_INK,
        "axes.labelcolor": NEON_TEXT_INK,
        "axes.titlecolor": NEON_TEXT_INK,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11.5,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": NEON_GRID_INK,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.55,
        "xtick.color": NEON_MUTED_INK,
        "ytick.color": NEON_MUTED_INK,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "text.color": NEON_TEXT_INK,
        "legend.facecolor": "#121a2b",
        "legend.edgecolor": NEON_GRID_INK,
        "legend.labelcolor": NEON_TEXT_INK,
        "legend.fontsize": 10,
        "legend.framealpha": 0.92,
        "font.family": "DejaVu Sans",
        "text.parse_math": False,       # So, "$" is always a dollar sign, never maths markup.
        "figure.dpi": 110,
        "savefig.dpi": 300,
    })


def neon_color(series_position):
    # Finally, colour follows the series position in the fixed order (never cycled past 6).
    if series_position < 0 or series_position >= len(NEON_SERIES_ORDER):
        raise ValueError(f"Oops! Only {len(NEON_SERIES_ORDER)} neon series colours exist, got {series_position}.")
    return NEON_SERIES_ORDER[series_position]


def new_neon_canvas(n_rows=1, n_cols=1, fig_width=11.0, fig_height=5.2, share_x=False):
    # So, every figure starts here, so the theme is always applied first.
    apply_neon_theme()
    neon_figure, neon_axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), sharex=share_x)
    return neon_figure, neon_axes


def glow_line(target_axis, x_values, y_values, line_color, line_label=None, line_width=1.6,
              glow_layers=3, line_alpha=1.0, z_order=3):
    # Let's do now the glow: the same line drawn wider and fainter underneath, then the crisp core.
    for glow_step in range(glow_layers, 0, -1):
        target_axis.plot(x_values, y_values, color=line_color, linewidth=line_width + 2.4 * glow_step,
                         alpha=0.06 * line_alpha, solid_capstyle="round", zorder=z_order - 1)
    return target_axis.plot(x_values, y_values, color=line_color, linewidth=line_width, alpha=line_alpha,
                            label=line_label, solid_capstyle="round", zorder=z_order)[0]


def glow_bars(target_axis, x_positions, bar_heights, bar_color, bar_label=None, bar_width=0.8):
    # So, a soft halo bar behind each crisp bar gives the neon look without hiding the value.
    target_axis.bar(x_positions, bar_heights, width=bar_width * 1.12, color=bar_color, alpha=0.18, zorder=2)
    return target_axis.bar(x_positions, bar_heights, width=bar_width, color=bar_color, alpha=0.92,
                           edgecolor=NEON_SURFACE, linewidth=1.0, label=bar_label, zorder=3)


def glow_scatter(target_axis, x_values, y_values, dot_color, dot_label=None, dot_size=18):
    target_axis.scatter(x_values, y_values, s=dot_size * 4, color=dot_color, alpha=0.10, linewidths=0, zorder=2)
    return target_axis.scatter(x_values, y_values, s=dot_size, color=dot_color, alpha=0.95,
                               edgecolors=NEON_SURFACE, linewidths=0.5, label=dot_label, zorder=3)


def zero_reference_line(target_axis, orientation="h", reference_value=0.0):
    # In my opinion a muted dashed zero line is the most useful reference on any spread plot.
    if orientation == "h":
        target_axis.axhline(reference_value, color=NEON_MUTED_INK, linewidth=0.9, linestyle="--", alpha=0.8, zorder=1)
    else:
        target_axis.axvline(reference_value, color=NEON_MUTED_INK, linewidth=0.9, linestyle="--", alpha=0.8, zorder=1)


def neon_note(target_axis, note_text, x_frac=0.01, y_frac=0.97):
    # So, a small muted annotation box in the corner (used for n, period, source).
    target_axis.text(x_frac, y_frac, note_text, transform=target_axis.transAxes, ha="left", va="top",
                     fontsize=9, color=NEON_MUTED_INK,
                     bbox=dict(boxstyle="round,pad=0.35", facecolor="#121a2b", edgecolor=NEON_GRID_INK, alpha=0.9))


def format_date_axis(target_axis, month_step=3):
    target_axis.xaxis.set_major_locator(mdates.MonthLocator(interval=month_step))
    target_axis.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))


def label_axis(target_axis, axis_title, x_label, y_label, show_legend=True, legend_place="best"):
    # Finally, every axis gets a title, both axis labels, and a legend when there is >1 series.
    target_axis.set_title(axis_title, pad=10)
    target_axis.set_xlabel(x_label)
    target_axis.set_ylabel(y_label)
    if show_legend:
        handles_found, labels_found = target_axis.get_legend_handles_labels()
        if len(labels_found) > 0:
            target_axis.legend(loc=legend_place)


def save_neon_figure(neon_figure, file_stem, super_title=None):
    # So, one call saves the PNG (300 dpi) used by the LaTeX report and closes the figure.
    ensure_output_folders()
    if super_title is not None:
        neon_figure.suptitle(super_title, color=NEON_TEXT_INK, fontsize=15, fontweight="bold")
    neon_figure.tight_layout()
    saved_path = OUTPUT_FIGURE_DIR / f"{file_stem}.png"
    neon_figure.savefig(saved_path, bbox_inches="tight")
    plt.close(neon_figure)
    return saved_path


def plain_log_axis(target_axis, which_axis="y"):
    # So, with maths markup switched off the default log labels would print "10^{-4}" as raw code;
    # plain numbers (0.0001, 0.001, ...) are clearer anyway.
    from matplotlib.ticker import FuncFormatter, NullFormatter
    chosen_axis = target_axis.yaxis if which_axis == "y" else target_axis.xaxis
    chosen_axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.6g}"))
    chosen_axis.set_minor_formatter(NullFormatter())


def symlog_ticks_dollars(target_axis, linear_threshold=10.0, lowest_value=-3000, highest_value=3000,
                         axis_limits=None):
    # Let's do now a symmetric-log y axis: heavy-tailed $ spreads stay readable near zero AND at $3,000.
    target_axis.set_yscale("symlog", linthresh=linear_threshold)
    tick_candidates = np.array([-3000, -1000, -300, -100, -30, -10, 0, 10, 30, 100, 300, 1000, 3000])
    tick_candidates = tick_candidates[(tick_candidates >= lowest_value) & (tick_candidates <= highest_value)]
    target_axis.set_yticks(tick_candidates)
    target_axis.set_yticklabels([f"{int(v):,}" for v in tick_candidates])
    if axis_limits is not None:
        target_axis.set_ylim(*axis_limits)


def cumulative_profit_panels(panel_series, panel_titles, figure_title, file_stem, n_columns=3,
                             reference_series=None, reference_label="Always INC"):
    # So, small multiples: one panel per family of strategies (never more than 6 colours in one panel),
    # with the same reference strategy drawn as a muted dashed line in every panel.
    n_panels = len(panel_series)
    n_rows = int(np.ceil(n_panels / n_columns))
    neon_figure, axis_grid = new_neon_canvas(n_rows, n_columns, fig_width=5.6 * n_columns, fig_height=4.3 * n_rows)
    axis_list = np.atleast_1d(axis_grid).ravel()
    shared_low, shared_high = 0.0, 0.0
    for each_series_list in panel_series:
        for _, daily_series in each_series_list:
            shared_low = min(shared_low, float(daily_series.cumsum().min()))
            shared_high = max(shared_high, float(daily_series.cumsum().max()))
    if reference_series is not None:
        shared_low = min(shared_low, float(reference_series.cumsum().min()))
        shared_high = max(shared_high, float(reference_series.cumsum().max()))
    padding = 0.06 * (shared_high - shared_low + 1.0)
    for panel_axis, each_series_list, panel_title in zip(axis_list, panel_series, panel_titles):
        if reference_series is not None:
            panel_axis.plot(reference_series.index, reference_series.cumsum(), color=NEON_MUTED_INK, linewidth=1.2,
                            linestyle="--", label=reference_label, zorder=2)
        for series_position, (series_label, daily_series) in enumerate(each_series_list):
            glow_line(panel_axis, daily_series.index, daily_series.cumsum(), neon_color(series_position), series_label,
                      1.5, glow_layers=2)
        zero_reference_line(panel_axis)
        panel_axis.set_ylim(shared_low - padding, shared_high + padding)
        panel_axis.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
        panel_axis.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
        panel_axis.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        label_axis(panel_axis, panel_title, "Delivery date", "Cumulative profit ($)", legend_place="upper left")
        panel_axis.legend(loc="upper left", fontsize=8.5)
    for unused_axis in axis_list[n_panels:]:
        unused_axis.set_visible(False)
    return save_neon_figure(neon_figure, file_stem, super_title=figure_title)
