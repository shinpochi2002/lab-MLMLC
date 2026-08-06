from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.transforms import Bbox
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_CSV = (
    PROJECT_ROOT
    / "codex/results/hierarchy_existing_models_condition_2x2/"
    / "metric_summaries_by_test_data/all_test_data_summary.csv"
)
DEFAULT_E_VALUES_CSV = Path(__file__).with_name("E_F1_layer_structure_for_paper.csv")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("pareto_by_test_data")
DEFAULT_F1_MIN = 0.25
DEFAULT_F1_MAX = 0.75


# The processed result names are the experiment pattern names.  The E table in
# layer_ana stores the rounded branching structure used in the original plots.
E_STRUCTURE_ALIASES = {
    "4-43": "4-38",
    "6-25": "6-26",
    "33-4": "33-5",
    "4-4-8": "4-4-10",
    "8-4-4": "8-4-5",
    "12-2-6": "12-2-7",
    "12-4-3": "12-4-4",
    "12-6-2": "12-6-3",
    "12-2-2-3": "12-2-2-4",
    "8-3-2-3": "8-3-2-4",
}


DISPLAY_STRUCTURE_LABELS = {
    "4-38": "4-152-909",
    "6-26": "6-152-909",
    "8-19": "8-152-909",
    "12-13": "12-152-909",
    "15-10": "15-152-909",
    "24-7": "24-152-909",
    "33-5": "33-152-909",
    "51-3": "51-152-909",
    "4-4-10": "4-16-152-909",
    "8-4-5": "8-32-152-909",
    "12-2-7": "12-24-152-909",
    "12-4-4": "12-48-152-909",
    "12-6-3": "12-72-152-909",
    "16-4-3": "16-64-152-909",
    "12-2-2-4": "12-24-48-152-909",
    "8-3-2-4": "8-24-48-152-909",
}


MARKER_MAP = {
    2: "o",
    3: "s",
    4: "^",
}

COLOR_MAP = {
    2: "#4C78A8",
    3: "#F58518",
    4: "#54A24B",
}

LABEL_OFFSETS = [
    (4, 4),
    (4, -6),
    (-4, 4),
    (-4, -6),
    (8, 14),
    (8, -16),
    (-8, 14),
    (-8, -16),
    (18, 6),
    (-18, 6),
    (18, -8),
    (-18, -8),
    (26, 16),
    (-26, 16),
    (26, -20),
    (-26, -20),
    (0, 24),
    (0, -26),
    (36, 0),
    (-36, 0),
    (42, 16),
    (-42, 16),
    (42, -20),
    (-42, -20),
]
LABEL_PADDING_PIXELS = 2
LABEL_FONT_SIZE = 7


def normalize_structure(value: object) -> str:
    return str(value).strip().replace("_", "-")


def get_png_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def align_for_offset(dx: int, dy: int) -> tuple[str, str]:
    if dx < 0:
        ha = "right"
    elif dx > 0:
        ha = "left"
    else:
        ha = "center"

    if dy < 0:
        va = "top"
    elif dy > 0:
        va = "bottom"
    else:
        va = "center"

    return ha, va


def pad_bbox(bbox: Bbox, pad: float = LABEL_PADDING_PIXELS) -> Bbox:
    return Bbox.from_extents(
        bbox.x0 - pad,
        bbox.y0 - pad,
        bbox.x1 + pad,
        bbox.y1 + pad,
    )


def overlap_area(a: Bbox, b: Bbox) -> float:
    x0 = max(a.x0, b.x0)
    y0 = max(a.y0, b.y0)
    x1 = min(a.x1, b.x1)
    y1 = min(a.y1, b.y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def outside_axes_penalty(bbox: Bbox, axes_bbox: Bbox) -> float:
    dx = max(axes_bbox.x0 - bbox.x0, 0.0) + max(bbox.x1 - axes_bbox.x1, 0.0)
    dy = max(axes_bbox.y0 - bbox.y0, 0.0) + max(bbox.y1 - axes_bbox.y1, 0.0)
    return dx + dy


def arrowprops_for_offset(dx: int, dy: int) -> dict[str, object] | None:
    if abs(dx) + abs(dy) < 28:
        return None
    return {
        "arrowstyle": "-",
        "color": "0.45",
        "linewidth": 0.4,
        "shrinkA": 0,
        "shrinkB": 4,
    }


def add_label_candidate(ax, text: str, x: float, y: float, dx: int, dy: int):
    ha, va = align_for_offset(dx, dy)
    return ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=LABEL_FONT_SIZE,
        ha=ha,
        va=va,
        annotation_clip=False,
        clip_on=False,
        arrowprops=arrowprops_for_offset(dx, dy),
    )


def place_non_overlapping_labels(
    ax,
    fig,
    label_rows: list[tuple[float, float, str]],
    *,
    avoid_artists: list[object] | None = None,
) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer=renderer).expanded(1.02, 1.06)

    occupied: list[Bbox] = []
    for artist in avoid_artists or []:
        try:
            occupied.append(pad_bbox(artist.get_window_extent(renderer)))
        except Exception:
            pass

    for x, y, _ in label_rows:
        px, py = ax.transData.transform((x, y))
        occupied.append(Bbox.from_extents(px - 5, py - 5, px + 5, py + 5))

    # Place the more central labels first; edge labels have more room outside.
    center_x = sum(ax.get_xlim()) / 2
    center_y = sum(ax.get_ylim()) / 2
    label_rows = sorted(
        label_rows,
        key=lambda item: abs(item[0] - center_x) + abs(item[1] - center_y),
    )

    for x, y, text in label_rows:
        best = None
        for dx, dy in LABEL_OFFSETS:
            ann = add_label_candidate(ax, text, x, y, dx, dy)
            fig.canvas.draw()
            bbox = pad_bbox(ann.get_window_extent(renderer))
            overlap = sum(overlap_area(bbox, other) for other in occupied)
            outside = outside_axes_penalty(bbox, axes_bbox)
            distance = abs(dx) + abs(dy)
            score = (overlap, outside * 100.0, distance)
            ann.remove()

            if best is None or score < best[0]:
                best = (score, dx, dy, bbox)
            if overlap == 0 and outside == 0:
                break

        _, dx, dy, _ = best
        ann = add_label_candidate(ax, text, x, y, dx, dy)
        fig.canvas.draw()
        occupied.append(pad_bbox(ann.get_window_extent(renderer)))


def get_pareto_front(
    df: pd.DataFrame,
    x_col: str = "E",
    y_col: str = "F1",
    *,
    x_smaller_is_better: bool = False,
    y_larger_is_better: bool = True,
) -> pd.DataFrame:
    pareto_idx = []

    for i, row_i in df.iterrows():
        dominated = False
        for j, row_j in df.iterrows():
            if i == j:
                continue

            if x_smaller_is_better:
                x_not_worse = row_j[x_col] <= row_i[x_col]
                x_strict_better = row_j[x_col] < row_i[x_col]
            else:
                x_not_worse = row_j[x_col] >= row_i[x_col]
                x_strict_better = row_j[x_col] > row_i[x_col]

            if y_larger_is_better:
                y_not_worse = row_j[y_col] >= row_i[y_col]
                y_strict_better = row_j[y_col] > row_i[y_col]
            else:
                y_not_worse = row_j[y_col] <= row_i[y_col]
                y_strict_better = row_j[y_col] < row_i[y_col]

            if x_not_worse and y_not_worse and (x_strict_better or y_strict_better):
                dominated = True
                break

        if not dominated:
            pareto_idx.append(i)

    return df.loc[pareto_idx].sort_values(x_col).reset_index(drop=True)


def load_points(summary_csv: Path, e_values_csv: Path) -> pd.DataFrame:
    summary_df = pd.read_csv(summary_csv)
    e_df = pd.read_csv(e_values_csv)

    required_summary_cols = {
        "test_data",
        "configuration",
        "layer",
        "target",
        "status",
        "avg_f1",
    }
    required_e_cols = {"stages", "category_layers", "category_structure", "E"}

    missing_summary = required_summary_cols - set(summary_df.columns)
    missing_e = required_e_cols - set(e_df.columns)
    if missing_summary:
        raise ValueError(f"Missing summary columns: {sorted(missing_summary)}")
    if missing_e:
        raise ValueError(f"Missing E-value columns: {sorted(missing_e)}")

    points = summary_df.copy()
    points = points[points["configuration"] != "layer1_direct_api"].copy()
    points = points[points["status"].fillna("") == "ok"].copy()

    points["category_layers"] = points["layer"].astype(int)
    points["summary_structure"] = points["target"].map(normalize_structure)
    points["E_structure"] = points["summary_structure"].replace(E_STRUCTURE_ALIASES)
    points["F1"] = points["avg_f1"].astype(float)

    e_values = e_df.copy()
    e_values["E_structure"] = e_values["category_structure"].map(normalize_structure)
    e_values = e_values[["E_structure", "stages", "category_layers", "E"]].rename(
        columns={
            "stages": "plot_stages",
            "category_layers": "E_category_layers",
        }
    )
    e_values["plot_stages"] = e_values["plot_stages"].astype(int)
    e_values["E_category_layers"] = e_values["E_category_layers"].astype(int)
    e_values["E"] = e_values["E"].astype(float)

    merged = points.merge(e_values, on="E_structure", how="left", validate="many_to_one")
    missing_e = merged[merged["E"].isna()]
    if not missing_e.empty:
        missing = (
            missing_e[["configuration", "summary_structure", "E_structure"]]
            .drop_duplicates()
            .sort_values("configuration")
        )
        raise ValueError(
            "No E value found for these structures:\n"
            + missing.to_string(index=False)
        )

    merged["expected_plot_stages"] = merged["category_layers"] + 1
    merged["stage_matches"] = merged["plot_stages"] == merged["expected_plot_stages"]
    merged["E_category_layer_matches"] = (
        merged["E_category_layers"] == merged["category_layers"]
    )
    merged["used_alias"] = merged["summary_structure"] != merged["E_structure"]
    merged["display_structure_plot"] = merged["E_structure"].map(DISPLAY_STRUCTURE_LABELS)

    missing_labels = merged[merged["display_structure_plot"].isna()]
    if not missing_labels.empty:
        missing = (
            missing_labels[["configuration", "summary_structure", "E_structure"]]
            .drop_duplicates()
            .sort_values("configuration")
        )
        raise ValueError(
            "No display label found for these structures:\n"
            + missing.to_string(index=False)
        )

    return merged.sort_values(["test_data", "category_layers", "E", "configuration"])


def plot_test_data(
    df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    output_base: Path,
    *,
    e_smaller_is_better: bool,
    f1_limits: tuple[float, float] | None,
    label_style: str,
    target_png_size: tuple[int, int] | None,
    png_dpi: int,
    numbered_plot_right: float,
    show_number_key: bool,
) -> None:
    is_numbered = label_style in {"numbered", "numbered_marker"}
    is_marker_numbered = label_style == "numbered_marker"
    if target_png_size is None:
        fig_width = 13.25 if is_numbered and show_number_key else 10
        fig_size = (fig_width, 5.625)
    else:
        fig_size = (target_png_size[0] / png_dpi, target_png_size[1] / png_dpi)
    fig, ax = plt.subplots(figsize=fig_size)
    label_rows: list[tuple[float, float, str]] = []
    marker_label_rows: list[tuple[float, float, str]] = []

    for category_layers in sorted(df["category_layers"].unique()):
        group = df[df["category_layers"] == category_layers].sort_values("E")
        plot_stages = int(group["plot_stages"].iloc[0])
        label = f"{plot_stages}-layers"
        ax.scatter(
            group["E"],
            group["F1"],
            label=label,
            marker=MARKER_MAP.get(category_layers, "o"),
            color=COLOR_MAP.get(category_layers),
            s=170 if is_marker_numbered else 64,
            alpha=0.9,
        )

        for _, row in group.iterrows():
            label_text = (
                str(int(row["label_id"]))
                if is_numbered
                else f'({row["display_structure_plot"]})'
            )
            target_rows = marker_label_rows if is_marker_numbered else label_rows
            target_rows.append((float(row["E"]), float(row["F1"]), label_text))

    ax.plot(
        pareto_df["E"],
        pareto_df["F1"],
        linestyle="--",
        linewidth=1.5,
        color="#222222",
        label="Pareto front",
    )
    ax.scatter(
        pareto_df["E"],
        pareto_df["F1"],
        s=195 if is_marker_numbered else 130,
        facecolors="none",
        edgecolors="#111111",
        linewidths=1.2,
    )

    if is_marker_numbered:
        for x, y, text in marker_label_rows:
            ax.text(
                x,
                y,
                text,
                ha="center",
                va="center",
                fontsize=6.5,
                fontweight="bold",
                color="black",
                path_effects=[
                    path_effects.withStroke(linewidth=1.4, foreground="white")
                ],
                zorder=6,
            )

    ax.set_xlabel(r"Design indicator $E$")
    ax.set_ylabel("Final API recommendation F1")
    if f1_limits is not None:
        ax.set_ylim(*f1_limits)
    legend = ax.legend(title="Number of layers", fontsize=8)
    ax.grid(True, alpha=0.25)

    key_artist = None
    if is_numbered and show_number_key:
        layout_rect = (0.04, 0.06, numbered_plot_right, 0.96) if target_png_size else (0, 0, numbered_plot_right, 1)
        fig.tight_layout(rect=layout_rect)
        key_df = df.sort_values("label_id")
        key_lines = ["ID  structure", "--  ------------------"]
        key_lines.extend(
            f'{int(row["label_id"]):>2}  {row["display_structure_plot"]}'
            for _, row in key_df.iterrows()
        )
        key_artist = ax.text(
            1.03,
            0.5,
            "\n".join(key_lines),
            transform=ax.transAxes,
            va="center",
            ha="left",
            fontsize=8,
            family="monospace",
            linespacing=1.25,
        )
    else:
        if target_png_size:
            fig.tight_layout(rect=(0.04, 0.06, 0.98, 0.96))
        else:
            fig.tight_layout()

    avoid_artists = [legend]
    if key_artist is not None:
        avoid_artists.append(key_artist)
    if label_rows:
        place_non_overlapping_labels(ax, fig, label_rows, avoid_artists=avoid_artists)

    if target_png_size is None:
        fig.savefig(output_base.with_suffix(".png"), dpi=png_dpi, bbox_inches="tight")
        fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    else:
        fig.savefig(output_base.with_suffix(".png"), dpi=png_dpi)
        fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)


def write_outputs(
    points: pd.DataFrame,
    output_dir: Path,
    *,
    e_smaller_is_better: bool,
    f1_limits: tuple[float, float] | None,
    label_style: str,
    target_png_size: tuple[int, int] | None,
    png_dpi: int,
    numbered_plot_right: float,
    show_number_key: bool,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    all_points_path = output_dir / "all_test_data_points_with_E.csv"
    points.to_csv(all_points_path, index=False)
    written.append(all_points_path)

    layer_check_cols = [
        "configuration",
        "summary_structure",
        "E_structure",
        "display_structure_plot",
        "category_layers",
        "plot_stages",
        "expected_plot_stages",
        "stage_matches",
        "E_category_layer_matches",
        "used_alias",
    ]
    layer_check = points[layer_check_cols].drop_duplicates()
    layer_check_path = output_dir / "layer_mapping_check.csv"
    layer_check.to_csv(layer_check_path, index=False)
    written.append(layer_check_path)

    for test_data, group in points.groupby("test_data", sort=True):
        safe_name = str(test_data).replace("/", "_")
        point_cols = [
            "test_data",
            "configuration",
            "summary_structure",
            "E_structure",
            "display_structure_plot",
            "used_alias",
            "category_layers",
            "plot_stages",
            "E",
            "F1",
            "avg_precision",
            "avg_recall",
            "sample_count",
        ]
        group_out = (
            group[point_cols]
            .sort_values(["E", "F1", "configuration"])
            .reset_index(drop=True)
        )
        group_out.insert(1, "label_id", range(1, len(group_out) + 1))

        points_path = output_dir / f"{safe_name}_E_F1_points.csv"
        group_out.to_csv(points_path, index=False)
        written.append(points_path)

        pareto_df = get_pareto_front(
            group_out,
            x_col="E",
            y_col="F1",
            x_smaller_is_better=e_smaller_is_better,
        )
        pareto_path = output_dir / f"{safe_name}_E_F1_pareto_front.csv"
        pareto_df.to_csv(pareto_path, index=False)
        written.append(pareto_path)

        figure_base = output_dir / f"{safe_name}_E_F1_scatter_with_pareto"
        plot_test_data(
            group_out,
            pareto_df,
            figure_base,
            e_smaller_is_better=e_smaller_is_better,
            f1_limits=f1_limits,
            label_style=label_style,
            target_png_size=target_png_size,
            png_dpi=png_dpi,
            numbered_plot_right=numbered_plot_right,
            show_number_key=show_number_key,
        )
        written.extend([figure_base.with_suffix(".png"), figure_base.with_suffix(".pdf")])

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot E-F1 Pareto fronts for each processed test-data pattern."
    )
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--e-values-csv", type=Path, default=DEFAULT_E_VALUES_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--f1-min", type=float, default=DEFAULT_F1_MIN)
    parser.add_argument("--f1-max", type=float, default=DEFAULT_F1_MAX)
    parser.add_argument("--png-dpi", type=int, default=300)
    parser.add_argument(
        "--numbered-plot-right",
        type=float,
        default=None,
        help="Right edge of the plot area for numbered-label figures. Larger values widen the graph.",
    )
    parser.add_argument(
        "--match-reference-png",
        type=Path,
        default=None,
        help="Save PNGs with the same pixel size as this reference image.",
    )
    parser.add_argument(
        "--auto-f1-axis",
        action="store_true",
        help="Use matplotlib auto-scaling for the F1 axis, matching the original output.",
    )
    parser.add_argument(
        "--label-style",
        choices=["structure", "numbered", "numbered_marker"],
        default="structure",
        help="Use full labels, offset numeric labels, or numeric labels inside colored markers.",
    )
    parser.add_argument(
        "--hide-number-key",
        action="store_true",
        help="Do not draw the right-side ID-to-structure key for numbered figures.",
    )
    parser.add_argument(
        "--e-smaller-is-better",
        action="store_true",
        help=(
            "Treat smaller E as better for Pareto extraction. The default matches "
            "layer_ana_pare.py, which treats larger E as better."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.auto_f1_axis and args.f1_min >= args.f1_max:
        raise ValueError("--f1-min must be smaller than --f1-max")

    f1_limits = None if args.auto_f1_axis else (args.f1_min, args.f1_max)
    target_png_size = (
        get_png_size(args.match_reference_png)
        if args.match_reference_png is not None
        else None
    )
    numbered_plot_right = (
        args.numbered_plot_right
        if args.numbered_plot_right is not None
        else (0.56 if target_png_size is not None else 0.68)
    )
    if not 0.3 <= numbered_plot_right <= 0.9:
        raise ValueError("--numbered-plot-right must be between 0.3 and 0.9")

    points = load_points(args.summary_csv, args.e_values_csv)

    if not points["stage_matches"].all():
        mismatches = points[~points["stage_matches"]][
            ["configuration", "category_layers", "plot_stages", "expected_plot_stages"]
        ].drop_duplicates()
        raise ValueError(
            "Layer/stage mismatch found:\n" + mismatches.to_string(index=False)
        )

    written = write_outputs(
        points,
        args.output_dir,
        e_smaller_is_better=args.e_smaller_is_better,
        f1_limits=f1_limits,
        label_style=args.label_style,
        target_png_size=target_png_size,
        png_dpi=args.png_dpi,
        numbered_plot_right=numbered_plot_right,
        show_number_key=not args.hide_number_key,
    )

    print("Layer check: all rows satisfy plot_stages = category_layers + 1.")
    print(f"Input summary CSV: {args.summary_csv}")
    print(f"E values CSV     : {args.e_values_csv}")
    print(f"Output dir       : {args.output_dir}")
    if f1_limits is None:
        print("F1 axis limits   : auto")
    else:
        print(f"F1 axis limits   : {f1_limits[0]} to {f1_limits[1]}")
    print(f"Label style      : {args.label_style}")
    if target_png_size is not None:
        print(f"PNG size         : {target_png_size[0]} x {target_png_size[1]}")
    if args.label_style == "numbered":
        print(f"Plot right edge  : {numbered_plot_right}")
    print(f"Test data patterns: {', '.join(sorted(points['test_data'].unique()))}")
    print(f"Files written    : {len(written)}")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
