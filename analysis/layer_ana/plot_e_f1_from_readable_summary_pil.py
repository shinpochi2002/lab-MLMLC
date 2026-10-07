from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_E_VALUES_CSV = Path(__file__).with_name("E_F1_layer_structure_for_paper.csv")

FIGURE_DPI = 300
FIGURE_WIDTH_PX = 2967
FIGURE_HEIGHT_PX = 1649
AXIS_LABEL_FONT_SIZE_PX = 65
TICK_FONT_SIZE_PX = 50
LEGEND_FONT_SIZE_PX = 50
LEGEND_TITLE_FONT_SIZE_PX = 52
ANNOTATION_FONT_SIZE_PX = 50

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

COLOR_MAP = {
    2: "#4C78A8",
    3: "#F58518",
    4: "#54A24B",
}

MARKER_MAP = {
    2: "circle",
    3: "square",
    4: "triangle",
}

LABEL_OFFSETS = [
    (38, -34),
    (38, 30),
    (-38, -34),
    (-38, 30),
    (0, -58),
    (0, 54),
    (78, 0),
    (-78, 0),
    (86, -46),
    (86, 46),
    (-86, -46),
    (-86, 46),
    (136, -18),
    (136, 26),
    (-136, -18),
    (-136, 26),
    (180, -65),
    (180, 65),
    (-180, -65),
    (-180, 65),
    (240, -100),
    (240, 100),
    (-240, -100),
    (-240, 100),
]

PREFERRED_LABEL_OFFSETS = {
    "layer3_4_4_8": (0, 36),
    "layer3_12_2_6": (-44, -40),
}

SUMMARY_COLUMNS = [
    "test_data",
    "configuration",
    "layer",
    "target",
    "layer1_category_num",
    "sample_count",
    "source_csv",
    "evaluation_data_path",
    "status",
    "seconds",
    "avg_precision",
    "avg_recall",
    "avg_f1",
    "avg_pred_apis_count",
    "avg_true_apis_count",
]

POINT_COLUMNS = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create Figure E style E-F1 Pareto plots from readable_model_summary.csv "
            "without pandas/matplotlib."
        )
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--e-values-csv", type=Path, default=DEFAULT_E_VALUES_CSV)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--test-data-name", default="overall_336")
    parser.add_argument("--png-width", type=int, default=FIGURE_WIDTH_PX)
    parser.add_argument("--png-height", type=int, default=FIGURE_HEIGHT_PX)
    parser.add_argument(
        "--e-smaller-is-better",
        action="store_true",
        help="Use smaller E as better for Pareto extraction. Default treats larger E as better.",
    )
    parser.add_argument(
        "--fixed-f1-axis",
        action="store_true",
        help="Use the paper's fixed F1 axis range instead of auto-scaling.",
    )
    parser.add_argument("--f1-min", type=float, default=0.25)
    parser.add_argument("--f1-max", type=float, default=0.75)
    return parser.parse_args()


def normalize_structure(value: object) -> str:
    return str(value).strip().replace("_", "-")


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def first_category_num(target: str) -> str:
    first = normalize_structure(target).split("-")[0]
    return first if first.isdigit() else ""


def as_float(value: object) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def as_int(value: object) -> int:
    return int(float(str(value)))


def load_layer1_eval_path(results_dir: Path) -> str:
    manifest_path = results_dir / "test_data_manifest.json"
    if not manifest_path.exists():
        return ""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest:
        if item.get("name") == "layer1_direct_api":
            return str(item.get("evaluation_data_path", ""))
    return ""


def convert_readable_summary(
    summary_csv: Path,
    results_dir: Path,
    test_data_name: str,
) -> list[dict[str, Any]]:
    rows = read_csv_dicts(summary_csv)
    eval_path = load_layer1_eval_path(results_dir)
    converted = []

    for row in rows:
        target = normalize_structure(row["分割パターン"])
        converted.append(
            {
                "test_data": test_data_name,
                "configuration": row["結果名"],
                "layer": as_int(row["階層数"]),
                "target": target,
                "layer1_category_num": first_category_num(target),
                "sample_count": as_int(row["サンプル数"]),
                "source_csv": str(summary_csv),
                "evaluation_data_path": eval_path,
                "status": row.get("status", ""),
                "seconds": row.get("実行時間秒", ""),
                "avg_precision": row.get("Macro_Precision平均", ""),
                "avg_recall": row.get("Macro_Recall平均", ""),
                "avg_f1": row.get("Macro_F1平均", ""),
                "avg_pred_apis_count": row.get("平均予測API数", ""),
                "avg_true_apis_count": row.get("平均正解API数", ""),
            }
        )

    return converted


def load_e_values(path: Path) -> dict[str, dict[str, Any]]:
    values = {}
    for row in read_csv_dicts(path):
        structure = normalize_structure(row["category_structure"])
        values[structure] = {
            "plot_stages": as_int(row["stages"]),
            "E_category_layers": as_int(row["category_layers"]),
            "E": as_float(row["E"]),
        }
    return values


def build_points(
    summary_rows: list[dict[str, Any]],
    e_values: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    points = []
    missing_e = []
    missing_labels = []

    for row in summary_rows:
        if row["configuration"] == "layer1_direct_api":
            continue
        if str(row.get("status", "")).strip() != "ok":
            continue

        category_layers = as_int(row["layer"])
        summary_structure = normalize_structure(row["target"])
        e_structure = E_STRUCTURE_ALIASES.get(summary_structure, summary_structure)
        e_row = e_values.get(e_structure)
        if e_row is None:
            missing_e.append((row["configuration"], summary_structure, e_structure))
            continue
        display_label = DISPLAY_STRUCTURE_LABELS.get(e_structure)
        if display_label is None:
            missing_labels.append((row["configuration"], summary_structure, e_structure))
            continue

        plot_stages = as_int(e_row["plot_stages"])
        point = {
            "test_data": row["test_data"],
            "configuration": row["configuration"],
            "summary_structure": summary_structure,
            "E_structure": e_structure,
            "display_structure_plot": display_label,
            "used_alias": summary_structure != e_structure,
            "category_layers": category_layers,
            "plot_stages": plot_stages,
            "expected_plot_stages": category_layers + 1,
            "stage_matches": plot_stages == category_layers + 1,
            "E_category_layer_matches": as_int(e_row["E_category_layers"]) == category_layers,
            "E": as_float(e_row["E"]),
            "F1": as_float(row["avg_f1"]),
            "avg_precision": as_float(row["avg_precision"]),
            "avg_recall": as_float(row["avg_recall"]),
            "sample_count": as_int(row["sample_count"]),
        }
        points.append(point)

    if missing_e:
        details = "\n".join(f"{a},{b},{c}" for a, b, c in missing_e)
        raise ValueError(f"No E value found for:\n{details}")
    if missing_labels:
        details = "\n".join(f"{a},{b},{c}" for a, b, c in missing_labels)
        raise ValueError(f"No display label found for:\n{details}")

    mismatches = [p for p in points if not p["stage_matches"]]
    if mismatches:
        details = "\n".join(
            f'{p["configuration"]}: category_layers={p["category_layers"]}, '
            f'plot_stages={p["plot_stages"]}'
            for p in mismatches
        )
        raise ValueError(f"Layer/stage mismatch found:\n{details}")

    return sorted(points, key=lambda p: (p["test_data"], p["category_layers"], p["E"], p["configuration"]))


def get_pareto_front(
    rows: list[dict[str, Any]],
    *,
    e_smaller_is_better: bool,
) -> list[dict[str, Any]]:
    front = []
    for row_i in rows:
        dominated = False
        for row_j in rows:
            if row_i is row_j:
                continue
            if e_smaller_is_better:
                x_not_worse = row_j["E"] <= row_i["E"]
                x_strict_better = row_j["E"] < row_i["E"]
            else:
                x_not_worse = row_j["E"] >= row_i["E"]
                x_strict_better = row_j["E"] > row_i["E"]

            y_not_worse = row_j["F1"] >= row_i["F1"]
            y_strict_better = row_j["F1"] > row_i["F1"]
            if x_not_worse and y_not_worse and (x_strict_better or y_strict_better):
                dominated = True
                break
        if not dominated:
            front.append(row_i)
    return sorted(front, key=lambda p: p["E"])


def nice_step(span: float, target_ticks: int = 6) -> float:
    if span <= 0:
        return 1.0
    raw = span / target_ticks
    power = 10 ** math.floor(math.log10(raw))
    for multiplier in (1, 2, 2.5, 5, 10):
        step = multiplier * power
        if raw <= step:
            return step
    return 10 * power


def nice_ticks(vmin: float, vmax: float, target_ticks: int = 6) -> list[float]:
    step = nice_step(vmax - vmin, target_ticks)
    start = math.ceil(vmin / step) * step
    ticks = []
    value = start
    while value <= vmax + step * 0.5:
        ticks.append(round(value, 10))
        value += step
    return ticks


def auto_limits(values: list[float], pad_fraction: float = 0.12) -> tuple[float, float]:
    vmin = min(values)
    vmax = max(values)
    span = vmax - vmin
    if span == 0:
        span = max(abs(vmin), 1.0) * 0.1
    return vmin - span * pad_fraction, vmax + span * pad_fraction


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def text_bbox(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    *,
    stroke_width: int = 0,
) -> tuple[float, float, float, float]:
    bbox = draw.textbbox(xy, text, font=font, stroke_width=stroke_width)
    return float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])


def pad_bbox(
    bbox: tuple[float, float, float, float],
    pad: float,
) -> tuple[float, float, float, float]:
    return bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad


def overlap_area(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def outside_plot_penalty(
    bbox: tuple[float, float, float, float],
    plot_bbox: tuple[float, float, float, float],
) -> float:
    dx = max(plot_bbox[0] - bbox[0], 0.0) + max(bbox[2] - plot_bbox[2], 0.0)
    dy = max(plot_bbox[1] - bbox[1], 0.0) + max(bbox[3] - plot_bbox[3], 0.0)
    return dx + dy


def choose_label_anchor(
    x: float,
    y: float,
    dx: int,
    dy: int,
    text_width: int,
    text_height: int,
) -> tuple[float, float]:
    if dx < 0:
        anchor_x = x + dx - text_width
    elif dx == 0:
        anchor_x = x - text_width / 2
    else:
        anchor_x = x + dx

    if dy < 0:
        anchor_y = y + dy - text_height
    elif dy == 0:
        anchor_y = y - text_height / 2
    else:
        anchor_y = y + dy

    return anchor_x, anchor_y


def draw_layer_pattern_labels(
    draw: ImageDraw.ImageDraw,
    rows: list[dict[str, Any]],
    point_positions: dict[str, tuple[float, float]],
    *,
    font: ImageFont.ImageFont,
    plot_bbox: tuple[float, float, float, float],
    avoid_bboxes: list[tuple[float, float, float, float]] | None = None,
) -> None:
    occupied: list[tuple[float, float, float, float]] = list(avoid_bboxes or [])
    for x, y in point_positions.values():
        occupied.append((x - 36, y - 36, x + 36, y + 36))

    center_x = (plot_bbox[0] + plot_bbox[2]) / 2
    center_y = (plot_bbox[1] + plot_bbox[3]) / 2

    def placement_order(row: dict[str, Any]) -> tuple[int, float]:
        x, y = point_positions[row["configuration"]]
        priority = 0 if row["configuration"] in PREFERRED_LABEL_OFFSETS else 1
        return priority, abs(x - center_x) + abs(y - center_y)

    ordered = sorted(
        rows,
        key=placement_order,
    )

    placements = []
    for row in ordered:
        x, y = point_positions[row["configuration"]]
        text = str(row["display_structure_plot"])
        text_width, text_height = text_size(draw, text, font)
        best = None

        offsets = []
        preferred_offset = PREFERRED_LABEL_OFFSETS.get(row["configuration"])
        if preferred_offset is not None:
            offsets.append(preferred_offset)
        offsets.extend(LABEL_OFFSETS)

        for dx, dy in offsets:
            anchor = choose_label_anchor(x, y, dx, dy, text_width, text_height)
            bbox = pad_bbox(
                text_bbox(draw, anchor, text, font, stroke_width=3),
                5,
            )
            overlap = sum(overlap_area(bbox, other) for other in occupied)
            outside = outside_plot_penalty(bbox, plot_bbox)
            line_end_x = min(max(x, bbox[0]), bbox[2])
            line_end_y = min(max(y, bbox[1]), bbox[3])
            distance = math.hypot(line_end_x - x, line_end_y - y)
            score = (outside, overlap, distance)
            if best is None or score < best[0]:
                best = (score, anchor, bbox)
        _, anchor, bbox = best
        placements.append((x, y, anchor, bbox, text))
        occupied.append(bbox)

    for x, y, _, bbox, _ in placements:
        line_end_x = min(max(x, bbox[0]), bbox[2])
        line_end_y = min(max(y, bbox[1]), bbox[3])
        dx = line_end_x - x
        dy = line_end_y - y
        distance = math.hypot(dx, dy)
        if distance == 0:
            continue
        marker_clearance = min(36.0, distance * 0.45)
        line_start_x = x + dx / distance * marker_clearance
        line_start_y = y + dy / distance * marker_clearance
        draw.line(
            (line_start_x, line_start_y, line_end_x, line_end_y),
            fill="#666666",
            width=3,
        )

    for _, _, anchor, _, text in placements:
        draw.text(
            anchor,
            text,
            font=font,
            fill="#111111",
            stroke_width=3,
            stroke_fill="white",
        )


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
    *,
    stroke_width: int = 0,
    stroke_fill: str | None = None,
) -> None:
    width, height = text_size(draw, text, font)
    draw.text(
        (xy[0] - width / 2, xy[1] - height / 2),
        text,
        font=font,
        fill=fill,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
    )


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    fill: str,
    width: int,
    dash: int = 16,
    gap: int = 10,
) -> None:
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    distance = math.hypot(dx, dy)
    if distance == 0:
        return
    ux = dx / distance
    uy = dy / distance
    pos = 0.0
    while pos < distance:
        seg_end = min(pos + dash, distance)
        draw.line(
            [(x1 + ux * pos, y1 + uy * pos), (x1 + ux * seg_end, y1 + uy * seg_end)],
            fill=fill,
            width=width,
        )
        pos += dash + gap


def draw_marker(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    marker: str,
    fill: str,
    *,
    size: int = 27,
    outline: str | None = None,
    width: int = 2,
) -> None:
    r = size
    if marker == "square":
        points = [(x - r, y - r), (x + r, y + r)]
        draw.rectangle(points, fill=fill, outline=outline, width=width)
    elif marker == "triangle":
        points = [(x, y - r * 1.15), (x - r * 1.12, y + r * 0.9), (x + r * 1.12, y + r * 0.9)]
        draw.polygon(points, fill=fill)
        if outline:
            draw.line(points + [points[0]], fill=outline, width=width)
    else:
        draw.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline=outline, width=width)


def draw_hollow_marker(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    marker: str,
    *,
    size: int = 34,
    outline: str = "#111111",
    width: int = 4,
) -> None:
    r = size
    if marker == "square":
        draw.rectangle((x - r, y - r, x + r, y + r), outline=outline, width=width)
    elif marker == "triangle":
        points = [(x, y - r * 1.15), (x - r * 1.12, y + r * 0.9), (x + r * 1.12, y + r * 0.9)]
        draw.line(points + [points[0]], fill=outline, width=width)
    else:
        draw.ellipse((x - r, y - r, x + r, y + r), outline=outline, width=width)


def plot_points(
    rows: list[dict[str, Any]],
    pareto_rows: list[dict[str, Any]],
    output_base: Path,
    *,
    width: int,
    height: int,
    fixed_f1_axis: bool,
    f1_limits: tuple[float, float],
) -> None:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    font_tick = load_font(TICK_FONT_SIZE_PX)
    font_label = load_font(AXIS_LABEL_FONT_SIZE_PX)
    font_legend = load_font(LEGEND_FONT_SIZE_PX)
    font_legend_title = load_font(LEGEND_TITLE_FONT_SIZE_PX, bold=True)
    font_point_label = load_font(ANNOTATION_FONT_SIZE_PX)

    left = 235
    right = width - 105
    top = 105
    bottom = height - 190
    plot_width = right - left
    plot_height = bottom - top

    x_values = [row["E"] for row in rows]
    y_values = [row["F1"] for row in rows]
    x_min, x_max = auto_limits(x_values, 0.08)
    if fixed_f1_axis:
        y_min, y_max = f1_limits
    else:
        y_min, y_max = auto_limits(y_values, 0.16)

    def x_to_px(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def y_to_px(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * plot_height

    grid_color = "#DADDE2"
    axis_color = "#222222"
    tick_color = "#333333"

    x_ticks = nice_ticks(x_min, x_max, 7)
    y_ticks = nice_ticks(y_min, y_max, 6)

    for tick in x_ticks:
        x = x_to_px(tick)
        draw.line((x, top, x, bottom), fill=grid_color, width=2)
        draw.line((x, bottom, x, bottom + 10), fill=axis_color, width=3)
        label = f"{tick:.2f}".rstrip("0").rstrip(".")
        tw, _ = text_size(draw, label, font_tick)
        draw.text((x - tw / 2, bottom + 24), label, font=font_tick, fill=tick_color)

    for tick in y_ticks:
        y = y_to_px(tick)
        draw.line((left, y, right, y), fill=grid_color, width=2)
        draw.line((left - 10, y, left, y), fill=axis_color, width=3)
        label = f"{tick:.2f}".rstrip("0").rstrip(".")
        tw, th = text_size(draw, label, font_tick)
        draw.text((left - 22 - tw, y - th / 2), label, font=font_tick, fill=tick_color)

    draw.line((left, bottom, right, bottom), fill=axis_color, width=4)
    draw.line((left, top, left, bottom), fill=axis_color, width=4)

    x_label = "Structural descriptor E"
    tw, th = text_size(draw, x_label, font_label)
    draw.text((left + plot_width / 2 - tw / 2, height - 82), x_label, font=font_label, fill="#111111")

    y_label = "Final API recommendation F1"
    y_width, y_height = text_size(draw, y_label, font_label)
    y_text = Image.new("RGBA", (y_width + 12, y_height + 12), (255, 255, 255, 0))
    y_draw = ImageDraw.Draw(y_text)
    y_draw.text((6, 6), y_label, font=font_label, fill="#111111")
    y_text = y_text.crop(y_text.getbbox()).rotate(90, expand=True)
    image.paste(y_text, (45, top + plot_height // 2 - y_text.height // 2), y_text)

    sorted_pareto = sorted(pareto_rows, key=lambda row: row["E"])
    for a, b in zip(sorted_pareto, sorted_pareto[1:]):
        draw_dashed_line(
            draw,
            (x_to_px(a["E"]), y_to_px(a["F1"])),
            (x_to_px(b["E"]), y_to_px(b["F1"])),
            fill="#222222",
            width=5,
        )

    point_positions: dict[str, tuple[float, float]] = {}
    for row in sorted(rows, key=lambda p: p["category_layers"]):
        category_layers = int(row["category_layers"])
        marker = MARKER_MAP.get(category_layers, "circle")
        color = COLOR_MAP.get(category_layers, "#777777")
        x = x_to_px(row["E"])
        y = y_to_px(row["F1"])
        point_positions[row["configuration"]] = (x, y)
        draw_marker(draw, x, y, marker, color, size=27)

    pareto_keys = {(row["configuration"], row["E"], row["F1"]) for row in pareto_rows}
    for row in rows:
        if (row["configuration"], row["E"], row["F1"]) not in pareto_keys:
            continue
        category_layers = int(row["category_layers"])
        draw_hollow_marker(
            draw,
            x_to_px(row["E"]),
            y_to_px(row["F1"]),
            MARKER_MAP.get(category_layers, "circle"),
        )

    legend_width = 540
    legend_rows = []
    for category_layers in sorted({int(row["category_layers"]) for row in rows}):
        plot_stages = category_layers + 1
        legend_rows.append((category_layers, f"{plot_stages}-layers"))
    legend_height = 70 + 56 * (len(legend_rows) + 1)
    legend_x = right - legend_width - 35
    legend_y = top + 32
    legend_bbox = (legend_x - 20, legend_y - 18, legend_x + legend_width, legend_y + legend_height)
    draw.rounded_rectangle(
        legend_bbox,
        radius=8,
        fill=(255, 255, 255),
        outline="#D2D2D2",
        width=2,
    )
    draw.text((legend_x, legend_y), "Number of layers", font=font_legend_title, fill="#111111")
    cy = legend_y + 66
    for category_layers, label in legend_rows:
        marker = MARKER_MAP.get(category_layers, "circle")
        color = COLOR_MAP.get(category_layers, "#777777")
        draw_marker(draw, legend_x + 22, cy + 15, marker, color, size=15)
        draw.text((legend_x + 56, cy), label, font=font_legend, fill="#111111")
        cy += 56
    draw_dashed_line(
        draw,
        (legend_x + 5, cy + 18),
        (legend_x + 42, cy + 18),
        fill="#222222",
        width=3,
        dash=9,
        gap=6,
    )
    draw.text((legend_x + 56, cy), "Pareto front", font=font_legend, fill="#111111")

    draw_layer_pattern_labels(
        draw,
        rows,
        point_positions,
        font=font_point_label,
        plot_bbox=(left, top, right, bottom),
        avoid_bboxes=[pad_bbox(legend_bbox, 10)],
    )

    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    image.save(png_path, dpi=(FIGURE_DPI, FIGURE_DPI))
    image.save(pdf_path, "PDF", resolution=float(FIGURE_DPI))


def write_outputs(
    summary_rows: list[dict[str, Any]],
    points: list[dict[str, Any]],
    output_dir: Path,
    *,
    e_smaller_is_better: bool,
    width: int,
    height: int,
    fixed_f1_axis: bool,
    f1_limits: tuple[float, float],
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []

    summary_path = output_dir / "overall_336_summary_for_pareto.csv"
    write_csv(summary_path, summary_rows, SUMMARY_COLUMNS)
    written.append(summary_path)

    all_points_path = output_dir / "all_test_data_points_with_E.csv"
    all_point_columns = POINT_COLUMNS + [
        "expected_plot_stages",
        "stage_matches",
        "E_category_layer_matches",
    ]
    write_csv(all_points_path, points, all_point_columns)
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
    seen = set()
    layer_check = []
    for point in points:
        key = tuple(point[col] for col in layer_check_cols)
        if key in seen:
            continue
        seen.add(key)
        layer_check.append(point)
    layer_check_path = output_dir / "layer_mapping_check.csv"
    write_csv(layer_check_path, layer_check, layer_check_cols)
    written.append(layer_check_path)

    test_names = sorted({str(point["test_data"]) for point in points})
    for test_name in test_names:
        group = [point.copy() for point in points if str(point["test_data"]) == test_name]
        group.sort(key=lambda p: (p["E"], p["F1"], p["configuration"]))
        safe_name = test_name.replace("/", "_")
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
        points_path = output_dir / f"{safe_name}_E_F1_points.csv"
        write_csv(points_path, group, point_cols)
        written.append(points_path)

        pareto = get_pareto_front(group, e_smaller_is_better=e_smaller_is_better)
        pareto_path = output_dir / f"{safe_name}_E_F1_pareto_front.csv"
        write_csv(pareto_path, pareto, point_cols)
        written.append(pareto_path)

        figure_base = output_dir / f"{safe_name}_E_F1_scatter_with_pareto"
        plot_points(
            group,
            pareto,
            figure_base,
            width=width,
            height=height,
            fixed_f1_axis=fixed_f1_axis,
            f1_limits=f1_limits,
        )
        written.extend([figure_base.with_suffix(".png"), figure_base.with_suffix(".pdf")])

    return written


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    summary_csv = args.summary_csv or results_dir / "readable_model_summary.csv"
    output_dir = args.output_dir or results_dir / "e_f1_pareto_latest_auto_y"

    if not summary_csv.exists():
        raise FileNotFoundError(summary_csv)
    if not args.e_values_csv.exists():
        raise FileNotFoundError(args.e_values_csv)

    summary_rows = convert_readable_summary(summary_csv, results_dir, args.test_data_name)
    e_values = load_e_values(args.e_values_csv)
    points = build_points(summary_rows, e_values)
    written = write_outputs(
        summary_rows,
        points,
        output_dir,
        e_smaller_is_better=args.e_smaller_is_better,
        width=args.png_width,
        height=args.png_height,
        fixed_f1_axis=args.fixed_f1_axis,
        f1_limits=(args.f1_min, args.f1_max),
    )

    print("Layer check: all rows satisfy plot_stages = category_layers + 1.")
    print(f"Input summary CSV: {summary_csv}")
    print(f"E values CSV     : {args.e_values_csv}")
    print(f"Output dir       : {output_dir}")
    print("F1 axis limits   : fixed" if args.fixed_f1_axis else "F1 axis limits   : auto")
    print(f"Test data patterns: {args.test_data_name}")
    print(f"Files written    : {len(written)}")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
