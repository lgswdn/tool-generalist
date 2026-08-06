#!/usr/bin/env python3
"""Plot name-derived category counts for an object-selection JSON file.

The script has no third-party dependencies. It writes SVG directly so it also
works in lightweight environments where matplotlib/numpy are unavailable.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_INPUT = Path(
    "configs/object_selections/"
    "panda_general_dpoc_gg_no_high_conf_free_but_high_conf_colliding.json"
)
DEFAULT_OUTPUT_DIR = Path("scripts/outputs/object_selection_categories_1450")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top", type=int, default=30)
    return parser.parse_args()


def normalize_category(value: str) -> str:
    """Normalize case/separators while preserving the category wording."""
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    value = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    # The two datasets spell this same category as cellphone and CellPhone.
    aliases = {
        "cell_phone": "cellphone",
        "mug_new": "mug",
        "rubik_cube": "rubiks_cube",
    }
    return aliases.get(value, value)


def object_category(name: str) -> str:
    """Extract the explicit category-like portion of a DGN object name.

    core/sem names have a standardized category field. The gd and YCB subsets
    also encode a category after their dataset prefix. KIT, BigBIRD and MuJoCo
    names are product/model names without a stable category field, so their
    remaining name is retained instead of assigning a guessed semantic label.
    """
    match = re.match(r"^(?:core|sem)-([^-]+)-", name)
    if match:
        return normalize_category(match.group(1))

    match = re.match(r"^ddg-gd_(.+?)_poisson_\d+$", name)
    if match:
        return normalize_category(match.group(1))

    match = re.match(r"^ddg-ycb_\d+(?:-[a-z])?_(.+)$", name)
    if match:
        return normalize_category(match.group(1))

    for prefix in ("ddg-kit_", "ddg-bigbird_", "mujoco-"):
        if name.startswith(prefix):
            return normalize_category(name[len(prefix) :])

    return normalize_category(name)


def source_family(name: str) -> str:
    if name.startswith("core-"):
        return "core"
    if name.startswith("sem-"):
        return "sem"
    if name.startswith("ddg-"):
        return name.split("_", 1)[0]
    if name.startswith("mujoco-"):
        return "mujoco"
    return "unknown"


def nice_tick_max(maximum: int) -> tuple[int, int]:
    if maximum <= 50:
        step = 10
    elif maximum <= 100:
        step = 20
    elif maximum <= 250:
        step = 50
    else:
        step = 100
    return ((maximum + step - 1) // step) * step, step


def write_svg(
    path: Path,
    rows: list[tuple[str, int, str]],
    *,
    total: int,
    title: str,
    subtitle: str,
) -> None:
    width = 1500
    left = 430
    right = 145
    top = 150
    bottom = 80
    row_height = 29
    plot_width = width - left - right
    height = top + bottom + row_height * len(rows)
    max_value = max(count for _, count, _ in rows)
    tick_max, tick_step = nice_tick_max(max_value)

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        "<style>",
        "text { font-family: Inter, 'Noto Sans CJK SC', 'Microsoft YaHei', sans-serif; fill: #172033; }",
        ".title { font-size: 30px; font-weight: 700; }",
        ".subtitle { font-size: 16px; fill: #596579; }",
        ".label { font-size: 15px; }",
        ".value { font-size: 14px; font-weight: 650; }",
        ".tick { font-size: 13px; fill: #718096; }",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#fbfcfe"/>',
        f'<text x="45" y="50" class="title">{html.escape(title)}</text>',
        f'<text x="45" y="83" class="subtitle">{html.escape(subtitle)}</text>',
    ]

    plot_bottom = top + row_height * len(rows)
    for tick in range(0, tick_max + 1, tick_step):
        x = left + plot_width * tick / tick_max
        parts.append(
            f'<line x1="{x:.1f}" y1="{top - 15}" x2="{x:.1f}" '
            f'y2="{plot_bottom}" stroke="#dfe5ec" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{plot_bottom + 28}" text-anchor="middle" '
            f'class="tick">{tick}</text>'
        )

    colors = {
        "category": "#3977d4",
        "remaining": "#8a98aa",
        "singleton": "#e79a3b",
    }
    for index, (label, count, kind) in enumerate(rows):
        y = top + index * row_height
        bar_y = y + 4
        bar_height = 20
        bar_width = max(2.0, plot_width * count / tick_max)
        text_y = y + 19
        percent = count / total * 100
        parts.extend(
            [
                (
                    f'<text x="{left - 14}" y="{text_y}" text-anchor="end" '
                    f'class="label">{html.escape(label)}</text>'
                ),
                (
                    f'<rect x="{left}" y="{bar_y}" width="{bar_width:.1f}" '
                    f'height="{bar_height}" rx="3" fill="{colors[kind]}"/>'
                ),
                (
                    f'<text x="{left + bar_width + 9:.1f}" y="{text_y}" '
                    f'class="value">{count}  ({percent:.1f}%)</text>'
                ),
            ]
        )

    parts.extend(
        [
            (
                f'<text x="{left + plot_width / 2:.1f}" y="{height - 18}" '
                'text-anchor="middle" class="subtitle">Object count</text>'
            ),
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_png(
    path: Path,
    rows: list[tuple[str, int, str]],
    *,
    total: int,
    title: str,
    subtitle: str,
) -> bool:
    """Write a PNG when Pillow is available; SVG remains the baseline output."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return False

    width = 1500
    left = 430
    right = 145
    top = 150
    bottom = 80
    row_height = 29
    plot_width = width - left - right
    height = top + bottom + row_height * len(rows)
    max_value = max(count for _, count, _ in rows)
    tick_max, tick_step = nice_tick_max(max_value)
    regular_font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    bold_font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    fonts = {
        "title": ImageFont.truetype(bold_font, 30),
        "subtitle": ImageFont.truetype(regular_font, 16),
        "label": ImageFont.truetype(regular_font, 15),
        "value": ImageFont.truetype(bold_font, 14),
        "tick": ImageFont.truetype(regular_font, 13),
    }
    image = Image.new("RGB", (width, height), "#fbfcfe")
    draw = ImageDraw.Draw(image)
    draw.text((45, 25), title, fill="#172033", font=fonts["title"])
    draw.text((45, 67), subtitle, fill="#596579", font=fonts["subtitle"])

    plot_bottom = top + row_height * len(rows)
    for tick in range(0, tick_max + 1, tick_step):
        x = left + plot_width * tick / tick_max
        draw.line((x, top - 15, x, plot_bottom), fill="#dfe5ec", width=1)
        tick_text = str(tick)
        box = draw.textbbox((0, 0), tick_text, font=fonts["tick"])
        draw.text(
            (x - (box[2] - box[0]) / 2, plot_bottom + 10),
            tick_text,
            fill="#718096",
            font=fonts["tick"],
        )

    colors = {
        "category": "#3977d4",
        "remaining": "#8a98aa",
        "singleton": "#e79a3b",
    }
    for index, (label, count, kind) in enumerate(rows):
        y = top + index * row_height
        bar_y = y + 4
        bar_height = 20
        bar_width = max(2.0, plot_width * count / tick_max)
        label_box = draw.textbbox((0, 0), label, font=fonts["label"])
        label_width = label_box[2] - label_box[0]
        draw.text(
            (left - 14 - label_width, y + 3),
            label,
            fill="#172033",
            font=fonts["label"],
        )
        draw.rounded_rectangle(
            (left, bar_y, left + bar_width, bar_y + bar_height),
            radius=3,
            fill=colors[kind],
        )
        draw.text(
            (left + bar_width + 9, y + 3),
            f"{count}  ({count / total * 100:.1f}%)",
            fill="#172033",
            font=fonts["value"],
        )

    axis_label = "Object count"
    box = draw.textbbox((0, 0), axis_label, font=fonts["subtitle"])
    draw.text(
        (left + (plot_width - (box[2] - box[0])) / 2, height - 30),
        axis_label,
        fill="#596579",
        font=fonts["subtitle"],
    )
    image.save(path)
    return True


def main() -> None:
    args = parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"Expected a JSON list: {args.input}")

    counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    sources: dict[str, Counter[str]] = defaultdict(Counter)
    for row in data:
        name = str(row["object"])
        category = object_category(name)
        counts[category] += 1
        sources[category][source_family(name)] += 1
        if len(examples[category]) < 3:
            examples[category].append(name)

    total = sum(counts.values())
    if total != len(data):
        raise AssertionError("Category counts do not cover every input object")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "category_counts.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["category", "count", "share_percent", "source_counts", "examples"])
        for category, count in counts.most_common():
            source_text = "; ".join(
                f"{source}={source_count}"
                for source, source_count in sources[category].most_common()
            )
            writer.writerow(
                [
                    category,
                    count,
                    f"{count / total * 100:.4f}",
                    source_text,
                    "; ".join(examples[category]),
                ]
            )

    recurring = [(category, count) for category, count in counts.most_common() if count >= 2]
    singleton_count = sum(count for count in counts.values() if count == 1)
    top_rows = recurring[: args.top]
    remaining_rows = recurring[args.top :]
    overview_rows = [(category, count, "category") for category, count in top_rows]
    if remaining_rows:
        overview_rows.append(
            (
                f"other recurring categories ({len(remaining_rows)} types)",
                sum(count for _, count in remaining_rows),
                "remaining",
            )
        )
    if singleton_count:
        overview_rows.append(
            (f"singleton categories ({singleton_count} types)", singleton_count, "singleton")
        )
    overview_rows.sort(key=lambda item: item[1], reverse=True)

    common_subtitle = (
        f"{total} objects | {len(counts)} name-derived categories | "
        "case/CamelCase variants merged (e.g. core-cellphone + sem-CellPhone)"
    )
    overview_title = f"Object category composition (Top {args.top} of {total} objects)"
    for extension, writer in (("svg", write_svg), ("png", write_png)):
        writer(
            args.output_dir / f"category_composition_overview.{extension}",
            overview_rows,
            total=total,
            title=overview_title,
            subtitle=common_subtitle,
        )

    full_rows = [(category, count, "category") for category, count in recurring]
    if singleton_count:
        full_rows.append(
            (f"singleton categories ({singleton_count} types)", singleton_count, "singleton")
        )
    full_rows.sort(key=lambda item: item[1], reverse=True)
    full_title = f"Recurring category composition ({total} objects total)"
    full_subtitle = common_subtitle + " | singleton categories aggregated in orange"
    for extension, writer in (("svg", write_svg), ("png", write_png)):
        writer(
            args.output_dir / f"category_composition_recurring_full.{extension}",
            full_rows,
            total=total,
            title=full_title,
            subtitle=full_subtitle,
        )

    metadata = {
        "input": str(args.input),
        "object_count": total,
        "category_count": len(counts),
        "recurring_category_count": len(recurring),
        "singleton_category_count": singleton_count,
        "top": args.top,
        "classification": {
            "core_sem": "category token between dataset prefix and object id",
            "ddg_gd": "text between ddg-gd_ and _poisson_<id>",
            "ddg_ycb": "description after the YCB inventory number",
            "other_product_datasets": "full product/model name retained; no semantic guess",
            "normalization": "lowercase snake_case; cell_phone aliases to cellphone",
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"objects={total}")
    print(f"categories={len(counts)}")
    print(f"recurring_categories={len(recurring)}")
    print(f"singleton_categories={singleton_count}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
