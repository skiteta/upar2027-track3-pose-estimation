"""Deterministic public-training-data visualizations for local validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .constants import IGNORED_KEYPOINT_INDICES, KEYPOINT_NAMES

# Zero-based form of the organizer-provided PoseTrack18-compatible skeleton.
SKELETON: tuple[tuple[int, int], ...] = (
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 5),
    (1, 6),
    (1, 2),
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
)

GT_COLOR = "#009E73"
PREDICTION_COLOR = "#CC79A7"
BBOX_COLOR = "#E69F00"
TEXT_COLOR = "#202124"
MUTED_COLOR = "#5F6368"


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / filename
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default()


def _scaled_keypoints(
    values: Sequence[float], *, left: float, top: float, scale_x: float, scale_y: float
) -> list[tuple[float, float, float]]:
    if len(values) != 3 * len(KEYPOINT_NAMES):
        raise ValueError(f"Expected {3 * len(KEYPOINT_NAMES)} keypoint values, got {len(values)}")
    return [
        (
            left + float(values[index * 3]) * scale_x,
            top + float(values[index * 3 + 1]) * scale_y,
            float(values[index * 3 + 2]),
        )
        for index in range(len(KEYPOINT_NAMES))
    ]


def _draw_pose(
    draw: ImageDraw.ImageDraw,
    keypoints: Sequence[tuple[float, float, float]],
    *,
    color: str,
    width: int,
    radius: int,
) -> None:
    ignored = set(IGNORED_KEYPOINT_INDICES)
    for first, second in SKELETON:
        if first in ignored or second in ignored:
            continue
        x1, y1, v1 = keypoints[first]
        x2, y2, v2 = keypoints[second]
        if v1 > 0 and v2 > 0:
            draw.line((x1, y1, x2, y2), fill=color, width=width)
    for index, (x, y, visibility) in enumerate(keypoints):
        if index in ignored or visibility <= 0:
            continue
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=color,
            outline="white",
            width=max(1, width // 2),
        )


def render_contact_sheet(
    samples: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    panel_width: int = 400,
    panel_height: int = 610,
) -> tuple[int, int]:
    """Render one GT/prediction overlay panel per public training domain."""

    if not samples:
        raise ValueError("At least one sample is required")
    header_height = 94
    margin = 18
    image_top = 58
    image_bottom = 500
    canvas = Image.new(
        "RGB", (panel_width * len(samples), panel_height + header_height), color="white"
    )
    draw = ImageDraw.Draw(canvas)
    title_font = _font(22, bold=True)
    legend_font = _font(15)
    panel_title_font = _font(18, bold=True)
    detail_font = _font(13)

    draw.text(
        (margin, 14),
        "UPAR Track 3 — leak-safe leave-one-domain-out baseline",
        fill=TEXT_COLOR,
        font=title_font,
    )
    legend_y = 58
    draw.line((margin, legend_y, margin + 28, legend_y), fill=GT_COLOR, width=5)
    draw.text((margin + 36, legend_y - 10), "Ground truth", fill=TEXT_COLOR, font=legend_font)
    draw.line((160, legend_y, 188, legend_y), fill=PREDICTION_COLOR, width=5)
    draw.text((196, legend_y - 10), "LODO mean pose", fill=TEXT_COLOR, font=legend_font)
    draw.line((350, legend_y, 378, legend_y), fill=BBOX_COLOR, width=4)
    draw.text((386, legend_y - 10), "Provided box", fill=TEXT_COLOR, font=legend_font)
    draw.text(
        (540, legend_y - 10),
        "Ear slots omitted (unlabelled / ignored by metric)",
        fill=MUTED_COLOR,
        font=legend_font,
    )

    for column, sample in enumerate(samples):
        panel_left = column * panel_width
        draw.rectangle(
            (panel_left, header_height, panel_left + panel_width - 1, header_height + panel_height - 1),
            outline="#DADCE0",
            width=1,
        )
        draw.text(
            (panel_left + margin, header_height + 14),
            str(sample["domain"]).upper(),
            fill=TEXT_COLOR,
            font=panel_title_font,
        )
        image_path = Path(sample["image_path"])
        with Image.open(image_path) as source:
            source_rgb = ImageOps.exif_transpose(source).convert("RGB")
            original_width, original_height = source_rgb.size
            fitted = ImageOps.contain(
                source_rgb,
                (panel_width - 2 * margin, image_bottom - image_top),
                method=Image.Resampling.NEAREST,
            )
        paste_left = panel_left + (panel_width - fitted.width) // 2
        paste_top = header_height + image_top
        canvas.paste(fitted, (paste_left, paste_top))
        scale_x = fitted.width / original_width
        scale_y = fitted.height / original_height
        overlay = ImageDraw.Draw(canvas)
        bbox = [float(value) for value in sample["bbox"]]
        x0 = paste_left + bbox[0] * scale_x
        y0 = paste_top + bbox[1] * scale_y
        x1 = paste_left + (bbox[0] + bbox[2]) * scale_x
        y1 = paste_top + (bbox[1] + bbox[3]) * scale_y
        overlay.rectangle((x0, y0, x1, y1), outline=BBOX_COLOR, width=3)
        prediction = _scaled_keypoints(
            sample["prediction_keypoints"],
            left=paste_left,
            top=paste_top,
            scale_x=scale_x,
            scale_y=scale_y,
        )
        ground_truth = _scaled_keypoints(
            sample["ground_truth_keypoints"],
            left=paste_left,
            top=paste_top,
            scale_x=scale_x,
            scale_y=scale_y,
        )
        _draw_pose(overlay, prediction, color=PREDICTION_COLOR, width=3, radius=4)
        _draw_pose(overlay, ground_truth, color=GT_COLOR, width=3, radius=4)

        details_y = header_height + image_bottom + 14
        filename = str(sample["file_name"])
        if len(filename) > 48:
            filename = "…" + filename[-47:]
        lines = (
            filename,
            f"image={sample['image_id']}  annotation={sample['annotation_id']}",
            f"confidence={float(sample['score']):.4f}  assigned image OKS={float(sample['assigned_image_oks']):.4f}",
        )
        for row, line in enumerate(lines):
            overlay.text(
                (panel_left + margin, details_y + row * 20),
                line,
                fill=TEXT_COLOR if row else MUTED_COLOR,
                font=detail_font,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=False, compress_level=9)
    return canvas.size


def verify_png(path: Path, *, expected_size: tuple[int, int] | None = None) -> tuple[int, int]:
    """Decode-check a PNG, then reopen it to report dimensions."""

    with Image.open(path) as image:
        if image.format != "PNG":
            raise ValueError(f"Expected PNG, got {image.format!r}")
        image.verify()
    with Image.open(path) as image:
        size = image.size
        image.load()
    if expected_size is not None and size != expected_size:
        raise ValueError(f"Expected image size {expected_size}, got {size}")
    return size
