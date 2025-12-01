# src/model_eval/vis/overlay.py

from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import pandas as pd

from model_eval.data.loaders import load_gt_pred_all
from model_eval.config import (
    DATA_RAW_VIDEOS,
    REPORTS_ANALYSIS_FIGURES,
    GT_SOURCE,
    PRED_SOURCE,
    OVERLAY_GT_COLOR,
    OVERLAY_PRED_COLOR
)


def _to_bgr(color) -> tuple[int, int, int]:
    """Convert '#RRGGBB' or (R,G,B) to OpenCV BGR."""
    if isinstance(color, str):
        c = color.lstrip("#")
        r = int(c[0:2], 16)
        g = int(c[2:4], 16)
        b = int(c[4:6], 16)
        return (b, g, r)
    if isinstance(color, (tuple, list)) and len(color) == 3:
        r, g, b = color
        return (b, g, r)
    raise ValueError(f"Unsupported color format: {color!r}")


def _find_frame_image(series_id: str, frame_number: int) -> Optional[Path]:
    """
    Find the image file for a global frame number.

    Assumes structure like:
      DATA_RAW_VIDEO/<series>/<start>_0/images/frame_0000.jpg

    where <start> is the starting global frame of that segment.
    """
    series_root = DATA_RAW_VIDEOS / series_id
    if not series_root.exists():
        return None

    # collect segment start frames from folder names (e.g. '1470_0')
    starts: list[int] = []
    for d in series_root.iterdir():
        if d.is_dir():
            name = d.name.split("_", 1)[0]
            try:
                starts.append(int(name))
            except ValueError:
                continue

    starts = [s for s in starts if s <= frame_number]
    if not starts:
        return None

    start = max(starts)
    idx = frame_number - start
    seg_dir = series_root / f"{start}_0"

    # common pattern: images/frame_0000.jpg
    for ext in (".jpg", ".png", ".jpeg"):
        p = seg_dir / "images" / f"frame_{idx:04d}{ext}"
        if p.exists():
            return p

    # simple fallbacks (no /images or different padding)
    for pad in (3, 5, 4):
        for ext in (".jpg", ".png", ".jpeg"):
            p = seg_dir / f"frame_{idx:0{pad}d}{ext}"
            if p.exists():
                return p

    return None


def _put_text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int],
    outline: tuple[int, int, int] = (0, 0, 0),
    scale: float = 0.65,
    thickness: int = 2
) -> None:
    """Draw text with an outline for readability."""
    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        outline,
        thickness + 2,
        cv2.LINE_AA
    )
    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def _clip_coords(x1, y1, x2, y2, w, h) -> tuple[int, int, int, int]:
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    return x1, y1, x2, y2


def _draw_box(
    img: np.ndarray,
    row: pd.Series,
    source: str,
    gt_color_bgr: tuple[int, int, int],
    pred_color_bgr: tuple[int, int, int]
) -> None:
    """
    Draw a single bbox.

    - If source == GT_SOURCE: draw GT_COLOR box + "GT"
    - If source == PRED_SOURCE: draw PRED_COLOR box + "id=.. conf=.." text
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = _clip_coords(row.x_min, row.y_min, row.x_max, row.y_max, w, h)

    if source == GT_SOURCE:
        color = gt_color_bgr
    else:
        color = pred_color_bgr

    # draw rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # build text
    if source == GT_SOURCE:
        text = "GT"
    elif source == PRED_SOURCE:
        parts: list[str] = []
        tid = row.get("track_id", None)
        conf = row.get("confidence", None)
        if pd.notna(tid):
            parts.append(f"track_id={int(tid)}")
        if pd.notna(conf):
            parts.append(f"conf={float(conf):.2f}")
        text = "  ".join(parts) if parts else ""
    else:
        text = ""

    if not text:
        return

    # compute text size
    scale = 0.55
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )

    # initial position: above top-left
    tx = x1 + 2
    ty = y1 - 4

    # if it doesn't fit above, move inside the box (just below top edge)
    if ty - th < 0:
        ty = y1 + th + 4

    # clamp horizontally inside image
    if tx + tw > w - 2:
        tx = max(2, w - tw - 2)

    # clamp vertically inside image
    if ty + baseline > h - 2:
        ty = max(th + 2, h - baseline - 2)

    _put_text(
        img,
        text,
        (tx, ty),
        color=color,
        outline=(255, 255, 255),
        scale=scale,
        thickness=thickness,
    )

def overlay_frame(
    video_series: str,
    frame_number: int,
    save_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Draw GT and PRED boxes for one series + global frame number.

    - GT in OVERLAY_GT_COLOR with label "GT"
    - PRED in OVERLAY_PRED_COLOR with "id=...  conf=..."
    - Adds a title with series and frame number at top-left.

    Returns Path to saved overlay, or None if frame image not found.
    """
    img_path = _find_frame_image(video_series, frame_number)
    if img_path is None:
        print(f"[WARN] No frame image found for {video_series} frame {frame_number}")
        return None

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARN] Could not read image: {img_path}")
        return None

    df = load_gt_pred_all()
    if df.empty:
        raise ValueError("gt_pred_all.csv is empty or not loaded correctly.")

    df_frame = df[
        (df["video_series"] == video_series)
        & (df["frame"] == frame_number)
    ]

    gt_color_bgr = _to_bgr(OVERLAY_GT_COLOR)
    pred_color_bgr = _to_bgr(OVERLAY_PRED_COLOR)

    # draw all boxes with a single helper, behavior depends on source
    for _, row in df_frame.iterrows():
        src = row["source"]
        _draw_box(img, row, source=src, gt_color_bgr=gt_color_bgr, pred_color_bgr=pred_color_bgr)

    # title
    title = f"{video_series}  frame {frame_number}"
    _put_text(
        img,
        title,
        (12, 28),
        color=(255, 255, 255),
        outline=(0, 0, 0),
        scale=0.8,
        thickness=2
    )

    # Save
    if save_dir is None:
        save_dir = REPORTS_ANALYSIS_FIGURES / "overlays" / video_series
    save_dir.mkdir(parents=True, exist_ok=True)

    out_path = save_dir / f"{video_series}_frame_{frame_number:05d}.png"
    cv2.imwrite(str(out_path), img)
    print(f"[OK] Saved overlay: {out_path}")
    return out_path


def overlay_frames(
    video_series: str,
    frames: Iterable[int],
    save_dir: Optional[Path] = None
) -> list[Path]:
    """
    Overlay multiple global frame numbers for one series.
    """
    paths: list[Path] = []
    for f in frames:
        p = overlay_frame(video_series, f, save_dir=save_dir)
        if p is not None:
            paths.append(p)
    return paths