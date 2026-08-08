from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..core.models import DesignSpec, PadShape, Polarity


ROLE_COLORS = {
    "battery": "#2563eb",
    "charge": "#16a34a",
    "discharge": "#ea580c",
    "temperature": "#9333ea",
    "identification": "#0891b2",
    "auxiliary": "#64748b",
}


@dataclass
class CalibrationInfo:
    """Photo calibration data for accurate preview rendering."""
    pixels_per_mm: float = 0.0
    transparent_png_path: Path | None = None


def write_mechanical_previews(
    spec: DesignSpec,
    output_dir: Path,
    calibrations: dict[str, CalibrationInfo] | None = None,
) -> list[Path]:
    """Generate mechanical preview SVG + PNG for each side.

    If *calibrations* is provided, the PNG uses the photo's actual
    pixels_per_mm so the rendering matches the real PCB at 1:1 scale.
    Pad polygons from *source_region* are drawn with accurate shape/size.
    Silk screen from *transparent_png_path* is overlaid at ~30 % opacity.
    """
    calibrations = calibrations or {}
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for side in ("front", "back"):
        svg_path = output_dir / f"mechanical_{side}.svg"
        png_path = output_dir / f"mechanical_{side}.png"
        svg_path.write_text(_svg(spec, side), encoding="utf-8")
        _png(spec, side, png_path, calibrations.get(side))
        paths.extend([svg_path, png_path])
    return paths


def _bounds(spec: DesignSpec) -> tuple[float, float, float, float]:
    xs = [p.x_mm for p in spec.outline.points]
    ys = [p.y_mm for p in spec.outline.points]
    return min(xs), min(ys), max(xs), max(ys)


def _svg(spec: DesignSpec, side: str) -> str:
    min_x, min_y, max_x, max_y = _bounds(spec)
    margin = 3
    width, height = max_x - min_x + 2 * margin, max_y - min_y + 2 * margin
    def display_x(x_mm: float) -> float:
        return (max_x - x_mm if side == "back" else x_mm - min_x) + margin

    points = " ".join(f"{display_x(p.x_mm):.3f},{p.y_mm-min_y+margin:.3f}" for p in spec.outline.points)
    items = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width*10:.0f}" height="{height*10:.0f}" viewBox="0 0 {width:.3f} {height:.3f}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<polygon points="{points}" fill="#14532d" fill-opacity="0.18" stroke="#0f172a" stroke-width="0.25"/>',
        f'<text x="1" y="2" font-size="1.4" fill="#334155">{html.escape(spec.name)} · {side}</text>',
    ]
    for terminal in spec.terminals:
        if terminal.side.value != side:
            continue
        x = display_x(terminal.position.x_mm)
        y = terminal.position.y_mm - min_y + margin
        color = ROLE_COLORS[sorted(role.value for role in terminal.roles)[0]]

        # ★ 优先使用 source_region.polygon 绘制真实焊盘形状
        polygon = (terminal.source_region.polygon
                   if terminal.source_region
                      and terminal.source_region.polygon
                   else None)
        if polygon and len(polygon) >= 3:
            poly_pts = " ".join(
                f"{display_x(p.x_mm):.3f},{p.y_mm - min_y + margin:.3f}"
                for p in polygon
            )
            items.append(f'<polygon points="{poly_pts}" fill="{color}" stroke="#111827" stroke-width="0.15"/>')
        elif terminal.shape == PadShape.CIRCLE:
            items.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{terminal.width_mm/2:.3f}" fill="{color}" stroke="#111827" stroke-width="0.15"/>')
        else:
            items.append(f'<rect x="{x-terminal.width_mm/2:.3f}" y="{y-terminal.height_mm/2:.3f}" width="{terminal.width_mm:.3f}" height="{terminal.height_mm:.3f}" rx="{terminal.height_mm/2 if terminal.shape == PadShape.OVAL else 0:.3f}" fill="{color}" stroke="#111827" stroke-width="0.15"/>')

        sign = "+" if terminal.polarity == Polarity.POSITIVE else ("−" if terminal.polarity == Polarity.NEGATIVE else "")
        roles = "/".join(sorted(role.value[0].upper() for role in terminal.roles))
        items.append(f'<text x="{x:.3f}" y="{y-terminal.height_mm/2-0.7:.3f}" text-anchor="middle" font-size="1.2" fill="#111827">{html.escape(terminal.id)} {roles}{sign}</text>')
    items.append("</svg>")
    return "\n".join(items)


def _png(spec: DesignSpec, side: str, path: Path, cal: CalibrationInfo | None = None) -> None:
    """Render mechanical preview PNG.

    Uses the photo's actual pixels_per_mm (from *cal*) so the rendering
    matches the real PCB at 1:1 pixel scale.  Pad polygons from
    *source_region* are drawn with accurate shape/size.  Silk screen
    from *transparent_png_path* is overlaid at ~30 % opacity.
    """
    min_x, min_y, max_x, max_y = _bounds(spec)
    pcb_w_mm = max_x - min_x
    pcb_h_mm = max_y - min_y

    # ── Scale: use photo resolution when available ──────────────────────
    if cal and cal.pixels_per_mm > 0:
        ppm = cal.pixels_per_mm
    else:
        # Fallback: aim for ~1200 px wide canvas
        target_w = 1200
        margin_px = 80
        ppm = max(15.0, min(60.0,
              (target_w - 2 * margin_px) / max(pcb_w_mm, 1.0)))

    # ── Canvas sizing (PCB + centered margin) ──────────────────────────
    margin_px = int(0.8 * ppm)           # ~0.8 mm margin
    pcb_px_w = int(round(pcb_w_mm * ppm))
    pcb_px_h = int(round(pcb_h_mm * ppm))
    width  = pcb_px_w + 2 * margin_px
    height = pcb_px_h + 2 * margin_px
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    ox = (width  - pcb_px_w) // 2        # centering offset
    oy = (height - pcb_px_h) // 2

    def to_px_x(x_mm: float) -> float:
        """mm (frame coords) → canvas pixel X.  Back side is mirrored."""
        if side == "back":
            return (max_x - x_mm) * ppm + ox
        return (x_mm - min_x) * ppm + ox

    def to_px_y(y_mm: float) -> float:
        return (y_mm - min_y) * ppm + oy

    # ── 1. PCB fill + outline ──────────────────────────────────────────
    outline_pts = np.array(
        [[to_px_x(p.x_mm), to_px_y(p.y_mm)] for p in spec.outline.points],
        np.int32)
    cv2.fillPoly(canvas, [outline_pts], (220, 245, 220))
    cv2.polylines(canvas, [outline_pts], True, (25, 25, 25), 2)

    # ── 2. Silk screen overlay from transparent PCB photo ──────────────
    if cal and cal.transparent_png_path and cal.transparent_png_path.exists():
        try:
            silk = cv2.imdecode(
                np.frombuffer(cal.transparent_png_path.read_bytes(), np.uint8),
                cv2.IMREAD_UNCHANGED)
            if (silk is not None
                    and len(silk.shape) == 3 and silk.shape[2] == 4):
                sh_full, sw_full = silk.shape[:2]
                # Extract PCB region from full-frame transparent PNG
                # Frame coords: (min_x, min_y) to (max_x, max_y) in mm
                x1 = max(0, int(round(min_x * ppm)))
                y1 = max(0, int(round(min_y * ppm)))
                x2 = min(sw_full, int(round(max_x * ppm)))
                y2 = min(sh_full, int(round(max_y * ppm)))
                if x2 > x1 and y2 > y1:
                    pcb_region = silk[y1:y2, x1:x2]
                    # Resize to match canvas PCB size if needed
                    if pcb_region.shape[1] != pcb_px_w or pcb_region.shape[0] != pcb_px_h:
                        pcb_region = cv2.resize(
                            pcb_region, (pcb_px_w, pcb_px_h),
                            interpolation=cv2.INTER_AREA)
                    # Mirror for back side
                    if side == "back":
                        pcb_region = cv2.flip(pcb_region, 1)
                    # Alpha blend at 30% opacity onto canvas at (ox, oy)
                    alpha = pcb_region[:, :, 3].astype(np.float32) / 255.0
                    a = np.clip(alpha * 0.3, 0, 1)
                    a3 = np.stack([a] * 3, axis=-1)
                    bgr = pcb_region[:, :, :3].astype(np.float32)
                    # Paste region onto canvas
                    canvas_roi = canvas[oy:oy+pcb_px_h, ox:ox+pcb_px_w]
                    canvas_f = canvas_roi.astype(np.float32)
                    blended = np.where(
                        a3 > 0.01,
                        (canvas_f * (1 - a3) + bgr * a3),
                        canvas_f).astype(np.uint8)
                    canvas[oy:oy+pcb_px_h, ox:ox+pcb_px_w] = blended
        except Exception:
            pass  # silk screen overlay is optional

    # ── 3. Terminal pads (real polygon when available) ─────────────────
    font_scale = max(0.35, min(0.65, ppm / 45.0))
    for terminal in spec.terminals:
        if terminal.side.value != side:
            continue
        color = ((220, 110, 30) if TerminalRoleName.discharge(terminal)
                 else (50, 170, 60))

        # Prefer polygon from source_region for accurate shape/size
        polygon = (terminal.source_region.polygon
                   if terminal.source_region
                      and terminal.source_region.polygon
                   else None)
        if polygon and len(polygon) >= 3:
            pts = np.array(
                [[to_px_x(p.x_mm), to_px_y(p.y_mm)] for p in polygon],
                np.int32)
            cv2.fillPoly(canvas, [pts], color)
            cv2.polylines(canvas, [pts], True, (20, 20, 20), 1)
            # Label position: right of polygon bounding box
            xs, ys = pts[:, 0], pts[:, 1]
            lx = int(xs.max()) + 4
            ly = int(ys.mean()) + 4
        else:
            # Fallback: circle/rect from width_mm / height_mm
            cx = int(to_px_x(terminal.position.x_mm))
            cy = int(to_px_y(terminal.position.y_mm))
            hw = max(3, int(terminal.width_mm  * ppm / 2))
            hh = max(3, int(terminal.height_mm * ppm / 2))
            if terminal.shape == PadShape.CIRCLE:
                cv2.circle(canvas, (cx, cy), hw, color, -1)
            else:
                cv2.rectangle(canvas,
                              (cx - hw, cy - hh),
                              (cx + hw, cy + hh), color, -1)
            lx = cx + hw + 4
            ly = cy + 4

        cv2.putText(canvas, terminal.id, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (10, 10, 10), 1, cv2.LINE_AA)

    # ── 4. Title bar ───────────────────────────────────────────────────
    title = f"{spec.name}  -  {side.upper()}"
    cv2.putText(canvas, title, (ox, max(oy - int(ppm * 0.3), 25)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 1,
                cv2.LINE_AA)

    cv2.imwrite(str(path), canvas)



class TerminalRoleName:
    @staticmethod
    def discharge(terminal) -> bool:
        return any(role.value == "discharge" for role in terminal.roles)
