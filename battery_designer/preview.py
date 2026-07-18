from __future__ import annotations

import html
from pathlib import Path

import cv2
import numpy as np

from .models import DesignSpec, PadShape, Polarity


ROLE_COLORS = {
    "battery": "#2563eb",
    "charge": "#16a34a",
    "discharge": "#ea580c",
    "temperature": "#9333ea",
    "identification": "#0891b2",
    "auxiliary": "#64748b",
}


def write_mechanical_previews(spec: DesignSpec, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for side in ("front", "back"):
        svg_path = output_dir / f"mechanical_{side}.svg"
        png_path = output_dir / f"mechanical_{side}.png"
        svg_path.write_text(_svg(spec, side), encoding="utf-8")
        _png(spec, side, png_path)
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
        if terminal.shape == PadShape.CIRCLE:
            items.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{terminal.width_mm/2:.3f}" fill="{color}" stroke="#111827" stroke-width="0.15"/>')
        else:
            items.append(f'<rect x="{x-terminal.width_mm/2:.3f}" y="{y-terminal.height_mm/2:.3f}" width="{terminal.width_mm:.3f}" height="{terminal.height_mm:.3f}" rx="{terminal.height_mm/2 if terminal.shape == PadShape.OVAL else 0:.3f}" fill="{color}" stroke="#111827" stroke-width="0.15"/>')
        sign = "+" if terminal.polarity == Polarity.POSITIVE else ("−" if terminal.polarity == Polarity.NEGATIVE else "")
        roles = "/".join(sorted(role.value[0].upper() for role in terminal.roles))
        items.append(f'<text x="{x:.3f}" y="{y-terminal.height_mm/2-0.7:.3f}" text-anchor="middle" font-size="1.2" fill="#111827">{html.escape(terminal.id)} {roles}{sign}</text>')
    items.append("</svg>")
    return "\n".join(items)


def _png(spec: DesignSpec, side: str, path: Path) -> None:
    min_x, min_y, max_x, max_y = _bounds(spec)
    scale, margin = 20, 60
    width = max(320, int((max_x - min_x) * scale + 2 * margin))
    height = max(240, int((max_y - min_y) * scale + 2 * margin))
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    def display_x(x_mm: float) -> float:
        return max_x - x_mm if side == "back" else x_mm - min_x

    polygon = np.array([[display_x(p.x_mm) * scale + margin, (p.y_mm - min_y) * scale + margin] for p in spec.outline.points], np.int32)
    cv2.fillPoly(canvas, [polygon], (220, 245, 220))
    cv2.polylines(canvas, [polygon], True, (25, 25, 25), 2)
    for terminal in spec.terminals:
        if terminal.side.value != side:
            continue
        x = int(display_x(terminal.position.x_mm) * scale + margin)
        y = int((terminal.position.y_mm - min_y) * scale + margin)
        color = (220, 110, 30) if TerminalRoleName.discharge(terminal) else (50, 170, 60)
        axes = (max(2, int(terminal.width_mm * scale / 2)), max(2, int(terminal.height_mm * scale / 2)))
        if terminal.shape == PadShape.CIRCLE:
            cv2.circle(canvas, (x, y), axes[0], color, -1)
        else:
            cv2.rectangle(canvas, (x - axes[0], y - axes[1]), (x + axes[0], y + axes[1]), color, -1)
        cv2.putText(canvas, terminal.id, (x + axes[0] + 3, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 10, 10), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


class TerminalRoleName:
    @staticmethod
    def discharge(terminal) -> bool:
        return any(role.value == "discharge" for role in terminal.roles)
