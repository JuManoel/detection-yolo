from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class YoloBox:
    """Normalized YOLO box: class_id xc yc w h in [0, 1]."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def clamp(self) -> YoloBox:
        return YoloBox(
            class_id=self.class_id,
            x_center=_clamp01(self.x_center),
            y_center=_clamp01(self.y_center),
            width=_clamp01(self.width),
            height=_clamp01(self.height),
        )

    def to_line(self) -> str:
        box = self.clamp()
        return (
            f"{box.class_id} {box.x_center:.6f} {box.y_center:.6f} "
            f"{box.width:.6f} {box.height:.6f}"
        )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def xywh_pixels_to_yolo(
    x: float,
    y: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int,
    class_id: int = 0,
) -> YoloBox:
    """Convert top-left xywh in pixels to normalized YOLO."""
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"Invalid image size: {img_w}x{img_h}")
    return YoloBox(
        class_id=class_id,
        x_center=(x + w / 2.0) / img_w,
        y_center=(y + h / 2.0) / img_h,
        width=w / img_w,
        height=h / img_h,
    ).clamp()


def voc_xyxy_to_yolo(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    img_w: int,
    img_h: int,
    class_id: int = 0,
) -> YoloBox:
    """Convert VOC xmin/ymin/xmax/ymax to normalized YOLO."""
    return xywh_pixels_to_yolo(
        x=xmin,
        y=ymin,
        w=xmax - xmin,
        h=ymax - ymin,
        img_w=img_w,
        img_h=img_h,
        class_id=class_id,
    )
