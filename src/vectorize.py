"""
vectorize.py — этап 5: из чистой бинарной маски делаем векторы (списки точек).

Идея: cv2.findContours обводит каждое белое пятно ломаной линией (контуром).
Затем cv2.approxPolyDP выкидывает лишние точки — линия становится гладкой и лёгкой.

Результат — список «фич»: каждая фича = одна линия с её точками (в пикселях),
типом (fault/boundary) и цветом-источником. Это уже почти GeoJSON, только без файла.

Координаты — пиксельные: (x вправо, y вниз). Геопривязки нет, это Трек 3.
"""

import cv2

from src import config


def vectorize(cleaned, prepared, saver):
    """
    cleaned — словарь из cleanup: {"color_masks", "combined", "canny"}.
    prepared — нужен только цветной кадр для рисования overlay.
    Возвращает список фич:
      [{"points": [(x, y), ...], "type": "fault", "color": "red", "length_px": float}, ...]
    """
    features = []

    # Векторизуем каждую очищенную цветную маску.
    for color_name, spec in cleaned["color_masks"].items():
        polylines = _mask_to_polylines(spec["mask"])
        for pts, length in polylines:
            features.append({
                "points": pts,
                "type": spec["type"],
                "color": color_name,
                "length_px": length,
            })

    # Если цвета не было (калька) — векторизуем края Canny как тип "edge".
    if not cleaned["color_masks"] and cleaned["canny"] is not None:
        polylines = _mask_to_polylines(cleaned["canny"])
        for pts, length in polylines:
            features.append({
                "points": pts,
                "type": "edge",
                "color": "none",
                "length_px": length,
            })

    # Самый важный debug-кадр: вектора поверх оригинала — проверяем глазами.
    overlay = _draw_overlay(prepared["color"], features)
    saver.save("vectors_overlay", overlay)

    return features


def _mask_to_polylines(mask):
    """
    Обвести белые пятна контурами и упростить.
    Возвращает список (points, length_px), где points — список (x, y).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = []
    for contour in contours:
        # Слишком короткие контуры — мусор, пропускаем.
        if len(contour) < config.MIN_CONTOUR_POINTS:
            continue
        # Упрощаем ломаную (Douglas-Peucker): меньше точек, та же форма.
        approx = cv2.approxPolyDP(contour, config.APPROX_EPSILON, True)
        pts = [(int(x), int(y)) for x, y in approx.reshape(-1, 2)]
        if len(pts) < 2:
            continue
        length = float(cv2.arcLength(approx, True))
        result.append((pts, length))
    return result


def _draw_overlay(color_image, features):
    """Нарисовать все вектора поверх копии цветного кадра (для визуальной проверки)."""
    overlay = color_image.copy()
    # Цвета обводки в BGR по типу объекта.
    type_colors = {
        "fault": (0, 0, 255),      # красный
        "boundary": (0, 255, 0),   # зелёный
        "edge": (255, 0, 0),       # синий
    }
    for f in features:
        color = type_colors.get(f["type"], (0, 255, 255))
        pts = f["points"]
        for i in range(len(pts) - 1):
            cv2.line(overlay, pts[i], pts[i + 1], color, 2)
    return overlay
