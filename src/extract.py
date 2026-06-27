"""
extract.py — этап 3: выделение объектов. САМЫЙ важный этап.

Две независимые ветки:
  1) Цвет (HSV): переводим цветной кадр в HSV и для каждого «целевого цвета»
     из профиля (config.PROFILES) вырезаем пиксели в заданном диапазоне -> бинарная маска.
  2) Края (Canny): по серому кадру ищем границы. Это запасной путь для калек,
     где цвета нет.

На выходе — словарь с масками. Белое (255) = «здесь объект», чёрное (0) = фон.
Каждая маска сохраняется в debug, чтобы видеть глазами, что именно поймалось.
"""

import cv2
import numpy as np

from src import config


def extract(prepared, profile_name, saver):
    """
    prepared — словарь из preprocess: {"color": BGR, "gray": серый}.
    profile_name — какой набор цветов брать из config.PROFILES.
    saver — DebugSaver для промежуточных кадров.

    Возвращает:
      {
        "color_masks": {"red": {"mask": ndarray, "type": "fault"}, ...},
        "canny": ndarray | None,
        "combined": ndarray,   # объединение всех цветных масок (для наглядности)
      }
    """
    color_image = prepared["color"]
    gray_image = prepared["gray"]
    profile = config.PROFILES.get(profile_name, {})

    # Переводим в HSV один раз (дальше все цвета режем из него).
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    color_masks = {}
    # Пустая «нулевая» маска нужного размера, в неё накапливаем объединение.
    combined = np.zeros(gray_image.shape, dtype=np.uint8)

    # --- Ветка 1: цвет ---
    for color_name, spec in profile.items():
        mask = _mask_for_color(hsv, spec["ranges"])
        color_masks[color_name] = {"mask": mask, "type": spec["type"]}
        combined = cv2.bitwise_or(combined, mask)
        saver.save(f"mask_{color_name}", mask)

    # --- Ветка 2: края (Canny) ---
    canny = None
    if config.USE_CANNY:
        canny = cv2.Canny(gray_image, config.CANNY_THRESHOLD_LOW, config.CANNY_THRESHOLD_HIGH)
        saver.save("canny", canny)

    saver.save("mask_combined", combined)

    return {"color_masks": color_masks, "canny": canny, "combined": combined}


def _mask_for_color(hsv, ranges):
    """
    Собрать одну бинарную маску для цвета, у которого может быть НЕСКОЛЬКО диапазонов
    HSV (например, красный — два диапазона по краям круга тонов).
    Маски диапазонов объединяем логическим ИЛИ.
    """
    h, w = hsv.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for lower, upper in ranges:
        lower_np = np.array(lower, dtype=np.uint8)
        upper_np = np.array(upper, dtype=np.uint8)
        part = cv2.inRange(hsv, lower_np, upper_np)
        mask = cv2.bitwise_or(mask, part)
    return mask
