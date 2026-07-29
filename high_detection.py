"""
high_detection.py

模組版高溫區域偵測工具。

用途：
- 將 test2app.py 裡原本的高溫圈選邏輯獨立成模組
- 圈選邏輯維持不變：熱影像轉灰階 → threshold 找高亮區 → contour 篩選 → 用紅色圓圈圈出高溫區
- 給 Streamlit 主程式直接 import 使用

注意：
OpenCV 影像格式請使用 BGR。
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


DEFAULT_HIGH_THRESHOLD = 200
DEFAULT_HIGH_MIN_AREA = 50


def detect_high_regions(
    img_bgr: np.ndarray,
    threshold: int = DEFAULT_HIGH_THRESHOLD,
    min_area: int = DEFAULT_HIGH_MIN_AREA,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    在熱影像中偵測高溫區域，並用紅色圓圈圈出。

    Parameters
    ----------
    img_bgr:
        OpenCV BGR 格式影像。
    threshold:
        灰階亮度門檻。像素亮度 > threshold 會被視為高溫候選區。
    min_area:
        最小連通區面積，小於此值視為雜訊不畫圈。

    Returns
    -------
    outlined:
        已畫出高溫圓圈的 BGR 影像。
    mask:
        單通道二值圖，255 代表高溫區域。
    ratio:
        高溫異常面積佔整張圖比例，單位為百分比。
    """
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("img_bgr 為空，無法進行高溫偵測。")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, high_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        high_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    outlined = img_bgr.copy()
    high_area = 0.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        high_area += area

        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)

        cv2.circle(outlined, center, radius, (0, 0, 255), 3)
        cv2.putText(
            outlined,
            "High Temp",
            (center[0] - 40, max(center[1] - radius - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    total_area = gray.shape[0] * gray.shape[1]
    ratio = round((high_area / total_area) * 100, 2) if total_area > 0 else 0.0

    return outlined, high_mask, ratio
