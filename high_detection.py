"""
high_detection.py

模組版高溫區域偵測工具。

用途：
- 將 test1app.py 裡原本的高溫圈選邏輯獨立成模組
- 圈選邏輯維持不變：熱影像轉灰階 → threshold 找高亮區 → contour 篩選 → 用紅色圓圈圈出高溫區
- 給 Streamlit 主程式直接 import 使用

注意：
OpenCV 影像格式請使用 BGR。

此版本使用「後來整合版」的高溫偵測邏輯：
1. Thermal BGR -> HSV
2. 使用 V channel
3. 取 V 亮度第 95 百分位（最高 5%）作為高溫候選
4. MORPH_OPEN 去除雜訊
5. MORPH_DILATE 擴張連通區
6. 依 min_area 過濾小區域
7. 回傳「過濾後」的 valid_high_mask，供 Wall/Floor 資訊融合

注意：
- 本模組不限制高溫只能發生在 Floor 或 Wall。
- Wall / Floor 的限制應由 test2app.py 在 Information Fusion 階段再做交集。
- OpenCV 影像格式為 BGR。
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


HOT_PERCENT = 95
DEFAULT_HIGH_MIN_AREA = 50
BOX_COLOR = (0, 255, 255)  # BGR 黃色
KERNEL = np.ones((5, 5), np.uint8)


def detect_high_regions(
    img_bgr: np.ndarray,
    min_area: int = DEFAULT_HIGH_MIN_AREA,
    hot_percent: float = HOT_PERCENT,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    偵測熱影像中的高溫候選區域。

    Parameters
    ----------
    img_bgr:
        OpenCV BGR 格式熱影像。
    min_area:
        最小連通區面積，小於此值視為雜訊。
    hot_percent:
        V channel 百分位門檻。
        預設 95，表示取亮度最高約 5% 的像素作為高溫候選。

    Returns
    -------
    outlined:
        已將有效高溫區塗紅並以黃色輪廓圈出的 BGR 圖。
    valid_high_mask:
        經 morphology 與 min_area 過濾後的二值 mask。
        255 = 有效高溫區。
    ratio:
        有效高溫區佔整張熱影像的百分比。
    """
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("img_bgr 為空，無法進行高溫偵測。")

    if not (0 < hot_percent < 100):
        raise ValueError("hot_percent 必須介於 0 與 100 之間。")

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    _h_channel, _s_channel, v_channel = cv2.split(hsv)

    # 後來整合版邏輯：以 V channel 第 95 百分位作為門檻。
    thresh_v = float(np.percentile(v_channel, hot_percent))
    cond_bright = v_channel >= thresh_v

    candidate_mask = np.zeros_like(v_channel, dtype=np.uint8)
    candidate_mask[cond_bright] = 255

    # 與後來整合版一致：OPEN + DILATE
    candidate_mask = cv2.morphologyEx(
        candidate_mask,
        cv2.MORPH_OPEN,
        KERNEL,
        iterations=1,
    )
    candidate_mask = cv2.morphologyEx(
        candidate_mask,
        cv2.MORPH_DILATE,
        KERNEL,
        iterations=1,
    )

    contours, _ = cv2.findContours(
        candidate_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    # 只把真正通過 min_area 的 contour 放進最後 mask。
    # 這樣畫面結果與後續 Information Fusion 使用的 mask 完全一致。
    valid_high_mask = np.zeros_like(v_channel, dtype=np.uint8)

    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(valid_high_mask, [cnt], -1, 255, -1)

    # 視覺化：有效區域塗紅 + 黃色不規則輪廓
    outlined = img_bgr.copy()
    overlay = outlined.copy()
    overlay[valid_high_mask == 255] = (0, 0, 255)
    cv2.addWeighted(overlay, 0.5, outlined, 0.5, 0, outlined)

    final_contours, _ = cv2.findContours(
        valid_high_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(outlined, final_contours, -1, BOX_COLOR, 2)

    # 保留後來整合版「在最大區域旁放標籤」的呈現方式，
    # 但不再寫死 (Floor)，因為區域分類會在主程式最後融合。
    if final_contours:
        largest_contour = max(final_contours, key=cv2.contourArea)
        x, y, _w, _h = cv2.boundingRect(largest_contour)
        cv2.putText(
            outlined,
            "High Temp",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            BOX_COLOR,
            2,
        )

    total_area = valid_high_mask.size
    high_area = int(np.count_nonzero(valid_high_mask))
    ratio = round((high_area / total_area) * 100, 2) if total_area else 0.0

    return outlined, valid_high_mask, ratio
