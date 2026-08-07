"""
cold_detection.py

模組版低溫區域偵測工具。

用途：
- 提供固定 HSV 參數的低溫偵測模組。不再需要 cold_train/ 或 learn_cold_model()
- 給 Streamlit 主程式直接 import 使用
- 支援整張圖或 bottom_half ROI；目前主程式固定使用 bottom_half
- 輸出使用實際不規則 contour，不再用矩形框。

注意：
OpenCV 影像格式請使用 BGR。

固定低溫參數：
- H: 72.30 ~ 134.59
- V <= 163.00
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np


DEFAULT_COLD_MODEL: Dict[str, float] = {
    "h_low": 72.30,
    "h_high": 134.59,
    "v_thresh": 163.00,
}

DEFAULT_COLD_MIN_AREA = 500
KERNEL = np.ones((5, 5), np.uint8)

# BGR
FILL_COLOR = (255, 0, 0)
CONTOUR_COLOR = (255, 255, 0)


def get_default_cold_model() -> Dict[str, float]:
    return DEFAULT_COLD_MODEL.copy()


def detect_cold_regions(
    img_bgr: np.ndarray,
    model: Optional[Dict[str, float]] = None,
    min_area: int = DEFAULT_COLD_MIN_AREA,
    roi_mode: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    outlined:
        低溫區以半透明藍色填色 + 不規則 contour 圈選。
    valid_cold_mask:
        經 HSV、ROI、morphology、min_area 過濾後的有效低溫 mask。
    """
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("img_bgr 為空，無法進行低溫偵測。")

    if model is None:
        model = get_default_cold_model()

    h_low = float(model["h_low"])
    h_high = float(model["h_high"])
    v_thresh = float(model["v_thresh"])

    if roi_mode not in (None, "top_half"):
        raise ValueError("roi_mode 只接受 None 或 'top_half'。")

    h_img, w_img = img_bgr.shape[:2]

    # ROI
    if roi_mode == "top_half":
        y_start = 0
        y_end = h_img // 2
    else:
        y_start = 0
        y_end = h_img
    
    roi = img_bgr[y_start:y_end, :]

    # BGR -> HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_channel, _s_channel, v_channel = cv2.split(hsv)

    # 固定 HSV 低溫條件
    cond_h = (h_channel >= h_low) & (h_channel <= h_high)
    cond_v = v_channel <= v_thresh
    cold_candidate = cond_h & cond_v

    candidate_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    candidate_mask[y_start:y_end, :] = np.uint8(cold_candidate) * 255

    # 去雜訊 + 補洞
    candidate_mask = cv2.morphologyEx(
        candidate_mask, cv2.MORPH_OPEN, KERNEL, iterations=1
    )
    candidate_mask = cv2.morphologyEx(
        candidate_mask, cv2.MORPH_CLOSE, KERNEL, iterations=1
    )

    # min_area 過濾，建立真正拿去計算比例/融合的 mask
    contours, _ = cv2.findContours(
        candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    valid_cold_mask = np.zeros_like(candidate_mask)

    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(valid_cold_mask, [cnt], -1, 255, thickness=-1)

    # 視覺化：不規則區域填色 + contour
    outlined = img_bgr.copy()
    overlay = outlined.copy()
    overlay[valid_cold_mask > 0] = FILL_COLOR
    cv2.addWeighted(overlay, 0.5, outlined, 0.5, 0, outlined)

    final_contours, _ = cv2.findContours(
        valid_cold_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        outlined, final_contours, -1, CONTOUR_COLOR, 2
    )

    if final_contours:
        largest = max(final_contours, key=cv2.contourArea)
        x, y, _w, _h = cv2.boundingRect(largest)
        cv2.putText(
            outlined,
            "Cold Temp",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            CONTOUR_COLOR,
            2,
            cv2.LINE_AA,
        )

    return outlined, valid_cold_mask


def calculate_mask_ratio(mask: np.ndarray) -> float:
    if mask is None or mask.size == 0:
        return 0.0

    abnormal_area = int(np.count_nonzero(mask))
    total_area = int(mask.size)

    return round((abnormal_area / total_area) * 100, 2) if total_area else 0.0
