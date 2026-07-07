"""
cold_detection.py

模組版低溫區域偵測工具。

用途：
- 提供固定參數的低溫偵測，不再需要 cold_train/ 或 learn_cold_model()
- 給 Streamlit 主程式直接 import 使用
- 支援整張圖或 bottom_half ROI；目前主程式固定使用 bottom_half

注意：
OpenCV 影像格式請使用 BGR。
"""

from __future__ import annotations

from typing import Dict, Literal, Tuple

import cv2
import numpy as np

ColdModel = Dict[str, float]
RoiMode = Literal[None, "bottom_half"]


# =========================
# 固定低溫判斷參數
# =========================
# OpenCV HSV：H 範圍 0~179，S/V 範圍 0~255
# 這組參數是展示版預設值：偏藍紫/冷色 + 亮度較低。
# 若之後低溫框太多/太少，只要微調這三個數字即可。
DEFAULT_COLD_MODEL: ColdModel = {
    "h_low": 90.0,
    "h_high": 135.0,
    "v_thresh": 130.0,
}


def get_default_cold_model() -> ColdModel:
    """回傳固定低溫模型參數，避免主程式直接修改全域常數。"""
    return DEFAULT_COLD_MODEL.copy()


def detect_cold_regions(
    img_bgr: np.ndarray,
    model: ColdModel | None = None,
    min_area: int = 500,
    roi_mode: RoiMode = "bottom_half",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    在熱影像中偵測低溫區域，並用矩形框圈出。

    Parameters
    ----------
    img_bgr:
        OpenCV BGR 格式影像。
    model:
        固定低溫模型參數，需包含 h_low、h_high、v_thresh。
        若未提供，使用 DEFAULT_COLD_MODEL。
    min_area:
        最小連通區面積，小於此值視為雜訊不畫框。
    roi_mode:
        None：整張圖偵測。
        "bottom_half"：只偵測下半部。

    Returns
    -------
    outlined:
        已畫出低溫矩形框的 BGR 影像。
    mask:
        單通道二值圖，255 代表低溫區域。
    """
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("img_bgr 為空，無法進行低溫偵測。")

    if model is None:
        model = get_default_cold_model()

    h_low = float(model["h_low"])
    h_high = float(model["h_high"])
    v_thresh = float(model["v_thresh"])

    h_img, w_img = img_bgr.shape[:2]

    # 只保留 None 或 bottom_half，不再使用 bottom_40 / bottom_100 等模式。
    if roi_mode == "bottom_half":
        y0 = h_img // 2
    elif roi_mode is None:
        y0 = 0
    else:
        raise ValueError("roi_mode 只支援 None 或 'bottom_half'。")

    roi = img_bgr[y0:, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H, _S, V = cv2.split(hsv)

    cond_h = (H >= h_low) & (H <= h_high)
    cond_v = V <= v_thresh
    cold = cond_h & cond_v

    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    mask[y0:, :] = np.uint8(cold) * 255

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    outlined = img_bgr.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(outlined, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(
            outlined,
            "Low Temp",
            (x, max(y - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2,
        )

    return outlined, mask


def calculate_mask_ratio(mask: np.ndarray) -> float:
    """計算二值 mask 佔整張圖的百分比。"""
    if mask is None or mask.size == 0:
        return 0.0
    return round((np.count_nonzero(mask) / mask.size) * 100, 2)
