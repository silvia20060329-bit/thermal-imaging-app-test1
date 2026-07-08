import numpy as np
import cv2


def to_uint8(img):
    """確保影像是 uint8，方便做直方圖與顯示。"""
    if img.dtype == np.uint8:
        return img
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def compute_mutual_information(a, b, bins=64, eps=1e-12, mask=None):
    """
    計算兩張灰階圖 a, b 的互資訊 (normalized)

    參數：
        a, b : 灰階影像 (ndarray)
        bins : 直方圖分箱數
        mask : 若提供，僅在 mask>0 的像素上計算 MI（用於 IR 熱區加權）
    """
    a = to_uint8(a)
    b = to_uint8(b)

    if mask is not None:
        mask = (mask > 0)
        if not np.any(mask):
            # 沒有有效像素，回傳 0（避免 nan）
            return 0.0
        a = a[mask]
        b = b[mask]

    H, _, _ = np.histogram2d(
        a.ravel(), b.ravel(),
        bins=bins,
        range=[[0, 255], [0, 255]]
    )

    pxy = H / max(H.sum(), eps)
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)

    Hx = -np.sum(px * np.log(px + eps))
    Hy = -np.sum(py * np.log(py + eps))
    Hxy = -np.sum(pxy * np.log(pxy + eps))

    mi = (Hx + Hy) / max(Hxy, eps)
    return float(mi)
