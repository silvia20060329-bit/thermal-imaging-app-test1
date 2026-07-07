# cold_detection.py

import os
import glob
import numpy as np
import cv2


# =========================
# Step 1. 學習「低溫顏色特徵」
# =========================

def learn_cold_model(
    image_paths,
    sample_fraction=0.1,
    cold_quantile=0.3,
):
    """
    根據多張熱影像，自動估計：
    - 冷色 Hue 範圍 (h_low, h_high)
    - 亮度門檻 v_thresh （比這更暗 + 是冷色，就視為低溫）

    image_paths      : 影像路徑列表
    sample_fraction  : 每張圖隨機取多少比例像素來學 (0~1)
    cold_quantile    : 取亮度最暗的多少比例像素，當作「冷候選」(0~1)
    """
    all_H = []
    all_V = []

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] 無法讀取圖片：{path}")
            continue

        # BGR -> HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)

        # 攤平成 1D
        h_flat = H.flatten()
        v_flat = V.flatten()

        # 隨機抽樣一部分像素，避免太大
        n_pixels = len(h_flat)
        n_sample = max(1000, int(n_pixels * sample_fraction))
        n_sample = min(n_sample, n_pixels)

        idx = np.random.choice(n_pixels, n_sample, replace=False)
        all_H.append(h_flat[idx])
        all_V.append(v_flat[idx])

    if not all_H:
        raise RuntimeError("沒有成功讀到任何影像，請檢查路徑。")

    H = np.concatenate(all_H).astype(np.float32)
    V = np.concatenate(all_V).astype(np.float32)

    # 取亮度最暗的一群像素，當作「冷候選」
    v_thresh = np.quantile(V, cold_quantile)
    cold_H = H[V <= v_thresh]

    # 用平均 ± 2σ 當作 Hue 範圍
    h_mean = float(np.mean(cold_H))
    h_std = float(np.std(cold_H)) + 1e-6

    h_low = max(0.0, h_mean - 2 * h_std)
    h_high = min(179.0, h_mean + 2 * h_std)

    print("=== 冷色模型參數 ===")
    print(f"Hue 平均值   : {h_mean:.2f}")
    print(f"Hue 標準差   : {h_std:.2f}")
    print(f"Hue 範圍     : [{h_low:.2f}, {h_high:.2f}]  (OpenCV 的 0~179)")
    print(f"亮度門檻 V   : <= {v_thresh:.2f} (0~255) 視為偏冷")

    model = {
        "h_low": h_low,
        "h_high": h_high,
        "v_thresh": float(v_thresh),
    }
    return model


# =========================
# Step 2. 在新圖上「用矩形圈出」低溫異常
# =========================

def detect_cold_regions(
    img,
    model,
    min_area=200,
    roi_mode=None,
):
    """
    根據學到的 model (h_low, h_high, v_thresh)
    在單張圖上找低溫區，輸出：
    - outlined : 已在低溫區畫出矩形框的彩色圖
    - mask     : 單通道二值圖，255 = 低溫區

    roi_mode:
        None         : 不限制，整張圖都偵測
        "bottom_half": 只抓下半部 (地板常在下方)
        "bottom_40"  : 只抓下 100% 高度
    """
    h_low = model["h_low"]
    h_high = model["h_high"]
    v_thresh = model["v_thresh"]

    h_img, w_img = img.shape[:2]

    # 決定 ROI 區域
    if roi_mode == "bottom_half":
        y0 = h_img // 2
    elif roi_mode == "bottom_100":
        y0 = int(h_img * 0.6)
    else:
        y0 = 0

    roi = img[y0:, :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # 條件：Hue 在冷色範圍內，且亮度夠暗
    cond_h = (H >= h_low) & (H <= h_high)
    cond_v = (V <= v_thresh)
    cold = cond_h & cond_v

    # 建立整張圖大小的 mask
    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    mask_roi = np.uint8(cold) * 255
    mask[y0:, :] = mask_roi

    # 形態學處理：去雜訊 + 補洞
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # ========= 新：改用「矩形」圈出低溫 =========
    outlined = img.copy()

    # 找出所有非零像素 (低溫區)
    ys, xs = np.where(mask > 0)

    if len(xs) > 0:
        # 先用 contour 篩掉太小的區塊，再合併
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # 保留面積夠大的 contour
        big_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

        if len(big_contours) == 0:
            # 沒有符合最小面積，就直接用所有低溫點算一個大矩形
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            cv2.rectangle(outlined, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        else:
            # 對每個大一點的區域畫一個矩形
            for c in big_contours:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(outlined, (x, y), (x + w, y + h), (0, 0, 255), 2)
    else:
        print("[INFO] 此圖沒有偵測到低溫區（mask 全為 0）。")

    return outlined, mask


# =========================
# main: 示範怎麼跑
# =========================

def main():
    # 0) 統一一個 base_dir，之後全部都從這裡長出去
    base_dir = r"C:\Users\USER\Desktop\project\temperature"

    train_dir = os.path.join(base_dir, r"test_images\tainan_temple")
    test_dir = os.path.join(base_dir, "train_images\LI_2")
    results_dir = os.path.join(base_dir, "results")

    # 1) 讀取訓練影像路徑
    train_paths = sorted(
        glob.glob(os.path.join(train_dir, "*.png"))
    )
    print(f"[TRAIN] 從 {train_dir} 找到 {len(train_paths)} 張訓練影像。")

    # 2) 學習冷色模型
    model = learn_cold_model(
        train_paths,
        sample_fraction=0.1,   # 每張取 10% 像素來學
        cold_quantile=0.2,     # 取最暗 20% 當冷候選
    )

    # 3) 對 test_images 底下的圖做偵測
    test_paths = sorted(
        glob.glob(os.path.join(test_dir, "*.png"))
    )
    print(f"[TEST] 從 {test_dir} 找到 {len(test_paths)} 張要偵測的影像。")

    os.makedirs(results_dir, exist_ok=True)
    print(f"[INFO] 結果會輸出到：{results_dir}")

    for path in test_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] 無法讀取圖片：{path}")
            continue

        outlined, mask = detect_cold_regions(
            img,
            model,
            min_area=500,
            roi_mode="bottom_35"
        )

        base = os.path.splitext(os.path.basename(path))[0]
        out_mask_path = os.path.join(results_dir, base + "_cold_mask.png")
        out_outline_path = os.path.join(results_dir, base + "_outlined_rect.png")

        cv2.imwrite(out_mask_path, mask)
        cv2.imwrite(out_outline_path, outlined)

        print(f"[OK] 輸出：{out_mask_path}")
        print(f"[OK] 輸出：{out_outline_path}")


if __name__ == "__main__":
    main()
