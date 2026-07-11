import os
import glob
import tempfile
from io import BytesIO

import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw

# YOLO segmentation
try:
    from ultralytics import YOLO
    YOLO_READY = True
except Exception as e:
    YOLO_READY = False
    YOLO_IMPORT_ERROR = e

# U-Net material segmentation
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    import matplotlib.pyplot as plt
    TORCH_READY = True
except Exception as e:
    TORCH_READY = False
    TORCH_IMPORT_ERROR = e

# Optional cold detection module
try:
    from cold_detection import learn_cold_model, detect_cold_regions
    COLD_MODULE_READY = True
except Exception as e:
    COLD_MODULE_READY = False
    COLD_IMPORT_ERROR = e


# =========================================================
# 0. Basic utilities
# =========================================================
def cv2_to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def rgb_to_cv2(img_rgb):
    return cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)


def image_to_png_bytes(img_rgb):
    pil_img = Image.fromarray(img_rgb)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


# =========================================================
# 1. Mutual Information tools, from utils_mi.py
# =========================================================
def to_uint8(img):
    """Ensure image is uint8 for histogram and display."""
    if img.dtype == np.uint8:
        return img
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def compute_mutual_information(a, b, bins=64, eps=1e-12, mask=None):
    """Compute normalized mutual information between two grayscale images."""
    a = to_uint8(a)
    b = to_uint8(b)

    if mask is not None:
        mask = mask > 0
        if not np.any(mask):
            return 0.0
        a = a[mask]
        b = b[mask]

    H, _, _ = np.histogram2d(
        a.ravel(), b.ravel(), bins=bins, range=[[0, 255], [0, 255]]
    )
    pxy = H / max(H.sum(), eps)
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)

    Hx = -np.sum(px * np.log(px + eps))
    Hy = -np.sum(py * np.log(py + eps))
    Hxy = -np.sum(pxy * np.log(pxy + eps))

    mi = (Hx + Hy) / max(Hxy, eps)
    return float(mi)


# =========================================================
# 2. VIS/IR hybrid alignment, adapted from align_hybrid_vis2ir.py
# =========================================================
def align_hybrid_images(
    vis_bgr,
    ir_bgr,
    enable_alignment=True,
    base_vis_size=(2048, 1536),
    base_crop=(554, 415, 939, 704),
    search_dx=40,
    search_dy=160,
    step_x=8,
    step_y=16,
):
    """
    Align visible image to infrared image by MI-based crop search.

    Input:
        vis_bgr: OpenCV BGR visible image
        ir_bgr: OpenCV BGR infrared/thermal image
    Output:
        aligned_vis_bgr: visible image cropped and resized to IR size
        aligned_ir_bgr: original IR image, unchanged
        info: alignment metadata
    """
    if not enable_alignment:
        return vis_bgr.copy(), ir_bgr.copy(), {
            "enabled": False,
            "mi_score": None,
            "crop_box": None,
            "message": "影像對齊已關閉，直接使用原始影像。",
        }

    if vis_bgr is None or ir_bgr is None:
        return vis_bgr, ir_bgr, {
            "enabled": False,
            "mi_score": None,
            "crop_box": None,
            "message": "影像讀取失敗，略過對齊。",
        }

    vis_h, vis_w = vis_bgr.shape[:2]
    ir_h, ir_w = ir_bgr.shape[:2]

    # Scale the original paper/reference crop if input VIS is not 2048x1536.
    base_w, base_h = base_vis_size
    start_x, start_y, tw, th = base_crop
    sx = vis_w / base_w
    sy = vis_h / base_h

    start_x = int(round(start_x * sx))
    start_y = int(round(start_y * sy))
    tw = int(round(tw * sx))
    th = int(round(th * sy))

    search_dx = max(1, int(round(search_dx * sx)))
    search_dy = max(1, int(round(search_dy * sy)))
    step_x = max(1, int(round(step_x * sx)))
    step_y = max(1, int(round(step_y * sy)))

    # If crop is invalid for smaller images, fall back to full resize.
    if tw <= 0 or th <= 0 or tw > vis_w or th > vis_h:
        aligned_vis = cv2.resize(vis_bgr, (ir_w, ir_h))
        return aligned_vis, ir_bgr.copy(), {
            "enabled": False,
            "mi_score": None,
            "crop_box": (0, 0, vis_w, vis_h),
            "message": "VIS 尺寸不符合基準裁切範圍，已改用整張縮放。",
        }

    ir_gray = cv2.cvtColor(ir_bgr, cv2.COLOR_BGR2GRAY) if len(ir_bgr.shape) == 3 else ir_bgr

    best_mi = -1.0
    best_coord = None

    for dy in range(-search_dy, search_dy + 1, step_y):
        for dx in range(-search_dx, search_dx + 1, step_x):
            nx, ny = start_x + dx, start_y + dy
            if nx < 0 or ny < 0 or nx + tw > vis_w or ny + th > vis_h:
                continue

            patch = vis_bgr[ny : ny + th, nx : nx + tw]
            patch_res = cv2.resize(patch, (ir_w, ir_h))
            patch_gray = cv2.cvtColor(patch_res, cv2.COLOR_BGR2GRAY)

            mi = compute_mutual_information(patch_gray, ir_gray)
            if mi > best_mi:
                best_mi = mi
                best_coord = (nx, ny)

    if best_coord is None:
        aligned_vis = cv2.resize(vis_bgr, (ir_w, ir_h))
        return aligned_vis, ir_bgr.copy(), {
            "enabled": False,
            "mi_score": None,
            "crop_box": (0, 0, vis_w, vis_h),
            "message": "找不到有效搜尋區域，已改用整張縮放。",
        }

    final_x, final_y = best_coord
    aligned_vis = cv2.resize(vis_bgr[final_y : final_y + th, final_x : final_x + tw], (ir_w, ir_h))

    return aligned_vis, ir_bgr.copy(), {
        "enabled": True,
        "mi_score": round(best_mi, 4),
        "crop_box": (final_x, final_y, tw, th),
        "message": f"已完成 MI 對齊，MI={best_mi:.4f}，crop={final_x},{final_y},{tw},{th}",
    }


def align_hybrid_logic(vis_path, ir_path, out_dir):
    """Path-based batch alignment kept for local run_hybrid_batch.py compatibility."""
    vis = cv2.imread(vis_path)
    ir = cv2.imread(ir_path)
    if vis is None or ir is None:
        return False

    vis_aligned, ir_aligned, info = align_hybrid_images(vis, ir, enable_alignment=True)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(vis_path))[0]
    cv2.imwrite(os.path.join(out_dir, f"{stem}_aligned.png"), vis_aligned)
    overlay = cv2.addWeighted(vis_aligned, 0.5, ir_aligned, 0.5, 0)
    cv2.imwrite(os.path.join(out_dir, f"{stem}_overlay.png"), overlay)
    print(f"完成 {stem}: {info['message']}")
    return True


def run_batch_alignment(base_dir=None):
    """Batch process data/visible and data/infrared with same filenames."""
    base_dir = base_dir or os.getcwd()
    vis_dir = os.path.join(base_dir, "data", "visible")
    ir_dir = os.path.join(base_dir, "data", "infrared")
    out_dir = os.path.join(base_dir, "data", "output")

    if not os.path.exists(vis_dir) or not os.path.exists(ir_dir):
        print("錯誤: 找不到 data/visible 或 data/infrared 資料夾")
        return

    vis_files = [
        f for f in os.listdir(vis_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    print(f"開始執行混合對齊處理，共 {len(vis_files)} 張...")

    for fname in vis_files:
        vis_path = os.path.join(vis_dir, fname)
        ir_path = os.path.join(ir_dir, fname)
        if os.path.exists(ir_path):
            align_hybrid_logic(vis_path, ir_path, out_dir)
        else:
            print(f"跳過: {fname} (找不到對應 IR)")


# =========================================================
# 3. U-Net material segmentation, adapted from inference_clean_0.2.py
# =========================================================
if TORCH_READY:
    class DoubleConv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True), nn.Dropout2d(p=0.1),
                nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True), nn.Dropout2d(p=0.1),
            )

        def forward(self, x):
            return self.net(x)


    class UNet(nn.Module):
        def __init__(self, in_ch=3, n_classes=6):
            super().__init__()
            self.down1 = DoubleConv(in_ch, 32); self.pool1 = nn.MaxPool2d(2)
            self.down2 = DoubleConv(32, 64); self.pool2 = nn.MaxPool2d(2)
            self.down3 = DoubleConv(64, 128); self.pool3 = nn.MaxPool2d(2)
            self.down4 = DoubleConv(128, 256); self.pool4 = nn.MaxPool2d(2)
            self.bottleneck = nn.Sequential(DoubleConv(256, 512), nn.Dropout2d(p=0.3))
            self.up4 = nn.ConvTranspose2d(512, 256, 2, 2); self.conv4 = DoubleConv(512, 256)
            self.up3 = nn.ConvTranspose2d(256, 128, 2, 2); self.conv3 = DoubleConv(256, 128)
            self.up2 = nn.ConvTranspose2d(128, 64, 2, 2); self.conv2 = DoubleConv(128, 64)
            self.up1 = nn.ConvTranspose2d(64, 32, 2, 2); self.conv1 = DoubleConv(64, 32)
            self.out_conv = nn.Conv2d(32, n_classes, kernel_size=1)

        def forward(self, x):
            c1 = self.down1(x); p1 = self.pool1(c1)
            c2 = self.down2(p1); p2 = self.pool2(c2)
            c3 = self.down3(p2); p3 = self.pool3(c3)
            c4 = self.down4(p3); p4 = self.pool4(c4)
            bn = self.bottleneck(p4)
            u4 = self.up4(bn); u4 = torch.cat([u4, c4], 1); c4 = self.conv4(u4)
            u3 = self.up3(c4); u3 = torch.cat([u3, c3], 1); c3 = self.conv3(u3)
            u2 = self.up2(c3); u2 = torch.cat([u2, c2], 1); c2 = self.conv2(u2)
            u1 = self.up1(c2); u1 = torch.cat([u1, c1], 1); c1 = self.conv1(u1)
            return self.out_conv(c1)


LABEL_MAP = {"material_1": 1, "material_2": 2, "floor_2": 3, "wall_3": 4, "floor_4": 5}
ID_TO_NAME = {v: k for k, v in LABEL_MAP.items()}


def remove_small_components(mask, n_classes=6, area_threshold_ratio=0.002, ignore_index=0):
    filtered = mask.copy()
    min_area = mask.size * area_threshold_ratio

    for cls in range(n_classes):
        if cls == ignore_index:
            continue
        cls_mask = (filtered == cls).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cls_mask, connectivity=8)
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area < min_area:
                filtered[labels == label_id] = ignore_index

    return filtered


def draw_visual_result(img_cv, mask_full, info_text=""):
    if not TORCH_READY:
        return Image.fromarray(cv2_to_rgb(img_cv))

    cmap = plt.get_cmap("tab10")
    h_orig, w_orig, _ = img_cv.shape
    overlay_img = img_cv.copy()
    occupied_rects = []
    detected_list = []

    for i in range(1, 6):
        if np.any(mask_full == i):
            color_bgr = [int(c * 255) for c in cmap(i)[:3][::-1]]
            detected_list.append(ID_TO_NAME.get(i, f"class_{i}"))

            mask_indices = mask_full == i
            overlay_img[mask_indices] = (
                overlay_img[mask_indices] * 0.5 + np.array(color_bgr) * 0.5
            ).astype(np.uint8)

            mask_cls = (mask_full == i).astype(np.uint8)
            contours, _ = cv2.findContours(mask_cls, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay_img, contours, -1, color_bgr, 3)

            for cnt in contours:
                if cv2.contourArea(cnt) > (w_orig * h_orig * 0.005):
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    label = ID_TO_NAME.get(i, f"class_{i}")
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

                    label_x, label_y = x, y - 10
                    rect_w, rect_h = tw + 10, th + 10

                    if label_x < 0:
                        label_x = 5
                    if label_x + rect_w > w_orig:
                        label_x = w_orig - rect_w - 5
                    if label_y - rect_h < 0:
                        label_y = rect_h + 10

                    conflict = True
                    while conflict:
                        conflict = False
                        for ox, oy, ow, oh in occupied_rects:
                            overlap = not (
                                label_x + rect_w < ox or label_x > ox + ow
                                or label_y < oy - oh or label_y - rect_h > oy
                            )
                            if overlap:
                                label_y += rect_h + 5
                                conflict = True
                                break
                        if label_y > h_orig - 5:
                            label_y = h_orig - 5
                            break

                    occupied_rects.append((label_x, label_y, rect_w, rect_h))
                    cv2.rectangle(overlay_img, (label_x, label_y - rect_h),
                                  (label_x + rect_w, label_y), color_bgr, -1)
                    cv2.putText(overlay_img, label, (label_x + 5, label_y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                                2, cv2.LINE_AA)

    res_pil = Image.fromarray(cv2_to_rgb(overlay_img))
    draw = ImageDraw.Draw(res_pil)
    draw.text((20, 20), info_text, fill="yellow")
    for idx, mat in enumerate(detected_list):
        draw.text((20, 50 + idx * 22), f"- {mat}", fill="white")
    return res_pil


def load_unet_model(model_path):
    if not TORCH_READY or not os.path.exists(model_path):
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_classes=6).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    transform = T.Compose([T.Resize((512, 512)), T.ToTensor()])
    return {"model": model, "device": device, "transform": transform}


def run_unet_material_inference(img_bgr, unet_bundle, small_area_threshold=0.002):
    if unet_bundle is None or not TORCH_READY:
        return img_bgr.copy(), None, 0.0, "U-Net 材質模型未載入。"

    img_pil = Image.fromarray(cv2_to_rgb(img_bgr)).convert("RGB")
    w_orig, h_orig = img_pil.size
    img_tensor = unet_bundle["transform"](img_pil).unsqueeze(0).to(unet_bundle["device"])

    with torch.no_grad():
        logits = unet_bundle["model"](img_tensor)
        pred_512 = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    pred_512_blur = cv2.medianBlur(pred_512, 5)
    pred_512_filtered = remove_small_components(
        pred_512_blur,
        n_classes=6,
        area_threshold_ratio=small_area_threshold,
    )
    pred_full = cv2.resize(pred_512_filtered, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    pred_pil = draw_visual_result(img_bgr, pred_full, info_text="U-Net Material Prediction")
    pred_rgb = np.array(pred_pil)
    material_ratio = round((np.count_nonzero(pred_full) / pred_full.size) * 100, 2)
    return cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR), pred_full, material_ratio, "U-Net 材質推論完成。"


# =========================================================
# 4. Cold detection helper
# =========================================================
def learn_cold_model_from_current_image(img_bgr, cold_quantile_value):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cv2.imwrite(tmp_path, img_bgr)
        model = learn_cold_model(
            [tmp_path],
            sample_fraction=0.1,
            cold_quantile=cold_quantile_value,
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return model


# =========================================================
# 5. Streamlit app
# =========================================================
st.set_page_config(layout="wide", page_title="熱影像與可見光整合分析系統")

st.title("熱影像與可見光整合分析系統")
st.markdown("上傳可見光影像與熱影像，系統會自動進行影像對齊、牆壁/地板分割、材質分割、高溫異常偵測與低溫異常偵測。")
st.info("系統流程：影像輸入 → VIS/IR 影像對齊 → YOLO 牆壁/地板分割 → U-Net 材質分割 → 高溫/低溫異常偵測 → 多圖層結果輸出")

with st.sidebar:
    st.header("⚙️ 偵測參數")

    enable_alignment = st.checkbox("啟用 VIS/IR 自動對齊", value=True)

    st.subheader("YOLO 牆壁/地板分割")
    seg_conf = st.slider("牆壁/地板分割信心度", 0.05, 0.95, 0.25, 0.05)

    st.subheader("U-Net 材質分割")
    unet_area_threshold = st.slider("材質分割小雜訊過濾比例", 0.000, 0.010, 0.002, 0.001)

    st.subheader("高溫偵測")
    high_threshold = st.slider("高溫亮度門檻", 0, 255, 150, 1)
    high_min_area = st.slider("高溫最小面積", 0, 5000, 50, 50)

    st.subheader("低溫偵測")
    cold_quantile = st.slider("低溫候選比例", 0.05, 0.50, 0.10, 0.05)
    cold_min_area = st.slider("低溫最小面積", 0, 5000, 1000, 50)
    roi_mode = st.selectbox(
        "低溫偵測範圍",
        options=[None, "bottom_half", "bottom_100"],
        format_func=lambda x: "整張影像" if x is None else x,
    )


@st.cache_resource
def load_models():
    models = {}

    if YOLO_READY:
        yolo_path = "weights/floor_wall_seg.pt"
        if os.path.exists(yolo_path):
            models["seg"] = YOLO(yolo_path)
        else:
            st.warning(f"找不到 YOLO 權重：{yolo_path}，牆壁/地板分割會略過。")
    else:
        st.warning(f"ultralytics 匯入失敗，YOLO 分割會略過：{YOLO_IMPORT_ERROR}")

    if TORCH_READY:
        candidate_paths = [
            "weights/unet_vis_best.pth",
            "checkpoints/unet_vis_best.pth",
            "unet_vis_best.pth",
        ]
        unet_path = next((p for p in candidate_paths if os.path.exists(p)), None)
        if unet_path:
            models["unet"] = load_unet_model(unet_path)
        else:
            st.warning("找不到 U-Net 權重：weights/unet_vis_best.pth 或 checkpoints/unet_vis_best.pth，材質分割會略過。")
    else:
        st.warning(f"PyTorch / torchvision / matplotlib 匯入失敗，U-Net 材質分割會略過：{TORCH_IMPORT_ERROR}")

    return models


with st.spinner("🧠 系統正在載入 AI 模型權重，請稍候..."):
    my_models = load_models()

if not COLD_MODULE_READY:
    st.warning(f"cold_detection.py 匯入失敗，低溫偵測會暫時略過。錯誤：{COLD_IMPORT_ERROR}")


def process_pipeline(
    rgb_img,
    thermal_img,
    models,
    enable_alignment_value=True,
    seg_conf_value=0.25,
    high_threshold_value=200,
    high_min_area_value=50,
    cold_quantile_value=0.2,
    cold_min_area_value=500,
    roi_mode_value=None,
    unet_area_threshold_value=0.002,
):
    rgb_cv = rgb_to_cv2(rgb_img)
    thermal_cv = rgb_to_cv2(thermal_img)

    # Stage 1: VIS/IR alignment
    aligned_rgb, aligned_thermal, align_info = align_hybrid_images(
        rgb_cv,
        thermal_cv,
        enable_alignment=enable_alignment_value,
    )

    # Stage 2: YOLO wall/floor segmentation
    yolo_visual = aligned_rgb.copy()
    target_ratio = 0.0
    yolo_desc = "YOLO 牆壁/地板模型未載入。"

    if models and models.get("seg") is not None:
        seg_results = models["seg"](aligned_rgb, conf=seg_conf_value)[0]
        yolo_visual = seg_results.plot()
        yolo_desc = "YOLO 牆壁/地板分割完成。"

        if seg_results.masks is not None:
            masks = seg_results.masks.data.cpu().numpy()
            union_mask = np.any(masks > 0.5, axis=0).astype(np.uint8)
            target_ratio = round((np.count_nonzero(union_mask) / union_mask.size) * 100, 2)

    # Stage 3: U-Net material segmentation
    material_visual, material_mask, material_ratio, material_desc = run_unet_material_inference(
        aligned_rgb,
        models.get("unet") if models else None,
        small_area_threshold=unet_area_threshold_value,
    )

    # Stage 4: high-temperature detection
    gray_thermal = cv2.cvtColor(aligned_thermal, cv2.COLOR_BGR2GRAY)
    _, high_temp_mask = cv2.threshold(gray_thermal, high_threshold_value, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(high_temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hot_visual = aligned_thermal.copy()
    high_temp_area = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= high_min_area_value:
            high_temp_area += area
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(hot_visual, center, radius, (0, 0, 255), 3)
            cv2.putText(
                hot_visual,
                "High Temp",
                (center[0] - 40, max(center[1] - radius - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

    total_area = gray_thermal.shape[0] * gray_thermal.shape[1]
    high_temp_ratio = round((high_temp_area / total_area) * 100, 2)

    # Stage 5: low-temperature detection
    cold_visual = aligned_thermal.copy()
    cold_ratio = 0.0
    cold_desc = "低溫偵測模組尚未啟用。"

    if COLD_MODULE_READY:
        try:
            train_paths = []
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                train_paths.extend(glob.glob(os.path.join("cold_train", ext)))

            if len(train_paths) > 0:
                cold_model = learn_cold_model(
                    sorted(train_paths),
                    sample_fraction=0.1,
                    cold_quantile=cold_quantile_value,
                )
                cold_desc = f"已使用 cold_train/ 內 {len(train_paths)} 張影像建立低溫模型。"
            else:
                cold_model = learn_cold_model_from_current_image(aligned_thermal, cold_quantile_value)
                cold_desc = "未找到 cold_train/，已使用本次上傳的熱影像自動估計低溫模型。"

            cold_visual, cold_mask = detect_cold_regions(
                aligned_thermal,
                cold_model,
                min_area=cold_min_area_value,
                roi_mode=roi_mode_value,
            )
            cold_ratio = round((np.count_nonzero(cold_mask) / cold_mask.size) * 100, 2)
        except Exception as e:
            cold_desc = f"低溫偵測失敗：{e}"
            cold_visual = aligned_thermal.copy()
            cold_ratio = 0.0

    # Stage 6: fusion overview
    fusion_visual = yolo_visual.copy()
    # Add high/low thermal contours onto fusion overview after resizing if needed.
    if hot_visual.shape[:2] == fusion_visual.shape[:2]:
        fusion_visual = cv2.addWeighted(fusion_visual, 0.75, hot_visual, 0.25, 0)

    # Stage 7: status decision
    if cold_ratio > 10.0:
        status_text = "⚠️ 疑似含水"
    elif high_temp_ratio > 10.0:
        status_text = "⚠️ 高溫異常"
    else:
        status_text = "✅ 正常"

    description = (
        f"{align_info['message']}\n\n"
        f"{yolo_desc}\n"
        f"{material_desc}\n"
        f"高溫異常面積佔比為 {high_temp_ratio}%，低溫異常面積佔比為 {cold_ratio}%。\n\n"
        f"{cold_desc}\n\n"
        "低溫區域可能與含水、陰影或材料熱特性有關；高溫區域可能與日照、反射或局部熱源有關，仍需搭配現場條件判讀。"
    )

    return {
        "aligned_rgb": cv2_to_rgb(aligned_rgb),
        "aligned_thermal": cv2_to_rgb(aligned_thermal),
        "fusion_img": cv2_to_rgb(fusion_visual),
        "yolo_img": cv2_to_rgb(yolo_visual),
        "material_img": cv2_to_rgb(material_visual),
        "hot_img": cv2_to_rgb(hot_visual),
        "cold_img": cv2_to_rgb(cold_visual),
        "target_ratio": target_ratio,
        "material_ratio": material_ratio,
        "high_ratio": high_temp_ratio,
        "cold_ratio": cold_ratio,
        "status": status_text,
        "desc": description,
        "align_info": align_info,
    }


st.markdown("### 📥 影像上傳")
col1, col2 = st.columns(2)

with col1:
    rgb_file = st.file_uploader("上傳可見光影像 (RGB)", type=["jpg", "png", "jpeg"], key="rgb")
    if rgb_file:
        st.image(rgb_file, caption="RGB 影像預覽", use_container_width=True)

with col2:
    thermal_file = st.file_uploader("上傳熱影像 (Thermal)", type=["jpg", "png", "jpeg"], key="thermal")
    if thermal_file:
        st.image(thermal_file, caption="Thermal 影像預覽", use_container_width=True)

st.divider()

if st.button("🚀 開始分析", type="primary", use_container_width=True):
    if rgb_file and thermal_file:
        img_rgb = Image.open(rgb_file).convert("RGB")
        img_thermal = Image.open(thermal_file).convert("RGB")

        with st.status("🧠 AI 模型深度分析中...", expanded=True) as status_msg:
            st.write("🔄 正在讀取影像...")
            st.write("📐 正在進行 VIS/IR 影像對齊...")
            st.write("🧱 正在進行 YOLO 牆壁/地板分割...")
            st.write("🧩 正在進行 U-Net 材質分割...")
            st.write("🔥 正在進行高溫異常偵測...")
            st.write("❄️ 正在進行低溫異常偵測...")

            results = process_pipeline(
                img_rgb,
                img_thermal,
                my_models,
                enable_alignment_value=enable_alignment,
                seg_conf_value=seg_conf,
                high_threshold_value=high_threshold,
                high_min_area_value=high_min_area,
                cold_quantile_value=cold_quantile,
                cold_min_area_value=cold_min_area,
                roi_mode_value=roi_mode,
                unet_area_threshold_value=unet_area_threshold,
            )

            status_msg.update(label="分析處理完成！", state="complete", expanded=False)

        st.markdown("### 📊 分析結果輸出")
        res_col1, res_col2 = st.columns([2, 1])

        with res_col1:
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "多圖層總覽",
                "對齊後 RGB / Thermal",
                "圖層一：YOLO 牆壁/地板分割",
                "圖層二：U-Net 材質分割",
                "圖層三：高溫異常區",
                "圖層四：低溫異常區",
            ])

            with tab1:
                st.image(results["fusion_img"], caption="融合分析總覽圖", use_container_width=True)
            with tab2:
                c1, c2 = st.columns(2)
                with c1:
                    st.image(results["aligned_rgb"], caption="對齊後 RGB", use_container_width=True)
                with c2:
                    st.image(results["aligned_thermal"], caption="對齊後 Thermal", use_container_width=True)
            with tab3:
                st.image(results["yolo_img"], caption="YOLO 牆壁/地板語意分割結果", use_container_width=True)
            with tab4:
                st.image(results["material_img"], caption="U-Net 材質分割結果", use_container_width=True)
            with tab5:
                st.image(results["hot_img"], caption="高溫異常區域圈選結果", use_container_width=True)
            with tab6:
                st.image(results["cold_img"], caption="低溫異常區域圈選結果", use_container_width=True)

        with res_col2:
            st.subheader("📝 結構狀態量化報告")
            st.metric(label="當前評估狀態", value=results["status"])

            st.markdown(f"""
            **數據統計：**
            * **YOLO 牆壁/地板目標區域比例**：{results["target_ratio"]}%
            * **U-Net 材質區域比例**：{results["material_ratio"]}%
            * **高溫異常面積佔比**：{results["high_ratio"]}%
            * **低溫異常面積佔比**：{results["cold_ratio"]}%

            **對齊資訊：**
            * **MI 分數**：{results["align_info"].get("mi_score")}
            * **Crop Box**：{results["align_info"].get("crop_box")}

            **詳細判定說明：**

            {results["desc"]}
            """)

            st.download_button(
                label="下載多圖層總覽圖",
                data=image_to_png_bytes(results["fusion_img"]),
                file_name="fusion_result.png",
                mime="image/png",
                use_container_width=True,
            )
            st.download_button(
                label="下載 YOLO 分割圖",
                data=image_to_png_bytes(results["yolo_img"]),
                file_name="yolo_result.png",
                mime="image/png",
                use_container_width=True,
            )
            st.download_button(
                label="下載 U-Net 材質分割圖",
                data=image_to_png_bytes(results["material_img"]),
                file_name="material_result.png",
                mime="image/png",
                use_container_width=True,
            )
            st.download_button(
                label="下載高溫異常圖",
                data=image_to_png_bytes(results["hot_img"]),
                file_name="hot_result.png",
                mime="image/png",
                use_container_width=True,
            )
            st.download_button(
                label="下載低溫異常圖",
                data=image_to_png_bytes(results["cold_img"]),
                file_name="cold_result.png",
                mime="image/png",
                use_container_width=True,
            )
    else:
        st.error("請確認兩張影像皆已成功上傳！")
