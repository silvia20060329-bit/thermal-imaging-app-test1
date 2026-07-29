import os
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

# High-temperature detection module
try:
    from high_detection import detect_high_regions
    HIGH_MODULE_READY = True
except Exception as e:
    HIGH_MODULE_READY = False
    HIGH_IMPORT_ERROR = e

# Cold-temperature detection module
try:
    from cold_detection import detect_cold_regions, get_default_cold_model
    COLD_MODULE_READY = True
except Exception as e:
    COLD_MODULE_READY = False
    COLD_IMPORT_ERROR = e


# =========================================================
# 0. Basic utilities & 高溫偵測專用常數與函式
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


# --- 高溫偵測專用常數 ---
HOT_PERCENT = 95 # 取亮度最高 5%
BOX_COLOR = (0, 255, 255) # 黃色
KERNEL = np.ones((5, 5), np.uint8)

# =========================================================
# 1. Mutual Information tools, from utils_mi.py
# =========================================================
def to_uint8(img):
    if img.dtype == np.uint8:
        return img
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def compute_mutual_information(a, b, bins=64, eps=1e-12, mask=None):
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
# 2. VIS/IR hybrid alignment
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


# =========================================================
# 3. U-Net material segmentation
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

# U-Net 顯示名稱：保留模型訓練時的 class ID，不改動權重對應關係
MATERIAL_DISPLAY_NAMES = {
    1: "紅磚",
    2: "短磚",
    3: "水泥",
    4: "瓷磚",
    5: "石磚",
}


def calculate_material_ratios(mask_full):
    """計算各材質佔整張對齊後 RGB 影像的像素比例。"""
    ratios = {name: 0.0 for name in MATERIAL_DISPLAY_NAMES.values()}
    ratios["背景"] = 0.0

    if mask_full is None or mask_full.size == 0:
        return ratios

    total_pixels = mask_full.size
    for class_id, display_name in MATERIAL_DISPLAY_NAMES.items():
        ratios[display_name] = round(
            np.count_nonzero(mask_full == class_id) / total_pixels * 100,
            2,
        )

    ratios["背景"] = round(
        np.count_nonzero(mask_full == 0) / total_pixels * 100,
        2,
    )
    return ratios

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
        empty_ratios = calculate_material_ratios(None)
        return img_bgr.copy(), None, empty_ratios, "U-Net 材質模型未載入。"

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
    material_ratios = calculate_material_ratios(pred_full)
    return cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR), pred_full, material_ratios, "U-Net 材質推論完成。"



# =========================================================
# 5. Streamlit app
# =========================================================
st.set_page_config(layout="wide", page_title="熱影像與可見光整合分析系統")

st.title("熱影像與可見光整合分析系統")
st.markdown(
    "上傳可見光影像與熱影像，系統會依照選擇的模組進行影像對齊、"
    "牆壁/地板分割、材質分割、高低溫偵測與資訊融合。"
)
st.info(
    "系統流程：影像輸入 → VIS/IR 對齊 → YOLO Wall/Floor → "
    "U-Net 材質分割 → High / Cold Detection → Wall/Floor × Temperature 資訊融合"
)

with st.sidebar:
    st.header("⚙️ 系統設定")

    st.subheader("1️⃣ 影像對齊")
    enable_alignment = st.checkbox("啟用 VIS/IR 自動對齊", value=True)

    st.divider()
    st.subheader("2️⃣ YOLO 牆壁 / 地板分割")
    enable_yolo = st.checkbox("啟用 YOLO 牆壁 / 地板分割", value=True)
    seg_conf = st.slider(
        "牆壁 / 地板分割信心度",
        0.05, 0.95, 0.25, 0.05,
        disabled=not enable_yolo,
    )

    st.divider()
    st.subheader("3️⃣ U-Net 材質分割")
    enable_unet = st.checkbox("啟用 U-Net 材質分割", value=True)
    unet_area_threshold = st.slider(
        "材質分割小雜訊過濾比例",
        0.000, 0.020, 0.010, 0.001,
        disabled=not enable_unet,
    )

    st.divider()
    st.subheader("4️⃣ 高溫偵測")
    enable_high = st.checkbox("啟用高溫偵測", value=True)
    high_min_area = st.slider(
        "高溫最小面積",
        0, 5000, 50, 50,
        disabled=not enable_high,
    )

    st.divider()
    st.subheader("5️⃣ 低溫偵測")
    enable_cold = st.checkbox("啟用低溫偵測", value=True)
    cold_min_area = st.slider(
        "低溫最小面積",
        0, 5000, 50, 50,
        disabled=not enable_cold,
    )
    roi_mode = st.selectbox(
        "低溫偵測範圍",
        options=[None, "bottom_half"],
        format_func=lambda x: "整張影像" if x is None else "下半部",
        disabled=not enable_cold,
    )

    st.divider()
    st.subheader("6️⃣ 資訊融合顯示")
    st.caption("可同時勾選多個結果") # High / Cold 先獨立偵測，最後才與 Wall / Floor mask 交集。

    show_floor_high = st.checkbox(
        "🔥 Floor 高溫",
        value=True,
        disabled=not (enable_yolo and enable_high),
    )
    show_floor_cold = st.checkbox(
        "❄️ Floor 低溫",
        value=True,
        disabled=not (enable_yolo and enable_cold),
    )
    show_wall_high = st.checkbox(
        "🔥 Wall 高溫",
        value=True,
        disabled=not (enable_yolo and enable_high),
    )
    show_wall_cold = st.checkbox(
        "❄️ Wall 低溫",
        value=True,
        disabled=not (enable_yolo and enable_cold),
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
            st.warning(
                "找不到 U-Net 權重：weights/unet_vis_best.pth、"
                "checkpoints/unet_vis_best.pth 或 unet_vis_best.pth，材質分割會略過。"
            )
    else:
        st.warning(f"PyTorch / torchvision / matplotlib 匯入失敗，U-Net 材質分割會略過：{TORCH_IMPORT_ERROR}")

    return models


with st.spinner("🧠 系統正在載入 AI 模型權重，請稍候..."):
    my_models = load_models()

if not HIGH_MODULE_READY:
    st.warning(f"high_detection.py 匯入失敗，高溫偵測會略過：{HIGH_IMPORT_ERROR}")

if not COLD_MODULE_READY:
    st.warning(f"cold_detection.py 匯入失敗，低溫偵測會略過：{COLD_IMPORT_ERROR}")


def mask_ratio_in_region(mask, region_mask):
    if mask is None or region_mask is None:
        return 0.0

    region = region_mask > 0
    region_area = int(np.count_nonzero(region))
    if region_area == 0:
        return 0.0

    abnormal_area = int(np.count_nonzero((mask > 0) & region))
    return round(abnormal_area / region_area * 100, 2)


def intersect_mask(mask, region_mask):
    if mask is None or region_mask is None:
        return None

    region_u8 = ((region_mask > 0).astype(np.uint8)) * 255
    return cv2.bitwise_and(mask, region_u8)


def draw_selected_temperature_layers(
    thermal_bgr,
    floor_high_mask,
    floor_cold_mask,
    wall_high_mask,
    wall_cold_mask,
    show_floor_high,
    show_floor_cold,
    show_wall_high,
    show_wall_cold,
):
    out = thermal_bgr.copy()
    overlay = out.copy()
    selected = []

    if show_floor_high and floor_high_mask is not None:
        overlay[floor_high_mask > 0] = (0, 0, 255)
        selected.append((floor_high_mask, "Floor High", (0, 255, 255)))

    if show_floor_cold and floor_cold_mask is not None:
        overlay[floor_cold_mask > 0] = (255, 0, 0)
        selected.append((floor_cold_mask, "Floor Cold", (255, 255, 0)))

    if show_wall_high and wall_high_mask is not None:
        overlay[wall_high_mask > 0] = (0, 0, 255)
        selected.append((wall_high_mask, "Wall High", (0, 255, 255)))

    if show_wall_cold and wall_cold_mask is not None:
        overlay[wall_cold_mask > 0] = (255, 0, 0)
        selected.append((wall_cold_mask, "Wall Cold", (255, 255, 0)))

    cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)

    label_y = 28
    for mask, label, color in selected:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 2)

        if contours:
            cv2.putText(
                out,
                label,
                (12, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )
            label_y += 27

    return out


def process_pipeline(
    rgb_img,
    thermal_img,
    models,
    enable_alignment_value=True,
    enable_yolo_value=True,
    enable_unet_value=True,
    enable_high_value=True,
    enable_cold_value=True,
    seg_conf_value=0.25,
    high_min_area_value=50,
    cold_min_area_value=500,
    roi_mode_value=None,
    unet_area_threshold_value=0.002,
    show_floor_high_value=True,
    show_floor_cold_value=True,
    show_wall_high_value=True,
    show_wall_cold_value=True,
):
    rgb_cv = rgb_to_cv2(rgb_img)
    thermal_cv = rgb_to_cv2(thermal_img)

    aligned_rgb, aligned_thermal, align_info = align_hybrid_images(
        rgb_cv,
        thermal_cv,
        enable_alignment=enable_alignment_value,
    )

    h_img, w_img = aligned_rgb.shape[:2]

    # YOLO Wall / Floor
    yolo_visual = aligned_rgb.copy()
    wall_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    floor_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    target_ratio = 0.0

    if not enable_yolo_value:
        yolo_desc = "YOLO 牆壁/地板分割已關閉。"

    elif models and models.get("seg") is not None:
        seg_results = models["seg"](aligned_rgb, conf=seg_conf_value)[0]
        yolo_visual = seg_results.plot()
        yolo_desc = "YOLO 牆壁/地板分割完成。"

        if seg_results.masks is not None and seg_results.boxes is not None:
            cls_names = seg_results.names

            for xy, cls_id in zip(seg_results.masks.xy, seg_results.boxes.cls):
                poly = np.asarray(xy, dtype=np.int32)
                class_id = int(cls_id)
                class_name = str(cls_names[class_id]).lower()

                if "wall" in class_name or "牆" in class_name or class_id == 1:
                    cv2.fillPoly(wall_mask, [poly], 1)

                elif "floor" in class_name or "地" in class_name or class_id == 0:
                    cv2.fillPoly(floor_mask, [poly], 1)

            union_mask = cv2.bitwise_or(wall_mask, floor_mask)
            target_ratio = round(
                np.count_nonzero(union_mask) / union_mask.size * 100,
                2,
            )
    else:
        yolo_desc = "YOLO 模型未載入。"

    # U-Net
    if enable_unet_value:
        material_visual, material_mask, material_ratios, material_desc = run_unet_material_inference(
            aligned_rgb,
            models.get("unet") if models else None,
            small_area_threshold=unet_area_threshold_value,
        )
    else:
        material_visual = aligned_rgb.copy()
        material_mask = None
        material_ratios = calculate_material_ratios(None)
        material_desc = "U-Net 材質分割已關閉。"

    # High
    high_visual = aligned_thermal.copy()
    high_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    high_ratio_full = 0.0

    if not enable_high_value:
        high_desc = "高溫偵測已關閉。"
    elif HIGH_MODULE_READY:
        try:
            high_visual, high_mask, high_ratio_full = detect_high_regions(
                aligned_thermal,
                min_area=high_min_area_value,
            )
            high_desc = "高溫偵測完成。"
        except Exception as e:
            high_desc = f"高溫偵測失敗：{e}"
    else:
        high_desc = "high_detection.py 未成功載入。"

    # Cold
    cold_visual = aligned_thermal.copy()
    cold_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    cold_ratio_full = 0.0

    if not enable_cold_value:
        cold_desc = "低溫偵測已關閉。"
    elif COLD_MODULE_READY:
        try:
            cold_model = get_default_cold_model()
            cold_visual, cold_mask = detect_cold_regions(
                aligned_thermal,
                model=cold_model,
                min_area=cold_min_area_value,
                roi_mode=roi_mode_value,
            )
            cold_ratio_full = round(
                np.count_nonzero(cold_mask) / cold_mask.size * 100,
                2,
            )
            roi_text = "整張影像" if roi_mode_value is None else "下半部"
            cold_desc = f"低溫偵測完成（固定 HSV；偵測範圍：{roi_text}）。"
        except Exception as e:
            cold_desc = f"低溫偵測失敗：{e}"
    else:
        cold_desc = "cold_detection.py 未成功載入。"

    # Information Fusion
    floor_high_mask = intersect_mask(high_mask, floor_mask)
    floor_cold_mask = intersect_mask(cold_mask, floor_mask)
    wall_high_mask = intersect_mask(high_mask, wall_mask)
    wall_cold_mask = intersect_mask(cold_mask, wall_mask)

    floor_high_ratio = mask_ratio_in_region(high_mask, floor_mask)
    floor_cold_ratio = mask_ratio_in_region(cold_mask, floor_mask)
    wall_high_ratio = mask_ratio_in_region(high_mask, wall_mask)
    wall_cold_ratio = mask_ratio_in_region(cold_mask, wall_mask)

    selected_temperature_visual = draw_selected_temperature_layers(
        aligned_thermal,
        floor_high_mask,
        floor_cold_mask,
        wall_high_mask,
        wall_cold_mask,
        show_floor_high_value,
        show_floor_cold_value,
        show_wall_high_value,
        show_wall_cold_value,
    )

    fusion_visual = cv2.addWeighted(
        yolo_visual,
        0.55,
        selected_temperature_visual,
        0.45,
        0,
    )

    selected_ratios = []
    if show_floor_high_value:
        selected_ratios.append(floor_high_ratio)
    if show_floor_cold_value:
        selected_ratios.append(floor_cold_ratio)
    if show_wall_high_value:
        selected_ratios.append(wall_high_ratio)
    if show_wall_cold_value:
        selected_ratios.append(wall_cold_ratio)

    max_selected_ratio = max(selected_ratios) if selected_ratios else 0.0

    if max_selected_ratio > 10.0:
        status_text = "⚠️ 偵測到顯著溫度異常"
    elif max_selected_ratio > 0:
        status_text = "🔎 偵測到局部溫度異常"
    else:
        status_text = "✅ 未偵測到所選區域的明顯異常"

    description = (
        f"{align_info['message']}\n\n"
        f"{yolo_desc}\n"
        f"{material_desc}\n"
        f"{high_desc} 全圖高溫候選比例：{high_ratio_full}%。\n"
        f"{cold_desc} 全圖低溫候選比例：{cold_ratio_full}%。\n\n"
        "High / Cold 與 Wall / Floor 分開運算，最後才於 Information Fusion 階段取交集。"
    )

    return {
        "aligned_rgb": cv2_to_rgb(aligned_rgb),
        "aligned_thermal": cv2_to_rgb(aligned_thermal),
        "fusion_img": cv2_to_rgb(fusion_visual),
        "selected_temp_img": cv2_to_rgb(selected_temperature_visual),
        "yolo_img": cv2_to_rgb(yolo_visual),
        "material_img": cv2_to_rgb(material_visual),
        "hot_img": cv2_to_rgb(high_visual),
        "cold_img": cv2_to_rgb(cold_visual),
        "target_ratio": target_ratio,
        "material_ratios": material_ratios,
        "high_ratio_full": high_ratio_full,
        "cold_ratio_full": cold_ratio_full,
        "floor_high_ratio": floor_high_ratio,
        "floor_cold_ratio": floor_cold_ratio,
        "wall_high_ratio": wall_high_ratio,
        "wall_cold_ratio": wall_cold_ratio,
        "status": status_text,
        "desc": description,
        "align_info": align_info,
    }


# =========================================================
# 6. Upload & Results
# =========================================================
st.markdown("### 🧩 本次啟用模組")
module_cols = st.columns(5)
module_cols[0].write("✅ Alignment" if enable_alignment else "⬜ Alignment")
module_cols[1].write("✅ YOLO" if enable_yolo else "⬜ YOLO")
module_cols[2].write("✅ U-Net" if enable_unet else "⬜ U-Net")
module_cols[3].write("✅ High" if enable_high else "⬜ High")
module_cols[4].write("✅ Cold" if enable_cold else "⬜ Cold")

st.markdown("### 📥 影像上傳")
col1, col2 = st.columns(2)

with col1:
    rgb_file = st.file_uploader(
        "上傳可見光影像 (RGB)",
        type=["jpg", "png", "jpeg"],
        key="rgb",
    )
    if rgb_file:
        st.image(rgb_file, caption="RGB 影像預覽", use_container_width=True)

with col2:
    thermal_file = st.file_uploader(
        "上傳熱影像 (Thermal)",
        type=["jpg", "png", "jpeg"],
        key="thermal",
    )
    if thermal_file:
        st.image(thermal_file, caption="Thermal 影像預覽", use_container_width=True)

st.divider()

if st.button("🚀 開始分析", type="primary", use_container_width=True):
    if not (rgb_file and thermal_file):
        st.error("請確認兩張影像皆已成功上傳！")
    else:
        img_rgb = Image.open(rgb_file).convert("RGB")
        img_thermal = Image.open(thermal_file).convert("RGB")

        with st.status("🧠 系統分析中...", expanded=True) as status_msg:
            st.write("📐 正在進行 VIS/IR 自動對齊..." if enable_alignment else "📐 VIS/IR 自動對齊已關閉。")
            st.write("🧱 正在進行 YOLO 牆壁/地板分割..." if enable_yolo else "🧱 YOLO 牆壁/地板分割已關閉。")
            st.write("🧩 正在進行 U-Net 材質分割..." if enable_unet else "🧩 U-Net 材質分割已關閉。")
            st.write("🔥 正在執行 high_detection.py..." if enable_high else "🔥 高溫偵測已關閉。")
            st.write("❄️ 正在執行 cold_detection.py..." if enable_cold else "❄️ 低溫偵測已關閉。")
            st.write("🔀 正在進行 Wall/Floor × High/Cold 資訊融合...")

            results = process_pipeline(
                img_rgb,
                img_thermal,
                my_models,
                enable_alignment_value=enable_alignment,
                enable_yolo_value=enable_yolo,
                enable_unet_value=enable_unet,
                enable_high_value=enable_high,
                enable_cold_value=enable_cold,
                seg_conf_value=seg_conf,
                high_min_area_value=high_min_area,
                cold_min_area_value=cold_min_area,
                roi_mode_value=roi_mode,
                unet_area_threshold_value=unet_area_threshold,
                show_floor_high_value=show_floor_high,
                show_floor_cold_value=show_floor_cold,
                show_wall_high_value=show_wall_high,
                show_wall_cold_value=show_wall_cold,
            )

            status_msg.update(label="分析處理完成！", state="complete", expanded=False)

        st.markdown("### 📊 分析結果輸出")
        res_col1, res_col2 = st.columns([2, 1])

        with res_col1:
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "多圖層總覽",
                "區域 × 溫度結果",
                "對齊後 RGB / Thermal",
                "圖層一：YOLO 牆壁/地板",
                "圖層二：U-Net 材質",
                "圖層三：高溫原始結果",
                "圖層四：低溫原始結果",
            ])

            with tab1:
                st.image(results["fusion_img"], caption="多圖層融合總覽", use_container_width=True)

            with tab2:
                st.image(
                    results["selected_temp_img"],
                    caption="依左側四個勾選項目顯示的 Wall/Floor × High/Cold 結果",
                    use_container_width=True,
                )

            with tab3:
                c1, c2 = st.columns(2)
                with c1:
                    st.image(results["aligned_rgb"], caption="對齊後 RGB", use_container_width=True)
                with c2:
                    st.image(results["aligned_thermal"], caption="對齊後 Thermal", use_container_width=True)

            with tab4:
                st.image(results["yolo_img"], caption="YOLO Wall/Floor 分割結果", use_container_width=True)

            with tab5:
                if enable_unet:
                    st.image(results["material_img"], caption="U-Net 材質分割結果", use_container_width=True)
                else:
                    st.info("本次分析未啟用 U-Net 材質分割。")

            with tab6:
                if enable_high:
                    st.image(results["hot_img"], caption="high_detection.py 原始高溫結果", use_container_width=True)
                else:
                    st.info("本次分析未啟用高溫偵測。")

            with tab7:
                if enable_cold:
                    st.image(results["cold_img"], caption="cold_detection.py 原始低溫結果", use_container_width=True)
                else:
                    st.info("本次分析未啟用低溫偵測。")

        with res_col2:
            st.subheader("📝 結構狀態量化報告")
            st.metric(label="當前評估狀態", value=results["status"])

            st.markdown(f"""
**區域 × 溫度異常比例：**

* 🔥 **Floor 高溫**：{results['floor_high_ratio']}%
* ❄️ **Floor 低溫**：{results['floor_cold_ratio']}%
* 🔥 **Wall 高溫**：{results['wall_high_ratio']}%
* ❄️ **Wall 低溫**：{results['wall_cold_ratio']}%

**U-Net 材質分布比例（佔整張圖）：**

* **紅磚(material_1)**：{results['material_ratios']['紅磚']}%
* **短磚(floor_2)   **：{results['material_ratios']['短磚']}%
* **水泥(material_3)**：{results['material_ratios']['水泥']}%
* **瓷磚(floor_4)   **：{results['material_ratios']['瓷磚']}%
* **石磚(wall_2)    **：{results['material_ratios']['石磚']}%

**其他資訊：**

* **YOLO 牆壁/地板目標區域比例**：{results['target_ratio']}%
* **全圖高溫候選比例**：{results['high_ratio_full']}%
* **全圖低溫候選比例**：{results['cold_ratio_full']}%

**對齊資訊：**

* **MI 分數**：{results['align_info'].get('mi_score')}
* **Crop Box**：{results['align_info'].get('crop_box')}

**詳細判定說明：**

{results['desc']}
""")

            st.download_button(
                "下載多圖層總覽圖",
                data=image_to_png_bytes(results["fusion_img"]),
                file_name="fusion_result.png",
                mime="image/png",
                use_container_width=True,
            )

            st.download_button(
                "下載區域 × 溫度結果",
                data=image_to_png_bytes(results["selected_temp_img"]),
                file_name="region_temperature_result.png",
                mime="image/png",
                use_container_width=True,
            )

            st.download_button(
                "下載 YOLO 分割圖",
                data=image_to_png_bytes(results["yolo_img"]),
                file_name="yolo_result.png",
                mime="image/png",
                use_container_width=True,
            )

            if enable_unet:
                st.download_button(
                    "下載 U-Net 材質分割圖",
                    data=image_to_png_bytes(results["material_img"]),
                    file_name="material_result.png",
                    mime="image/png",
                    use_container_width=True,
                )

            if enable_high:
                st.download_button(
                    "下載高溫原始結果",
                    data=image_to_png_bytes(results["hot_img"]),
                    file_name="hot_result.png",
                    mime="image/png",
                    use_container_width=True,
                )

            if enable_cold:
                st.download_button(
                    "下載低溫原始結果",
                    data=image_to_png_bytes(results["cold_img"]),
                    file_name="cold_result.png",
                    mime="image/png",
                    use_container_width=True,
                )
