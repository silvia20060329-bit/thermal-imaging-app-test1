import os
from io import BytesIO

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

def cv2_to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def rgb_to_cv2(img_rgb):
    return cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)


def image_to_png_bytes(img_rgb):
    pil_img = Image.fromarray(img_rgb)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


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
    2: "水泥",
    3: "短磚",
    4: "石磚",
    5: "瓷磚",
}


def calculate_material_ratios(mask_full):
    """計算各材質佔整張 RGB 影像的像素比例。"""
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
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
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


def run_surface_inference(img_bgr, model, confidence=0.25, extend_regions=True):
    """Segment the original photo; optional extension preserves the existing workflow."""
    height, width = img_bgr.shape[:2]
    wall = np.zeros((height, width), dtype=np.uint8)
    floor = np.zeros_like(wall)
    result = model.predict(img_bgr, conf=confidence, verbose=False)[0]
    if result.masks is not None and result.boxes is not None:
        for points, class_id in zip(result.masks.xy, result.boxes.cls):
            polygon = np.asarray(points, dtype=np.int32)
            if len(polygon) < 3:
                continue
            name = str(result.names[int(class_id)]).lower()
            if "wall" in name or "牆" in name:
                cv2.fillPoly(wall, [polygon], 1)
            elif "floor" in name or "地" in name:
                cv2.fillPoly(floor, [polygon], 1)

    original_wall, original_floor = wall.copy(), floor.copy()
    if extend_regions:
        wall_any, floor_any = wall.any(axis=0), floor.any(axis=0)
        wall_top, floor_top = np.argmax(wall, axis=0), np.argmax(floor, axis=0)
        wall_bottom = height - 1 - np.argmax(wall[::-1], axis=0)
        floor_bottom = height - 1 - np.argmax(floor[::-1], axis=0)
        y = np.arange(height)[:, None]
        only_wall, only_floor = wall_any & ~floor_any, floor_any & ~wall_any
        both = wall_any & floor_any
        wall[(y < wall_top) & (only_wall | both)] = 1
        floor[(y > wall_bottom) & only_wall] = 1
        wall[(y < floor_top) & only_floor] = 1
        floor[(y > floor_bottom) & (only_floor | both)] = 1
        wall[original_floor > 0] = 0
        floor[original_wall > 0] = 0

    # Resolve overlapping predictions so reported areas are mutually exclusive.
    floor[wall > 0] = 0
    rgb = cv2_to_rgb(img_bgr)
    overlay = rgb.copy()
    overlay[wall > 0] = (255, 165, 0)
    overlay[floor > 0] = (0, 190, 220)
    visual = cv2.addWeighted(rgb, 0.55, overlay, 0.45, 0)
    return visual, wall, floor
