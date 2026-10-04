from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

import surface_models as models


ROOT = Path(__file__).resolve().parent


@st.cache_resource
def load_models():
    if not models.YOLO_READY:
        raise RuntimeError(f"牆壁／地板模型套件無法載入：{models.YOLO_IMPORT_ERROR}")
    if not models.TORCH_READY:
        raise RuntimeError(f"材質模型套件無法載入：{models.TORCH_IMPORT_ERROR}")
    surface_path = ROOT / "weights" / "floor_wall_seg.pt"
    material_path = ROOT / "weights" / "unet_vis_best.pth"
    for path in (surface_path, material_path):
        if not path.is_file():
            raise FileNotFoundError(f"找不到模型權重：{path}")
    return models.YOLO(str(surface_path)), models.load_unet_model(str(material_path))


def analyze(rgb, confidence, area_threshold, extend_regions):
    surface_model, material_model = load_models()
    bgr = models.rgb_to_cv2(rgb)
    surface_image, wall, floor = models.run_surface_inference(
        bgr, surface_model, confidence, extend_regions
    )
    material_image, material_mask, ratios, _ = models.run_unet_material_inference(
        bgr, material_model, area_threshold
    )
    return {
        "original": rgb,
        "surface": surface_image,
        "material": models.cv2_to_rgb(material_image),
        "ratios": ratios,
        "wall_ratio": round(float(np.count_nonzero(wall) / wall.size * 100), 2),
        "floor_ratio": round(float(np.count_nonzero(floor) / floor.size * 100), 2),
    }


def main():
    st.set_page_config(page_title="牆壁、地板與材質辨識", layout="wide")
    st.title("牆壁、地板與材質辨識")
    st.write("上傳一張可見光照片，查看牆壁、地板及各種材質的分割結果。")
    confidence = 0.05
    extend_regions = True
    area_threshold = 0.002

    uploaded = st.file_uploader("上傳可見光照片", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.session_state.pop("surface_result", None)
        st.info("請先上傳照片，再按「開始辨識」。")
        return
    raw = uploaded.getvalue()
    result_key = (sha256(raw).hexdigest(), confidence, area_threshold, extend_regions)
    saved = st.session_state.get("surface_result")
    if saved is not None and saved[0] != result_key:
        st.session_state.pop("surface_result", None)
        saved = None
    try:
        with Image.open(BytesIO(raw)) as photo:
            rgb = np.array(ImageOps.exif_transpose(photo).convert("RGB"))
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        st.error(f"無法讀取照片，請重新上傳 JPG 或 PNG 檔案：{exc}")
        return

    if saved is None:
        st.image(rgb, caption="照片預覽", use_column_width=True)
    if st.button("開始辨識", type="primary"):
        st.session_state.pop("surface_result", None)
        try:
            with st.spinner("正在辨識牆壁、地板與材質…"):
                result = analyze(rgb, confidence, area_threshold, extend_regions)
            st.session_state["surface_result"] = (result_key, result)
        except Exception as exc:
            st.error(f"辨識未完成：{exc}")
            return

    saved = st.session_state.get("surface_result")
    if saved is None:
        return
    result = saved[1]
    st.success("辨識完成")
    original_tab, surface_tab, material_tab = st.tabs(["原始照片", "牆壁／地板", "材質"])
    with original_tab:
        st.image(result["original"], caption="原始照片", use_column_width=True)
    with surface_tab:
        st.image(result["surface"], caption="橘色：牆壁　藍綠色：地板", use_column_width=True)
        left, right = st.columns(2)
        left.metric("牆壁占整張照片", f"{result['wall_ratio']:.2f}%")
        right.metric("地板占整張照片", f"{result['floor_ratio']:.2f}%")
        if not (result["wall_ratio"] or result["floor_ratio"]):
            st.info("未偵測到牆壁或地板，請嘗試更換照片。")
        if extend_regions:
            st.caption("顯示與比例包含上下延伸補齊的推估區域。")
        st.download_button("下載牆壁／地板結果", models.image_to_png_bytes(result["surface"]),
                           file_name="wall_floor_result.png", mime="image/png")
    with material_tab:
        st.image(result["material"], caption="材質分割結果", use_column_width=True)
        st.caption("比例以整張照片的像素數計算；背景包含未分類及過濾掉的小區域。")
        st.table([
            {"材質": name, "占整張照片比例": f"{ratio:.2f}%"}
            for name, ratio in result["ratios"].items()
        ])
        with st.expander("圖中標籤對照"):
            st.write("material_1：紅磚；material_2：石頭；floor_2：短磚；wall_3：水泥；floor_4：瓷磚。")
        st.download_button("下載材質結果", models.image_to_png_bytes(result["material"]),
                           file_name="material_result.png", mime="image/png")


if __name__ == "__main__":
    main()
