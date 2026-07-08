import os
import glob
import tempfile
from io import BytesIO

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# 匯入同層資料夾的低溫偵測模組
# 你的資料夾需要有：test1app.py 與 cold_detection.py 在同一層
try:
    from cold_detection import learn_cold_model, detect_cold_regions
    COLD_MODULE_READY = True
except Exception as e:
    COLD_MODULE_READY = False
    COLD_IMPORT_ERROR = e


# ==========================================
# 0. Streamlit 頁面設定
# ==========================================
st.set_page_config(layout="wide", page_title="熱影像與可見光整合分析系統")

st.title("熱影像與可見光整合分析系統")
st.markdown("上傳可見光影像與熱影像，系統會自動進行牆壁/地板分割、高溫異常偵測與低溫異常偵測。")

st.info("系統流程：影像輸入 → 影像對齊 → 牆壁/地板分割 → 高溫異常偵測 → 低溫異常偵測 → 多圖層結果輸出")


# ==========================================
# 1. 側邊欄參數設定
# ==========================================
with st.sidebar:
    st.header("⚙️ 偵測參數")

    seg_conf = st.slider("牆壁/地板分割信心度", 0.05, 0.95, 0.25, 0.05)

    st.subheader("高溫偵測")
    high_threshold = st.slider("高溫亮度門檻", 0, 255, 200, 1)
    high_min_area = st.slider("高溫最小面積", 0, 5000, 50, 50)

    st.subheader("低溫偵測")
    cold_quantile = st.slider("低溫候選比例", 0.05, 0.50, 0.20, 0.05)
    cold_min_area = st.slider("低溫最小面積", 0, 5000, 1000, 50)
    roi_mode = st.selectbox(
        "低溫偵測範圍",
        options=[None, "bottom_half", "bottom_100"],
        format_func=lambda x: "整張影像" if x is None else x,
    )

    # st.caption("如果同層資料夾有 cold_train/，系統會用裡面的熱影像學習低溫模型；若沒有，會直接用本次上傳的熱影像自動估計。")


# ==========================================
# 2. 工具函式
# ==========================================
def cv2_to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def image_to_png_bytes(img_rgb):
    """將 RGB numpy image 轉成 PNG bytes，供下載按鈕使用。"""
    pil_img = Image.fromarray(img_rgb)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def learn_cold_model_from_current_image(img_bgr, cold_quantile_value):
    """
    因為 cold_detection.learn_cold_model() 原本吃的是圖片路徑，
    所以這裡先把目前上傳的熱影像暫存成 png，再交給原本函式學習低溫模型。
    """
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


# ==========================================
# 3. 載入模型
# ==========================================
@st.cache_resource
def load_models():
    """載入 YOLO 權重。"""
    models = {}

    weight_path = "weights/floor_wall_seg.pt"
    if os.path.exists(weight_path):
        models["seg"] = YOLO(weight_path)
    else:
        st.warning(f"找不到模型權重：{weight_path}，牆壁/地板分割會暫時略過。")

    return models


with st.spinner("🧠 系統正在載入 AI 模型權重，請稍候..."):
    my_models = load_models()

if not COLD_MODULE_READY:
    st.warning(f"cold_detection.py 匯入失敗，低溫偵測會暫時略過。錯誤：{COLD_IMPORT_ERROR}")


# ==========================================
# 4. 後端核心運算 Pipeline
# ==========================================
def process_pipeline(
    rgb_img,
    thermal_img,
    models,
    seg_conf_value=0.25,
    high_threshold_value=200,
    high_min_area_value=50,
    cold_quantile_value=0.2,
    cold_min_area_value=500,
    roi_mode_value=None,
):
    """影像處理與模型推論流程。"""

    # PIL Image -> OpenCV BGR
    rgb_cv = cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)
    thermal_cv = cv2.cvtColor(np.array(thermal_img), cv2.COLOR_RGB2BGR)

    # ---- 階段 1：影像對齊 ----
    # 目前先保留 copy，之後可以把配準演算法接在這裡
    aligned_rgb = rgb_cv.copy()
    aligned_thermal = thermal_cv.copy()

    # ---- 階段 2：牆壁/地板分割 ----
    seg_visual = aligned_rgb.copy()
    target_ratio = 0.0

    if models and "seg" in models:
        seg_results = models["seg"](aligned_rgb, conf=seg_conf_value)[0]
        seg_visual = seg_results.plot()

        # 如果模型有輸出 mask，簡單估算目標區域比例
        if seg_results.masks is not None:
            masks = seg_results.masks.data.cpu().numpy()
            union_mask = np.any(masks > 0.5, axis=0).astype(np.uint8)
            target_ratio = round((np.count_nonzero(union_mask) / union_mask.size) * 100, 2)

    # ---- 階段 3：高溫異常偵測 ----
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

    # ---- 階段 4：低溫異常偵測，接 cold_detection.py ----
    cold_visual = aligned_thermal.copy()
    cold_ratio = 0.0
    cold_desc = "低溫偵測模組尚未啟用。"

    if COLD_MODULE_READY:
        try:
            # 優先使用同層 cold_train/ 內的熱影像學習低溫模型
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
                cold_model = learn_cold_model_from_current_image(
                    aligned_thermal,
                    cold_quantile_value,
                )
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

    # ---- 階段 5：多圖層總覽 ----
    # 目前總覽先使用分割圖，之後可以改成把熱異常疊到 RGB 上
    fusion_visual = seg_visual.copy()

    # ---- 階段 6：狀態判定 ----
    if cold_ratio > 10.0:
        status_text = "⚠️ 疑似含水"
    elif high_temp_ratio > 10.0:
        status_text = "⚠️ 高溫異常"
    else:
        status_text = "✅ 正常"

    description = (
        f"高溫異常面積佔比為 {high_temp_ratio}%，低溫異常面積佔比為 {cold_ratio}%。\n\n"
        f"{cold_desc}\n\n"
        "低溫區域可能與含水、陰影或材料熱特性有關；高溫區域可能與日照、反射或局部熱源有關，仍需搭配現場條件判讀。"
    )

    # BGR -> RGB for Streamlit
    fusion_out = cv2_to_rgb(fusion_visual)
    seg_out = cv2_to_rgb(seg_visual)
    hot_out = cv2_to_rgb(hot_visual)
    cold_out = cv2_to_rgb(cold_visual)

    return {
        "fusion_img": fusion_out,
        "material_img": seg_out,
        "hot_img": hot_out,
        "cold_img": cold_out,
        "target_ratio": target_ratio,
        "high_ratio": high_temp_ratio,
        "cold_ratio": cold_ratio,
        "status": status_text,
        "desc": description,
    }


# ==========================================
# 5. 前端介面佈局
# ==========================================
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
            st.write("🧱 正在進行牆壁/地板分割...")
            st.write("🔥 正在進行高溫異常偵測...")
            st.write("❄️ 正在進行低溫異常偵測...")

            results = process_pipeline(
                img_rgb,
                img_thermal,
                my_models,
                seg_conf_value=seg_conf,
                high_threshold_value=high_threshold,
                high_min_area_value=high_min_area,
                cold_quantile_value=cold_quantile,
                cold_min_area_value=cold_min_area,
                roi_mode_value=roi_mode,
            )

            status_msg.update(label="分析處理完成！", state="complete", expanded=False)

        st.markdown("### 📊 分析結果輸出")
        res_col1, res_col2 = st.columns([2, 1])

        with res_col1:
            tab1, tab2, tab3, tab4 = st.tabs([
                "多圖層總覽",
                "圖層一：牆壁/地板分割",
                "圖層二：高溫異常區",
                "圖層三：低溫異常區",
            ])

            with tab1:
                st.image(results["fusion_img"], caption="融合分析總覽圖", use_container_width=True)
            with tab2:
                st.image(results["material_img"], caption="牆壁/地板語意分割結果", use_container_width=True)
            with tab3:
                st.image(results["hot_img"], caption="高溫異常區域圈選結果", use_container_width=True)
            with tab4:
                st.image(results["cold_img"], caption="低溫異常區域圈選結果", use_container_width=True)

        with res_col2:
            st.subheader("📝 結構狀態量化報告")
            st.metric(label="當前評估狀態", value=results["status"])

            st.markdown(f"""
            **數據統計：**
            * **牆壁/地板目標區域比例**：{results["target_ratio"]}%
            * **高溫異常面積佔比**：{results["high_ratio"]}%
            * **低溫異常面積佔比**：{results["cold_ratio"]}%

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
