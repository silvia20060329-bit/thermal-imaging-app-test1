import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO  # 引入 YOLO 套件

# 設定頁面佈局為寬版
st.set_page_config(layout="wide", page_title="熱影像與可見光整合分析系統")

st.title("熱影像與可見光整合分析系統")
st.markdown("上傳可見光(RGB)與熱影像(Thermal)照片，進行自動化材質分類與異常溫度檢測。")

# ==========================================
# 1. 真正載入模型 (使用 快取 確保只載入一次)
# ==========================================
@st.cache_resource
def load_models():
    """
    載入位於 weights 資料夾中的模型權重
    """
    try:
        # 真正載入你的牆壁地板分割模型
        seg_model = YOLO("weights/floor_wall_seg.pt")
        
        models = {
            "seg": seg_model,
            # 未來如果有材質模型或熱影像模型，可以依此類推加在這裡：
            # "material": YOLO("weights/material.pt"),
        }
        return models
    except Exception as e:
        st.error(f"模型載入失敗，請檢查 weights/floor_wall_seg.pt 是否存在。錯誤訊息: {e}")
        return None

# 執行載入
with st.spinner("🧠 系統正在載入 AI 模型權重，請稍候..."):
    my_models = load_models()


# ==========================================
# 2. 後端核心運算 Pipeline
# ==========================================
def process_pipeline(rgb_img, thermal_img, models):
    """
    影像處理與模型推論流程
    """
    # 將 PIL Image 轉換為 OpenCV 格式 (BGR)，方便模型與 CV2 處理
    rgb_cv = cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)
    thermal_cv = cv2.cvtColor(np.array(thermal_img), cv2.COLOR_RGB2BGR)
    
    # ---- 階段 1: 影像對齊 (暫時維持模擬，未來可加入對齊演算法) ----
    aligned_rgb = rgb_cv.copy()
    aligned_thermal = thermal_cv.copy()
    
    # ---- 階段 2: 牆壁/地板區域切割 (使用真實模型) ----
    seg_visual = aligned_rgb.copy()
    
    if models and "seg" in models:
        # 執行 YOLO 分割模型推論
        # conf=0.25 表示信心度門檻，可自由調整
        seg_results = models["seg"](aligned_rgb, conf=0.25)[0]
        
        # 將模型的預測結果（框、遮罩）繪製到圖片上
        seg_visual = seg_results.plot() 
        
        # 這裡可以提取 mask 資訊以供後續分析（範例：計算面積或特定區域限制）
        # if seg_results.masks is not None:
        #     masks = seg_results.masks.data
    
    # ---- 階段 3 & 4: 其他分支與資訊融合 (暫時維持模擬) ----
    # 模擬材質與異常數據
    wood_ratio = 85.0
    anomaly_ratio = 12.5
    
    # 融合總覽：暫時將分割結果作為總覽展示
    fusion_visual = seg_visual.copy()
    
    # 將 OpenCV 的 BGR 格式轉回 RGB 供 Streamlit 顯示
    fusion_visual = cv2.cvtColor(fusion_visual, cv2.COLOR_BGR2RGB)
    seg_visual = cv2.cvtColor(seg_visual, cv2.COLOR_BGR2RGB)
    thermal_out = cv2.cvtColor(aligned_thermal, cv2.COLOR_BGR2RGB)
    
    # ---- 階段 5: 狀態判定 ----
    if anomaly_ratio > 10.0:
        status_text = "⚠️ 疑似含水"
        description = f"在切割出的目標區域內偵測到顯著的低溫異常（佔比 {anomaly_ratio}%），高度吻合水氣滲漏特徵。"
    else:
        status_text = "✅ 正常"
        description = "各區域溫度與材質分佈均勻，未偵測到明顯之異常。"

    return {
        "fusion_img": fusion_visual,
        "material_img": seg_visual,  # 這裡先帶入真實的分割結果
        "anomaly_img": thermal_out,
        "wood_ratio": wood_ratio,
        "anomaly_ratio": anomaly_ratio,
        "status": status_text,
        "desc": description
    }


# ==========================================
# 3. 前端介面佈局
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
            st.write("🔄 正在讀取影像並調用牆壁地板分割模型...")
            results = process_pipeline(img_rgb, img_thermal, my_models)
            status_msg.update(label="分析處理完成！", state="complete", expanded=False)

        # 呈現結果
        st.markdown("### 📊 分析結果輸出")
        res_col1, res_col2 = st.columns([2, 1])
        
        with res_col1:
            tab1, tab2, tab3 = st.tabs(["疊合總覽結果", "圖層一：牆壁地板分割結果", "圖層二：熱影像特徵"])
            
            with tab1:
                st.image(results["fusion_img"], caption="融合分析總覽圖", use_container_width=True)
            with tab2:
                st.image(results["material_img"], caption="真實模型推論：牆壁/地板語意分割區域", use_container_width=True)
            with tab3:
                st.image(results["anomaly_img"], caption="熱影像原始分佈", use_container_width=True)
                
        with res_col2:
            st.subheader("📝 結構狀態量化報告")
            st.metric(label="當前評估狀態", value=results["status"])
            
            st.markdown(f"""
            **數據統計：**
            * **目標區域比例**：{results["wood_ratio"]}%
            * **溫度異常面積佔比**：{results["anomaly_ratio"]}%
            
            **詳細判定說明：**
            {results["desc"]}
            """)
    else:
        st.error("請確認兩張影像皆已成功上傳！")
