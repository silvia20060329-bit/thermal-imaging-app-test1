import streamlit as st
import time

# 設定頁面佈局為寬版
st.set_page_config(layout="wide", page_title="熱影像與可見光整合分析系統")

st.title("熱影像與可見光整合分析系統")
st.markdown("上傳可見光(RGB)與熱影像(Thermal)照片，進行自動化材質分類與異常溫度檢測。")

# ==========================================
# 1. 介面：圖片上傳區塊
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

# ==========================================
# 2. 介面：執行按鈕與流程展示
# ==========================================
st.divider()

# 只有當兩張圖都上傳時，按鈕才會有作用
if st.button("🚀 開始分析", type="primary", use_container_width=True):
    if rgb_file and thermal_file:
        
        # 模擬後端自動執行的完整流程 (未來這裡替換成你的模型推論程式碼)
        with st.status("後端模型處理中...", expanded=True) as status:
            st.write("⏳ 1. 執行影像對齊 (Image Registration)...")
            time.sleep(1) # 模擬運算時間
            st.write("⏳ 2. 執行牆壁/地板分類切割 (Semantic Segmentation)...")
            time.sleep(1)
            st.write("⏳ 3. 進行材質分類與高低溫異常判斷...")
            time.sleep(1.5)
            st.write("⏳ 4. 融合多圖層結果與計算量化數據...")
            time.sleep(1)
            status.update(label="分析完成！", state="complete", expanded=False)

        # ==========================================
        # 3. 介面：結果展示與文字輸出
        # ==========================================
        st.markdown("### 📊 分析結果輸出")
        
        # 建立兩個欄位，左邊放圖片結果，右邊放文字報告
        res_col1, res_col2 = st.columns([2, 1])
        
        with res_col1:
            # 使用 Tabs 來切換多圖層結果
            tab1, tab2, tab3 = st.tabs(["疊合總覽", "圖層一：材質分類", "圖層二：高低溫異常"])
            
            with tab1:
                # 未來這裡放整合後的圖片 st.image(fusion_result_img)
                st.info("🖼️ 這裡將顯示影像對齊並融合所有資訊的最終疊合結果圖。")
            with tab2:
                # 未來這裡放材質切割圖片 st.image(material_img)
                st.info("🟫 這裡顯示模型切割出的材質分佈圖（例如：木質地板區域標示）。")
            with tab3:
                # 未來這裡放溫度異常圖片 st.image(thermal_anomaly_img)
                st.info("❄️🔥 這裡顯示標記出異常高低溫圈選的熱影像遮罩。")
                
        with res_col2:
            st.subheader("📝 狀態量化報告")
            st.markdown("""
            **區域分析 (地板)**
            * **材質比例**：木質地板 85%, 磚牆 15%
            * **溫度異常佔比**：低溫異常 12% 
            
            **系統綜合判定**
            * **當前狀態**：⚠️ **疑似含水**
            * **說明**：在木質地板區域偵測到顯著的低溫異常集中，高度吻合水氣滲漏或積水特徵，建議進一步進行物理檢測。
            """)
            
            # 提供下載報告或圖片的按鈕
            st.download_button(
                label="📥 下載完整分析報告",
                data="這是一份模擬的報告內容",
                file_name="analysis_report.txt",
                mime="text/plain"
            )
            
    else:
        st.error("請先在上方上傳「可見光」與「熱影像」兩張照片，才能開始分析喔！")