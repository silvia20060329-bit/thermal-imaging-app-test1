import os
from align_hybrid_vis2ir import align_hybrid_logic

def main():
    base_dir = os.getcwd()
    # 確保資料夾路徑正確
    vis_dir = os.path.join(base_dir, "data", "visible")
    ir_dir = os.path.join(base_dir, "data", "infrared")
    out_dir = os.path.join(base_dir, "data", "output")

    if not os.path.exists(vis_dir) or not os.path.exists(ir_dir):
        print("錯誤: 找不到 data/visible 或 data/infrared 資料夾")
        return

    vis_files = [f for f in os.listdir(vis_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"開始執行混合對齊處理，共 {len(vis_files)} 張...")

    for fname in vis_files:
        vis_path = os.path.join(vis_dir, fname)
        # 尋找對應的 IR 檔案
        ir_path = os.path.join(ir_dir, fname)
        
        if os.path.exists(ir_path):
            # 現在參數數量一致了 (vis_path, ir_path, out_dir)
            align_hybrid_logic(vis_path, ir_path, out_dir)
        else:
            print(f"跳過: {fname} (找不到對應 IR)")

if __name__ == "__main__":
    main()
