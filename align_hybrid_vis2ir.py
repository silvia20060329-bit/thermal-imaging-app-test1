import cv2
import numpy as np
import os
from utils_mi import compute_mutual_information

def align_hybrid_logic(vis_path, ir_path, out_dir):
    # 讀取影像
    vis = cv2.imread(vis_path)
    ir = cv2.imread(ir_path)
    if vis is None or ir is None: return False

    # 1. 論文基準座標 (VIS 2048x1536 -> IR 640x480) 
    start_x, start_y = 554, 415
    tw, th = 939, 704
    
    vis_h, vis_w = vis.shape[:2]
    ir_gray = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY) if len(ir.shape)==3 else ir
    ir_h, ir_w = ir.shape[:2]

    best_mi = -1.0
    best_coord = (start_x, start_y)

    # 2. 精細搜尋：垂直 +/- 160, 水平 +/- 40
    for dy in range(-160, 161, 16):
        for dx in range(-40, 41, 8):
            nx, ny = start_x + dx, start_y + dy
            if nx < 0 or ny < 0 or nx + tw > vis_w or ny + th > vis_h: continue
                
            patch = vis[ny:ny+th, nx:nx+tw]
            patch_res = cv2.resize(patch, (ir_w, ir_h))
            patch_gray = cv2.cvtColor(patch_res, cv2.COLOR_BGR2GRAY)
            
            mi = compute_mutual_information(patch_gray, ir_gray)
            if mi > best_mi:
                best_mi = mi
                best_coord = (nx, ny)

    # 3. 輸出處理
    final_x, final_y = best_coord
    vis_aligned = cv2.resize(vis[final_y:final_y+th, final_x:final_x+tw], (ir_w, ir_h))
    
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(vis_path))[0]
    cv2.imwrite(os.path.join(out_dir, f"{stem}_aligned.png"), vis_aligned)
    
    overlay = cv2.addWeighted(vis_aligned, 0.5, ir, 0.5, 0)
    cv2.imwrite(os.path.join(out_dir, f"{stem}_overlay.png"), overlay)
    
    print(f"完成 {stem}: MI = {best_mi:.4f}")
    return True
