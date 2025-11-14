#!/usr/bin/env python3
"""
測試腳本：驗證 'rack' 類別的顏色匹配問題
"""
import cv2
import numpy as np
import os

# 載入語意表函數（從 part3_v6.py 複製）
def load_semantic_table(excel_path):
    import pandas as pd
    df = pd.read_excel(excel_path)
    color_map = {}
    for _, row in df.iterrows():
        name = str(row["Name"]).strip().lower()
        rgb_str = str(row["Color_Code (R,G,B)"]).strip()
        if rgb_str.startswith("(") and rgb_str.endswith(")"):
            rgb_str = rgb_str[1:-1]
        parts = rgb_str.split(",")
        if len(parts) == 3:
            r, g, b = [int(x.strip()) for x in parts]
            color_map[name] = (r, g, b)
    return color_map

# 主測試
if __name__ == "__main__":
    print("=== 測試 'rack' 顏色匹配問題 ===\n")
    
    # 路徑設定
    currdir = os.path.dirname(os.path.abspath(__file__))
    MAP_PATH = os.path.join(currdir, "map100.png")
    EXCEL_PATH = os.path.join(currdir, "color_coding_semantic_segmentation_classes.xlsx")
    
    # 1. 載入語意表
    print("1. 載入語意表...")
    color_map = load_semantic_table(EXCEL_PATH)
    print(f"   ✓ 成功載入 {len(color_map)} 個語意分類")
    
    # 2. 檢查 'rack' 是否在語意表中
    print("\n2. 檢查 'rack' 是否在語意表中...")
    if 'rack' in color_map:
        rack_rgb = color_map['rack']
        print(f"   ✓ 'rack' 在語意表中")
        print(f"   ✓ 'rack' 的 RGB 顏色: {rack_rgb}")
        rack_bgr = tuple(reversed(rack_rgb))
        print(f"   ✓ 'rack' 的 BGR 顏色: {rack_bgr}")
    else:
        print("   ✗ 'rack' 不在語意表中")
        exit(1)
    
    # 3. 載入地圖
    print("\n3. 載入地圖...")
    map_img = cv2.imread(MAP_PATH)
    if map_img is None:
        print(f"   ✗ 無法載入地圖: {MAP_PATH}")
        exit(1)
    print(f"   ✓ 地圖尺寸: {map_img.shape}")
    
    # 4. 找出地圖中所有唯一的顏色
    print("\n4. 分析地圖中的顏色...")
    unique_colors = np.unique(map_img.reshape(-1, 3), axis=0)
    unique_set = {tuple(c.tolist()) for c in unique_colors}
    print(f"   ✓ 地圖中有 {len(unique_set)} 種唯一顏色")
    
    # 5. 檢查精確匹配
    print("\n5. 檢查精確顏色匹配...")
    if rack_bgr in unique_set:
        print(f"   ✓ 'rack' 的 BGR 顏色 {rack_bgr} 在地圖中找到了！")
    else:
        print(f"   ✗ 'rack' 的 BGR 顏色 {rack_bgr} 不在精確匹配中")
        
        # 6. 檢查近似匹配（容差匹配）
        print("\n6. 檢查近似顏色匹配（容差 = 5）...")
        tolerance = 5
        found_approx = False
        for unique_color in unique_set:
            diff = np.abs(np.array(rack_bgr) - np.array(unique_color))
            if np.all(diff <= tolerance):
                print(f"   ✓ 找到近似匹配！")
                print(f"     語意表顏色: {rack_bgr}")
                print(f"     地圖顏色:   {unique_color}")
                print(f"     差異:       {diff}")
                found_approx = True
                break
        
        if not found_approx:
            print(f"   ✗ 在容差 {tolerance} 內沒有找到近似匹配")
            
            # 7. 檢查更寬的容差
            print("\n7. 檢查更寬的容差匹配（容差 = 10）...")
            tolerance = 10
            found_approx = False
            for unique_color in unique_set:
                diff = np.abs(np.array(rack_bgr) - np.array(unique_color))
                if np.all(diff <= tolerance):
                    print(f"   ✓ 找到近似匹配！")
                    print(f"     語意表顏色: {rack_bgr}")
                    print(f"     地圖顏色:   {unique_color}")
                    print(f"     差異:       {diff}")
                    found_approx = True
                    break
            
            if not found_approx:
                print(f"   ✗ 在容差 {tolerance} 內沒有找到近似匹配")
                
                # 8. 找出最接近的顏色
                print("\n8. 找出最接近的顏色...")
                min_diff = float('inf')
                closest_color = None
                for unique_color in unique_set:
                    diff = np.sum(np.abs(np.array(rack_bgr) - np.array(unique_color)))
                    if diff < min_diff:
                        min_diff = diff
                        closest_color = unique_color
                
                print(f"   最接近的顏色: {closest_color}")
                print(f"   差異總和:     {min_diff}")
                print(f"   各通道差異:   {np.abs(np.array(rack_bgr) - np.array(closest_color))}")
    
    # 9. 檢查地圖中是否有 'rack' 的像素
    print("\n9. 檢查地圖中是否有 'rack' 的像素...")
    rack_mask = np.all(map_img == np.array(rack_bgr), axis=2)
    rack_pixel_count = np.sum(rack_mask)
    print(f"   精確匹配的像素數: {rack_pixel_count}")
    
    if rack_pixel_count == 0:
        print("   ✗ 地圖中沒有精確匹配 'rack' 顏色的像素")
        print("   → 這可能是因為：")
        print("     1. 地圖壓縮導致顏色改變")
        print("     2. 地圖格式轉換導致顏色偏差")
        print("     3. 'rack' 物件實際上不在這個地圖中")
    else:
        print(f"   ✓ 地圖中有 {rack_pixel_count} 個 'rack' 像素")
    
    # 10. 檢查其他類別的匹配情況
    print("\n10. 檢查其他類別的匹配情況...")
    matched_count = 0
    unmatched_count = 0
    unmatched_classes = []
    
    for name, rgb in color_map.items():
        bgr = tuple(reversed(rgb))
        if bgr in unique_set:
            matched_count += 1
        else:
            unmatched_count += 1
            unmatched_classes.append(name)
    
    print(f"   匹配的類別數: {matched_count}")
    print(f"   未匹配的類別數: {unmatched_count}")
    if unmatched_count > 0 and unmatched_count <= 20:
        print(f"   未匹配的類別: {unmatched_classes}")
    elif unmatched_count > 20:
        print(f"   未匹配的類別（前20個）: {unmatched_classes[:20]}...")
    
    print("\n=== 測試完成 ===")

