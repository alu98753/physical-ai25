#!/usr/bin/env python3
"""
快速測試：驗證容差匹配是否解決 'rack' 問題
"""
import cv2
import numpy as np
import sys
sys.path.insert(0, '/home/clu98753cs13/Desktop/course/phyai/physical-ai25/hw2')
from part3_v6 import load_semantic_table

# 路徑設定
currdir = '/home/clu98753cs13/Desktop/course/phyai/physical-ai25/hw2'
MAP_PATH = f"{currdir}/map100.png"
EXCEL_PATH = f"{currdir}/color_coding_semantic_segmentation_classes.xlsx"

# 載入語意表
color_map = load_semantic_table(EXCEL_PATH)

# 載入地圖
map_img = cv2.imread(MAP_PATH)
unique_colors = np.unique(map_img.reshape(-1, 3), axis=0)
unique_set = {tuple(c.tolist()) for c in unique_colors}

# 使用容差匹配
COLOR_TOLERANCE = 5
available_classes = []
for name, rgb in color_map.items():
    bgr = tuple(reversed(rgb))
    bgr_array = np.array(bgr)
    
    # 先嘗試精確匹配
    if bgr in unique_set:
        available_classes.append(name)
    else:
        # 如果精確匹配失敗，嘗試容差匹配
        found = False
        for unique_color in unique_set:
            unique_array = np.array(unique_color)
            diff = np.abs(bgr_array - unique_array)
            if np.all(diff <= COLOR_TOLERANCE):
                available_classes.append(name)
                found = True
                break

print(f"可用類別總數: {len(available_classes)}")
print(f"'rack' 是否在可用類別中: {'rack' in available_classes}")
if 'rack' in available_classes:
    print("✓ 成功！'rack' 現在可以被找到了！")
else:
    print("✗ 失敗！'rack' 仍然找不到")

