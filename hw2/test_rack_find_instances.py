#!/usr/bin/env python3
"""
測試：驗證 find_all_object_instances 是否能找到 'rack' 的實例
"""
import sys
sys.path.insert(0, '/home/clu98753cs13/Desktop/course/phyai/physical-ai25/hw2')
from rrt_star import find_all_object_instances, load_semantic_table

# 路徑設定
currdir = '/home/clu98753cs13/Desktop/course/phyai/physical-ai25/hw2'
MAP_PATH = f"{currdir}/map100.png"
EXCEL_PATH = f"{currdir}/color_coding_semantic_segmentation_classes.xlsx"

# 載入語意表
color_map = load_semantic_table(EXCEL_PATH)

# 測試 'rack'
print("=== 測試 find_all_object_instances 是否能找到 'rack' ===\n")
goals_list, goal_mask = find_all_object_instances(MAP_PATH, color_map, 'rack')

if goals_list and goal_mask is not None:
    print(f"✓ 成功！找到 {len(goals_list)} 個 'rack' 實例")
    print(f"  實例中心點: {goals_list[:5]}...")  # 只顯示前5個
    print(f"  Mask 非零像素數: {goal_mask.sum()}")
else:
    print("✗ 失敗！仍然找不到 'rack' 的實例")

