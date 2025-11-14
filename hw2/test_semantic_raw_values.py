#!/usr/bin/env python3
"""
深入檢查 semantic_raw 的實際值和編碼方式
"""
import sys
import os
import numpy as np
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis

sys.path.insert(0, '/home/clu98753cs13/Desktop/course/phyai/physical-ai25/hw2')
from part3_v6 import (
    load_semantic_ID_table,
    HabitatEnvWrapper
)

currdir = '/home/clu98753cs13/Desktop/course/phyai/physical-ai25/hw2'
SCENE_PATH = f"{currdir}/replica_v1/apartment_0/habitat/mesh_semantic.ply"
NAVMESH_PATH = f"{currdir}/replica_v1/apartment_0/habitat/mesh_semantic.navmesh"

print("=== 深入檢查 semantic_raw 的實際值 ===\n")

# 載入 id_map
id_map = load_semantic_ID_table(f"{currdir}/color_coding_semantic_segmentation_classes.xlsx")
target_class = 'rack'
target_id = id_map[target_class.lower()] % 40
target_id_full = id_map[target_class.lower()]

print(f"目標類別: {target_class}")
print(f"完整語義 ID: {target_id_full}")
print(f"語義 ID % 40: {target_id}\n")

# 初始化環境
sim_settings = {
    "scene": SCENE_PATH,
    "default_agent": 0,
    "sensor_height": 1.5,
    "sensor_foot_height": 0.2,
    "width": 512,
    "height": 512,
    "sensor_pitch": 0,
}

try:
    habitat_sim.Simulator.close()
except Exception:
    pass

env = HabitatEnvWrapper(sim_settings)
env.sim.pathfinder.load_nav_mesh(NAVMESH_PATH)

# 獲取多個位置的觀測
print("檢查多個位置的語義觀測...\n")

positions = [
    (0.0, 0.0, 0.0),  # 原點
    (2.0, 0.0, 2.0),  # 移動位置
    (4.0, 0.0, 4.0),  # 更遠位置
]

for i, (x, y, z) in enumerate(positions):
    print(f"位置 {i+1}: ({x}, {y}, {z})")
    
    # 設置 agent 位置
    agent_state = env.agent.get_state()
    agent_state.position = np.array([x, y, z], dtype=np.float32)
    env.agent.set_state(agent_state)
    
    # 獲取觀測
    obs = env.step("move_forward")
    semantic_raw = obs["semantic_raw"]
    
    # 分析 semantic_raw
    unique_values = np.unique(semantic_raw)
    print(f"  唯一值數量: {len(unique_values)}")
    print(f"  唯一值: {unique_values[:10]}...")  # 只顯示前10個
    
    # 檢查各種匹配方式
    mask_mod40 = (semantic_raw % 40 == target_id).astype(np.uint8)
    mask_full = (semantic_raw == target_id_full).astype(np.uint8)
    mask_mod40_alt = (semantic_raw % 40 == target_id_full % 40).astype(np.uint8)
    
    print(f"  semantic_raw % 40 == {target_id}: {mask_mod40.sum()} 像素")
    print(f"  semantic_raw == {target_id_full}: {mask_full.sum()} 像素")
    print(f"  semantic_raw % 40 == {target_id_full % 40}: {mask_mod40_alt.sum()} 像素")
    
    # 檢查是否有接近的值
    close_values = unique_values[(unique_values % 40 == target_id) | (unique_values == target_id_full)]
    if len(close_values) > 0:
        print(f"  找到接近的值: {close_values}")
    
    print()

# 檢查語義場景中的物件
print("檢查語義場景中的 'rack' 物件...\n")
scene = env.sim.semantic_scene
rack_objects = []
if scene and scene.objects:
    for obj in scene.objects:
        if obj and obj.category:
            obj_semantic_id = None
            if hasattr(obj, 'semantic_id'):
                obj_semantic_id = obj.semantic_id
            elif hasattr(obj.category, 'index'):
                obj_semantic_id = obj.category.index()
            
            if obj_semantic_id is not None:
                obj_id_mod40 = obj_semantic_id % 40
                if obj_id_mod40 == target_id:
                    rack_objects.append({
                        'id': obj.id,
                        'semantic_id': obj_semantic_id,
                        'semantic_id_mod40': obj_id_mod40,
                        'aabb': obj.aabb if hasattr(obj, 'aabb') and obj.aabb else None
                    })

print(f"找到 {len(rack_objects)} 個 'rack' 物件:")
for i, obj_info in enumerate(rack_objects[:5]):  # 只顯示前5個
    print(f"  [{i+1}] ID: {obj_info['id']}, semantic_id: {obj_info['semantic_id']}, semantic_id % 40: {obj_info['semantic_id_mod40']}")
    if obj_info['aabb']:
        center = obj_info['aabb'].center
        print(f"      位置: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")

# 嘗試移動到一個 rack 物件附近
if len(rack_objects) > 0 and rack_objects[0]['aabb']:
    print(f"\n移動到第一個 'rack' 物件附近...")
    center = rack_objects[0]['aabb'].center
    # 稍微偏移，避免在物件內部
    pos = np.array([center[0] + 0.5, center[1], center[2]], dtype=np.float32)
    
    agent_state = env.agent.get_state()
    agent_state.position = pos
    env.agent.set_state(agent_state)
    
    obs = env.step("move_forward")
    semantic_raw = obs["semantic_raw"]
    
    mask_mod40 = (semantic_raw % 40 == target_id).astype(np.uint8)
    mask_full = (semantic_raw == target_id_full).astype(np.uint8)
    
    print(f"  在 'rack' 物件附近:")
    print(f"  semantic_raw % 40 == {target_id}: {mask_mod40.sum()} 像素")
    print(f"  semantic_raw == {target_id_full}: {mask_full.sum()} 像素")
    
    unique_values = np.unique(semantic_raw)
    print(f"  唯一值: {unique_values[:10]}...")
    
    # 檢查是否有匹配的值
    matching_values = unique_values[(unique_values % 40 == target_id) | (unique_values == target_id_full)]
    if len(matching_values) > 0:
        print(f"  ✓ 找到匹配的值: {matching_values}")
    else:
        print(f"  ✗ 沒有找到匹配的值")

env.sim.close()
print("\n=== 測試完成 ===")

