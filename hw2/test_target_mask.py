#!/usr/bin/env python3
"""
測試腳本：檢查 target_mask 無法正確匹配的三個可能原因
1. id_map 中沒有該類別
2. 語義 ID 映射不一致
3. target_class 變數作用域問題
"""
import sys
import os
import numpy as np
import cv2
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis

# 添加路徑
sys.path.insert(0, '/home/clu98753cs13/Desktop/course/phyai/physical-ai25/hw2')
from part3_v6 import (
    load_semantic_table, 
    load_semantic_ID_table,
    HabitatEnvWrapper
)

# 路徑設定
currdir = '/home/clu98753cs13/Desktop/course/phyai/physical-ai25/hw2'
EXCEL_PATH = f"{currdir}/color_coding_semantic_segmentation_classes.xlsx"
SCENE_PATH = f"{currdir}/replica_v1/apartment_0/habitat/mesh_semantic.ply"
NAVMESH_PATH = f"{currdir}/replica_v1/apartment_0/habitat/mesh_semantic.navmesh"

print("=== 測試 target_mask 無法正確匹配的原因 ===\n")

# ==========================================================
# 測試 1: 檢查 id_map 中是否有該類別
# ==========================================================
print("1. 檢查 id_map 中是否有 'rack' 類別...")
id_map = load_semantic_ID_table(EXCEL_PATH)
color_map = load_semantic_table(EXCEL_PATH)

test_class = 'rack'
if test_class.lower() in id_map:
    print(f"   ✓ '{test_class}' 在 id_map 中")
    print(f"   ✓ id_map['{test_class}'] = {id_map[test_class.lower()]}")
    print(f"   ✓ id_map['{test_class}'] % 40 = {id_map[test_class.lower()] % 40}")
else:
    print(f"   ✗ '{test_class}' 不在 id_map 中")
    print(f"   ✗ 可用的類別: {list(id_map.keys())[:10]}...")

# ==========================================================
# 測試 2: 檢查語義 ID 映射是否一致
# ==========================================================
print("\n2. 檢查語義 ID 映射是否一致...")

# 檢查 color_map 和 id_map 是否都有相同的類別
if test_class.lower() in color_map and test_class.lower() in id_map:
    print(f"   ✓ '{test_class}' 同時在 color_map 和 id_map 中")
    
    # 檢查語義 ID 是否合理
    semantic_id = id_map[test_class.lower()]
    semantic_id_mod40 = semantic_id % 40
    print(f"   ✓ 語義 ID: {semantic_id}")
    print(f"   ✓ 語義 ID % 40: {semantic_id_mod40}")
    
    if 0 <= semantic_id_mod40 < 40:
        print(f"   ✓ 語義 ID % 40 在有效範圍內 [0, 39]")
    else:
        print(f"   ✗ 語義 ID % 40 不在有效範圍內")
else:
    print(f"   ✗ '{test_class}' 不在 color_map 或 id_map 中")

# ==========================================================
# 測試 3: 檢查 target_class 變數作用域問題
# ==========================================================
print("\n3. 檢查 target_class 變數作用域問題...")

# 模擬 target_mask 函數的實現
def test_target_mask_scope(target_class, id_map, semantic_raw):
    """
    測試 target_mask 函數是否能正確訪問 target_class
    """
    try:
        if target_class.lower() not in id_map:
            return None, f"target_class '{target_class}' not in id_map"
        
        target_id = id_map[target_class.lower()] % 40
        mask = (semantic_raw % 40 == target_id).astype(np.uint8)
        return mask, None
    except NameError as e:
        return None, f"NameError: {e} (可能是作用域問題)"
    except Exception as e:
        return None, f"其他錯誤: {e}"

# 創建一個模擬的 semantic_raw（使用測試 ID）
test_semantic_id = id_map.get(test_class.lower(), 0) % 40
test_semantic_raw = np.full((512, 512), test_semantic_id, dtype=np.uint32)

mask, error = test_target_mask_scope(test_class, id_map, test_semantic_raw)
if error:
    print(f"   ✗ 作用域問題: {error}")
else:
    print(f"   ✓ 沒有作用域問題")
    print(f"   ✓ 生成的 mask 有 {mask.sum()} 個非零像素")

# ==========================================================
# 測試 4: 實際測試 Habitat 環境中的語義 ID
# ==========================================================
print("\n4. 實際測試 Habitat 環境中的語義 ID...")

try:
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
    loaded = env.sim.pathfinder.load_nav_mesh(NAVMESH_PATH)
    print(f"   ✓ NavMesh 載入: {loaded}")
    
    # 獲取一次觀測
    obs = env.step("move_forward")
    semantic_raw = obs["semantic_raw"]
    
    print(f"   ✓ 成功獲取語義觀測")
    print(f"   ✓ semantic_raw 形狀: {semantic_raw.shape}")
    print(f"   ✓ semantic_raw 數據類型: {semantic_raw.dtype}")
    print(f"   ✓ semantic_raw 唯一值數量: {len(np.unique(semantic_raw))}")
    print(f"   ✓ semantic_raw 唯一值範圍: [{np.min(semantic_raw)}, {np.max(semantic_raw)}]")
    
    # 檢查目標 ID 是否出現在語義觀測中
    target_id = id_map[test_class.lower()] % 40
    target_id_full = id_map[test_class.lower()]
    
    # 檢查 % 40 後的匹配
    mask_mod40 = (semantic_raw % 40 == target_id).astype(np.uint8)
    print(f"\n   使用 semantic_raw % 40 == {target_id}:")
    print(f"   - 匹配的像素數: {mask_mod40.sum()}")
    
    # 檢查完整 ID 匹配
    mask_full = (semantic_raw == target_id_full).astype(np.uint8)
    print(f"\n   使用 semantic_raw == {target_id_full} (完整 ID):")
    print(f"   - 匹配的像素數: {mask_full.sum()}")
    
    # 檢查語義場景中的物件
    scene = env.sim.semantic_scene
    if scene and scene.objects:
        print(f"\n   語義場景中的物件:")
        rack_objects = []
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
                            'semantic_id_mod40': obj_id_mod40
                        })
        
        print(f"   - 找到 {len(rack_objects)} 個可能的 'rack' 物件（語義 ID % 40 == {target_id}）")
        if len(rack_objects) > 0:
            print(f"   - 前 3 個物件:")
            for i, obj_info in enumerate(rack_objects[:3]):
                print(f"     [{i+1}] ID: {obj_info['id']}, semantic_id: {obj_info['semantic_id']}, semantic_id % 40: {obj_info['semantic_id_mod40']}")
        else:
            print(f"   ⚠ 沒有找到語義 ID % 40 == {target_id} 的物件")
            print(f"   ⚠ 這可能表示場景中沒有 'rack' 物件，或者語義 ID 映射不正確")
    
    env.sim.close()
    
except Exception as e:
    print(f"   ✗ 環境初始化失敗: {e}")
    import traceback
    traceback.print_exc()

# ==========================================================
# 測試 5: 檢查 target_mask 函數的實際行為
# ==========================================================
print("\n5. 檢查 target_mask 函數的實際行為...")

# 模擬 target_mask 函數（需要全局變數 target_class）
# 注意：這需要 target_class 在全局作用域中
print("   注意：target_mask 函數需要全局變數 'target_class'")
print("   在 part3_v6.py 中，target_class 是在主程式中定義的")
print("   如果 target_mask 函數無法訪問 target_class，會導致 NameError")

# 檢查 part3_v6.py 中 target_mask 的定義
print("\n   建議檢查:")
print("   1. target_class 是否在主程式中正確定義")
print("   2. target_mask 函數是否能訪問全局的 target_class")
print("   3. 如果使用閉包，確保 target_class 被正確捕獲")

print("\n=== 測試完成 ===")

