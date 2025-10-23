import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

# ==========================================================
# 路徑設定
# 確保這個路徑與您 RRT 腳本中的 MAP_PATH 一致
# ==========================================================
# 假設您的 map.png 位於 RRT 腳本所在的目錄
# 或者使用您的絕對路徑
MAP_PATH = "/home/clu98753cs13/Desktop/course/phyai/physical-ai25/hw2/map.png"
OUTPUT_FILE = "selected_coords.txt"

def select_points_interactively(map_path, num_points=None):
    """
    從地圖上互動式點選座標。
    
    Args:
        map_path (str): 地圖圖片的路徑。
        num_points (int, optional): 預期點選的點數。
                                    如果為 None，則由使用者按 Enter 結束。
    
    Returns:
        list: 點選的 (x, y) 座標列表。
    """
    # 1. 載入圖片 (使用 OpenCV 載入 BGR 格式)
    img = cv2.imread(map_path)
    if img is None:
        print(f"❌ 錯誤：找不到地圖文件: {map_path}")
        return []
    
    # 2. 轉換為 Matplotlib 接受的 RGB 格式
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 3. 顯示圖片並啟用互動點擊
    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb)
    
    if num_points:
        plt.title(f"請點選 {num_points} 個點。")
    else:
        plt.title("請在地圖上點選多個點，點擊完成後按 Enter 鍵結束。")
    
    # 使用 ginput 進行互動點擊
    # n=num_points: 點擊的點數限制
    # timeout=0: 永不超時 (直到按 Enter 或達到點數)
    try:
        points = plt.ginput(n=num_points if num_points else -1, timeout=0)
    except Exception as e:
        print(f"互動點擊過程發生錯誤: {e}")
        points = []

    plt.close() # 關閉繪圖視窗

    # 4. 整理座標
    coords = []
    for x, y in points:
        # 將浮點數座標轉換為整數 (像素座標)
        coords.append((int(round(x)), int(round(y))))
        
    return coords

def save_coordinates(coords, output_file):
    """將座標儲存到檔案中。"""
    with open(output_file, 'w') as f:
        f.write("# 選定的座標列表 (格式: X Y)\n")
        for x, y in coords:
            f.write(f"{x} {y}\n")
    print(f"\n[INFO] 成功儲存 {len(coords)} 個座標到: {output_file}")
    for i, (x, y) in enumerate(coords):
        print(f"點 {i+1}: X={x}, Y={y}")


if __name__ == "__main__":
    # img = cv2.imread(MAP_PATH)
    # print(img.shape)

    
    
    print("=== 互動式地圖座標選擇工具 ===")
    
    # 讓使用者選擇是否限制點數
    try:
        num_input = input("您想要點選多少個點？ (輸入數字，或留空點選任意個數後按 Enter 結束): ").strip()
        num_points_to_select = int(num_input) if num_input.isdigit() and int(num_input) > 0 else None
    except ValueError:
        num_points_to_select = None

    selected_coords = select_points_interactively(MAP_PATH, num_points_to_select)
    
    if selected_coords:
        save_coordinates(selected_coords, OUTPUT_FILE)
    else:
        print("沒有選定任何座標。")