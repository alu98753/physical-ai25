import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
from collections import namedtuple

# ==========================================================
# 路徑設定
# ==========================================================
MAP_PATH = "/home/clu98753cs13/Desktop/course/phyai/physical-ai25/hw2/map.png"
EXCEL_PATH = "/home/clu98753cs13/Desktop/course/phyai/physical-ai25/hw2/color_coding_semantic_segmentation_classes.xlsx"

# RRT 參數
STEP_SIZE = 10
MAX_ITER = 5000
GOAL_SAMPLE_RATE = 0.05

# ==========================================================
# 輔助結構與函式
# ==========================================================
Node = namedtuple("Node", ["x", "y", "parent"])

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def steer(from_node, to_point, step_size):
    dx = to_point[0] - from_node.x
    dy = to_point[1] - from_node.y
    dist = math.hypot(dx, dy)
    if dist == 0:
        return None
    ratio = step_size / dist
    new_x = int(from_node.x + dx * ratio)
    new_y = int(from_node.y + dy * ratio)
    return Node(new_x, new_y, from_node)

def is_collision_free(map_img, node1, node2):
    """沿線取樣確認是否穿越障礙 (黑色)"""
    x1, y1 = node1.x, node1.y
    x2, y2 = node2.x, node2.y
    line_points = np.linspace((x1, y1), (x2, y2), num=20)
    for x, y in line_points:
        if (int(y) >= map_img.shape[0] or int(x) >= map_img.shape[1] or
            int(x) < 0 or int(y) < 0):
            return False
        if map_img[int(y), int(x)] == 0:  # 黑色障礙物
            return False
    return True

def nearest(nodes, point):
    nearest_node = nodes[0]
    min_dist = float("inf")
    for node in nodes:
        d = distance((node.x, node.y), point)
        if d < min_dist:
            nearest_node = node
            min_dist = d
    return nearest_node

def extract_path(goal_node):
    path = []
    node = goal_node
    while node is not None:
        path.append((node.x, node.y))
        node = node.parent
    path.reverse()
    return path

# ==========================================================
# 語意表處理
# ==========================================================
def load_semantic_table(excel_path):
    """
    從 Excel 語意表讀取每個類別的 RGB 顏色。
    支援格式：
    - Color_Code (R,G,B)
    - Color
    - Name
    """
    df = pd.read_excel(excel_path)
    
    # 自動找欄位名稱（有時 Excel 欄位有空白）
    color_col = None
    name_col = None
    for c in df.columns:
        if "Color_Code" in c and "(R" in c:  # 找 "(R,G,B)"
            color_col = c
        elif c.strip().lower() in ["name", "class", "object", "color name", "Color"]:
            name_col = c

    if color_col is None or name_col is None:
        raise ValueError(f"❌ 無法在 Excel 中找到顏色或名稱欄位，檢查欄名：{list(df.columns)}")

    color_map = {}

    for _, row in df.iterrows():
        name = str(row[name_col]).strip().lower()
        color_str = str(row[color_col]).strip()

        # 解析字串格式 "(R, G, B)"
        nums = [int(v) for v in color_str.replace("(", "").replace(")", "").split(",") if v.strip().isdigit()]
        if len(nums) != 3:
            continue
        color_map[name] = tuple(nums)

    print(f"[INFO] 成功載入 {len(color_map)} 個語意分類。")
    return color_map


def find_object_region(map_path, color_map, target_class):
    img = cv2.imread(map_path)
    if img is None:
        raise FileNotFoundError(f"❌ 找不到地圖: {map_path}")

    target_class = target_class.lower()
    if target_class not in color_map:
        raise ValueError(f"⚠️ 類別 '{target_class}' 不在語意表中。")

    # Excel 是 RGB，OpenCV 是 BGR
    bgr_color = tuple(reversed(color_map[target_class]))
    mask = cv2.inRange(img, bgr_color, bgr_color)

    coords = cv2.findNonZero(mask)
    if coords is None:
        raise ValueError(f"⚠️ 找不到目標類別 '{target_class}' 的區域。")

    mean = np.mean(coords, axis=0)[0]
    goal = (int(mean[0]), int(mean[1]))

    print(f"[INFO] {target_class} 目標中心座標: {goal}")
    return goal, mask

def select_start(map_path, goal):
    img = cv2.imread(map_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.scatter(goal[0], goal[1], c='red', s=80, label='Goal')
    plt.title("點選起點 (Start)")
    plt.legend()
    pts = plt.ginput(1, timeout=0)
    plt.close()
    start = tuple(map(int, pts[0]))
    return start

# ==========================================================
# RRT 主演算法
# ==========================================================
def rrt_planning(map_img, start, goal):
    start_node = Node(start[0], start[1], None)
    goal_node = Node(goal[0], goal[1], None)
    nodes = [start_node]

    for i in range(MAX_ITER):
        # 隨機取樣 + goal bias
        if random.random() < GOAL_SAMPLE_RATE:
            x_rand = goal
        else:
            h, w = map_img.shape
            x_rand = (random.randint(0, w - 1), random.randint(0, h - 1))

        nearest_node = nearest(nodes, x_rand)
        new_node = steer(nearest_node, x_rand, STEP_SIZE)

        if new_node is None:
            continue
        if not is_collision_free(map_img, nearest_node, new_node):
            continue

        nodes.append(new_node)

        # 是否接近目標
        if distance((new_node.x, new_node.y), goal) < STEP_SIZE:
            # print(f"[SUCCESS] 第 {i} 次迭代到達目標。")
            final_node = Node(goal[0], goal[1], new_node)
            path = extract_path(final_node)
            return path, nodes

    print("❌ 未找到可行路徑。")
    return None, nodes

# ==========================================================
# 視覺化
# ==========================================================
def visualize_rrt(map_img, nodes, start, goal, path):
    plt.figure(figsize=(8, 8))
    plt.imshow(map_img, cmap='gray')
    plt.plot(start[0], start[1], "go", markersize=8, label="Start")
    plt.plot(goal[0], goal[1], "ro", markersize=8, label="Goal")

    for node in nodes:
        if node.parent is not None:
            plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "b-", linewidth=0.4)

    if path:
        px, py = zip(*path)
        plt.plot(px, py, "r-", linewidth=2.0, label="Path")

    plt.legend()
    plt.axis("equal")
    plt.show()

# ==========================================================
# 主流程
# ==========================================================
if __name__ == "__main__":
    print("=== HW2 Part2: Semantic-guided RRT Path Planning ===")
    # 載入地圖
    map_img_gray = cv2.imread(MAP_PATH, cv2.IMREAD_GRAYSCALE)
    binary_map = np.where(map_img_gray > 100, 255, 0).astype(np.uint8)

    _, binary_map = cv2.threshold(map_img_gray, 240, 255, cv2.THRESH_BINARY)

    # ==========================================================
    # Step 1. 載入語意表與地圖
    # ==========================================================
    color_map = load_semantic_table(EXCEL_PATH)
    map_img = cv2.imread(MAP_PATH)
    if map_img is None:
        raise FileNotFoundError(f"❌ 找不到地圖: {MAP_PATH}")

    # ==========================================================
    # Step 2. 掃描地圖中實際出現的顏色
    # ==========================================================
    # 將 map.png 所有像素壓平成一組唯一 RGB 集合
    unique_colors = np.unique(map_img.reshape(-1, 3), axis=0)
    unique_colors_set = {tuple(color.tolist()) for color in unique_colors}

    # 找出這些顏色在語意表中對應的物件
    available_classes = []
    for name, rgb in color_map.items():
        bgr = tuple(reversed(rgb))  # Excel 是 RGB, OpenCV 是 BGR
        if bgr in unique_colors_set:
            available_classes.append(name)

    if not available_classes:
        raise RuntimeError("❌ 無法在 map.png 找到任何語意類別，請確認顏色與語意表一致。")

    print(f"[INFO] 此地圖中實際可用的目標類別共有 {len(available_classes)} 種：")
    print(available_classes)

    # ==========================================================
    # Step 3. 讓使用者輸入目標物件（僅限存在於地圖的）
    # ==========================================================
    target_class = input(f"請輸入目標類別 {available_classes}: ").strip().lower()
    if target_class not in available_classes:
        raise ValueError(f"⚠️ '{target_class}' 不在地圖可用清單中。")

    # 找出目標區域中心
    goal, mask = find_object_region(MAP_PATH, color_map, target_class)

    # 讓使用者選擇起點
    start = select_start(MAP_PATH, goal)

    # 執行 RRT
    path, nodes = rrt_planning(binary_map, start, goal)

    # 顯示結果
    if path:
        visualize_rrt(binary_map, nodes, start, goal, path)
        print(f"[INFO] 路徑長度: {len(path)}")
        result_img = cv2.cvtColor(binary_map, cv2.COLOR_GRAY2BGR)
        for i in range(len(path)-1):
            cv2.line(result_img, path[i], path[i+1], (0, 0, 255), 2)
        cv2.imwrite(f"rrt_result_{target_class}.png", result_img)
        print(f"[INFO] 結果已儲存至 rrt_result_{target_class}.png")
