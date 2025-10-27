import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
from collections import deque
import os
# import scipy
# from scipy.spatial import KDTree
from rtree import index
# ==========================================================
# 路徑設定
# ==========================================================
currdir = os.path.dirname(os.path.abspath(__file__))
DPI = 300
MAP_PATH = os.path.join(currdir, f"map{DPI}.png")
EXCEL_PATH = os.path.join(
    currdir, "color_coding_semantic_segmentation_classes.xlsx")

# ==========================================================
# RRT* 參數（像素座標系）
# ==========================================================
STEP_SIZE = 5 * DPI//100     # 每次延伸步長（px）
MAX_ITER = 20000     # 迭代上限 TODO ：可以根據距離增加
GOAL_SAMPLE_RATE = 0.25     # 目標偏置機率
NEIGHBOR_COEFF = 60.0     # 鄰居半徑係數 (r = coeff * sqrt(log(n)/n))
SMOOTH_ITER = 5      # 路徑平滑化嘗試次數
INFORMED_SAMPLING = True     # 找到初始路徑後啟用 Informed RRT*
GOAL_REACH_THRESH = 3*STEP_SIZE  # 新節點到目標多少距離內視為可接通
COLLISION_SAMPLES_PER_STEP = 4  # 線段碰撞取樣密度（距離/STEP_SIZE*此係數）

# ==========================================================
# 結構
# ==========================================================


class Node:
    __slots__ = ("x", "y", "parent", "cost")

    def __init__(self, x: float, y: float, parent=None, cost: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.parent = parent
        self.cost = float(cost)

    @property
    def pt(self):
        return (self.x, self.y)


# ==========================================================
# 幾何輔助
# ==========================================================
def distance(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])


def is_inside(map_img, x, y):
    return 0 <= x < map_img.shape[1] and 0 <= y < map_img.shape[0]


def is_free_pixel(map_img, x, y, radius=1.0):
    """
    支援浮點座標的安全性檢查（修正為悲觀邏輯）：
    - 在 (x, y) 周圍取 9 個鄰點 (含中心, 3x3)
    - 必須「所有」鄰點都是 free (像素值 >=128) 才視為 free
    - 只要「有任何一個」鄰點是 obstacle (<128) 或出界，就視為 obstacle
    - radius 控制取樣的距離 (像素)：
    """
    H, W = map_img.shape[:2]
    cx, cy = int(round(x)), int(round(y))

    # 9 點 pattern (3x3 鄰域)
    offsets = [
        (0, 0),  # 中心
        (-1, 0), (1, 0), (0, -1), (0, 1),  # 十字
        (-1, -1), (-1, 1), (1, -1), (1, 1),  # 斜角
    ]

    for dx, dy in offsets:
        xx = int(round(cx + dx * radius))
        yy = int(round(cy + dy * radius))

        # 檢查邊界：任何一個採樣點出界，都視為 "不安全"
        if not (0 <= xx < W and 0 <= yy < H):
            return False

        # 檢查障礙：任何一個採樣點碰到障礙 ( < 128)，都視為 "不安全"
        if map_img[yy, xx] < 128:
            return False

    # 迴圈跑完，代表所有 9 個點都在界內且都是 free
    return True


def line_collision_free(map_img, p1, p2):
    """支援浮點採樣的線段碰撞"""
    dist = distance(p1, p2)
    n = int(COLLISION_SAMPLES_PER_STEP * dist / max(1.0, STEP_SIZE)) + 2
    xs = np.linspace(p1[0], p2[0], n)
    ys = np.linspace(p1[1], p2[1], n)
    for x, y in zip(xs, ys):
        # 這裡不傳遞 radius，使用 is_free_pixel 預設的安全檢查
        if not is_free_pixel(map_img, x, y):
            return False
    return True


def steer(from_node, to_point, step_size=STEP_SIZE):
    """從 from_node 朝 to_point 延伸一步（或到 to_point），傳回新 Node"""
    dx = to_point[0] - from_node.x
    dy = to_point[1] - from_node.y
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return None
    if dist <= step_size:
        nx, ny = to_point[0], to_point[1]
    else:
        nx = from_node.x + step_size * dx / dist
        ny = from_node.y + step_size * dy / dist

    return Node(nx, ny)


def nearest(nodes, point):
    best, best_d = None, float("inf")
    px, py = point
    for n in nodes:
        d = (n.x - px)**2 + (n.y - py)**2  # 用平方距離省 sqrt
        if d < best_d:
            best_d = d
            best = n
    return best


def near(nodes, new_node, radius):
    r2 = radius*radius
    out = []
    for n in nodes:
        dx = n.x - new_node.x
        dy = n.y - new_node.y
        if dx*dx + dy*dy <= r2:
            out.append(n)
    return out


def extract_path(goal_node):
    path = []
    n = goal_node
    while n is not None:
        path.append((n.x, n.y))
        n = n.parent
    path.reverse()
    return path


def path_length_px(path):
    if not path or len(path) < 2:
        return 0.0
    return sum(distance(path[i], path[i+1]) for i in range(len(path)-1))

# ==========================================================
# Informed 取樣（在已知最佳路徑長 c_best 後，於包含 start/goal 的橢圓內取樣）
# ==========================================================
def sample_informed(start, goal, c_best, rng, map_shape, labels, start_label):
    # 若 c_best 無限大，回傳 None 代表退回 uniform
    c_min = distance(start, goal)
    if not np.isfinite(c_best) or c_best <= c_min + 1e-6:
        return None

    # 橢圓參數 ( ... 保持不變 ... )
    a = c_best / 2.0
    b = math.sqrt(max(a*a - (c_min/2.0)**2, 1e-6))
    theta = math.atan2(goal[1]-start[1], goal[0]-start[0])

    # 在單位圓內取樣 ( ... 保持不變 ... )
    r = math.sqrt(rng.random())
    ang = 2*math.pi*rng.random()
    x_e = r * math.cos(ang) * a
    y_e = r * math.sin(ang) * b

    # 旋轉 & 平移 ( ... 保持不變 ... )
    c = math.cos(theta)
    s = math.sin(theta)
    x = x_e * c - y_e * s + (start[0] + goal[0]) / 2.0
    y = x_e * s + y_e * c + (start[1] + goal[1]) / 2.0

    # ✅ 【修改】: 檢查點的有效性
    x_i, y_i = int(round(x)), int(round(y))
    H, W = map_shape

    # 必須同時滿足：
    # 1. 在地圖邊界內
    # 2. 落在我們關心的 'start_label' 連通區域 (隱含了不在障礙物上)
    if 0 <= x_i < W and 0 <= y_i < H:
        if labels[y_i, x_i] == start_label:
            return (x_i, y_i)

    # 否則 (越界、在障礙物上、在其他連通區)，返回 None
    return None
# ==========================================================
# 路徑平滑化（Shortcut smoothing）
# ==========================================================


def smooth_path(map_img, path, iterations=SMOOTH_ITER):
    if not path or len(path) < 3:
        return path
    P = list(path)
    rng = random.Random(0xC0FFEE)
    for _ in range(iterations):
        i, j = sorted(rng.sample(range(len(P)), 2))
        if j - i <= 1:
            continue
        if line_collision_free(map_img, P[i], P[j]):
            P = P[:i+1] + P[j:]
    return P


def get_sector_distances(dist_map, center, heading_rad, fov_deg=60, radius=30):
    """
    取得 agent 前方扇形區域的距離牆分佈。
    參數：
        dist_map : np.ndarray - cv2.distanceTransform 結果
        center : (x, y)
        heading_rad : float - 朝向 (弧度)
        fov_deg : float - 視野角 (例如60度代表 ±30度)
        radius : int - 搜索半徑像素
    回傳：
        np.ndarray - 扇形區域內所有距離值（用於 np.percentile）
    """
    H, W = dist_map.shape
    cx, cy = center
    cos_h, sin_h = math.cos(heading_rad), math.sin(heading_rad)
    half_fov = math.radians(fov_deg / 2)

    values = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            x, y = int(cx + dx), int(cy + dy)
            if 0 <= x < W and 0 <= y < H:
                # (1) 距離中心太遠就略過
                d = math.hypot(dx, dy)
                if d > radius:
                    continue
                # (2) 計算方向角是否在扇形範圍內
                ang = math.atan2(dy, dx)
                rel_ang = (ang - heading_rad + math.pi) % (2*math.pi) - math.pi
                if abs(rel_ang) <= half_fov:
                    values.append(dist_map[y, x])
    return np.array(values)


# ==========================================================
# 工具函數：計算帶有安全懲罰的邊緣成本
# ==========================================================
def calculate_edge_cost(parent_node, child_point, goal_point, dist_map, SAFE_WEIGHT, GOAL_SAFETY_EXEMPT_DIST):
    """計算從 parent_node 延伸到 child_point 的邊緣成本 (包含安全懲罰)"""

    # 1. 幾何距離成本
    geometric_cost = distance(parent_node.pt, child_point)

    safety_penalty = 0.0
    if dist_map is not None:
        # 獲取 child_point (新節點) 的安全距離
        iy = int(np.clip(child_point[1], 0, dist_map.shape[0]-1))
        ix = int(np.clip(child_point[0], 0, dist_map.shape[1]-1))
        d_safe = float(dist_map[iy, ix])

        # 獲取 child_point 到目標的距離
        dist_to_goal = distance(child_point, goal_point)

        # 僅在離目標一定距離外計算懲罰
        if dist_to_goal > GOAL_SAFETY_EXEMPT_DIST:
            # 統一使用倒數平方懲罰：d_safe 越小，懲罰越大
            # SAFE_WEIGHT / (d_safe^2)
            safe_penalty = SAFE_WEIGHT / ((d_safe + 1e-3) ** 2)

    return geometric_cost + safety_penalty


# ============
# Speed up
###########

def get_safety_penalty(point, goal_point, dist_map, SAFE_WEIGHT, GOAL_SAFETY_EXEMPT_DIST):
    """計算指定點的安全懲罰項 (不包含幾何距離)"""
    safety_penalty = 0.0
    if dist_map is not None:
        # 獲取點的安全距離
        iy = int(np.clip(point[1], 0, dist_map.shape[0]-1))
        ix = int(np.clip(point[0], 0, dist_map.shape[1]-1))
        d_safe = float(dist_map[iy, ix])

        # 獲取點到目標的距離
        dist_to_goal = distance(point, goal_point)

        # 僅在離目標一定距離外計算懲罰
        if dist_to_goal > GOAL_SAFETY_EXEMPT_DIST:
            # 統一使用倒數平方懲罰
            safety_penalty = SAFE_WEIGHT / ((d_safe + 1e-3) ** 2)

    return safety_penalty

# ==========================================================
# 🔍 工具函式：找最近可行走點 (保持不變)
# ==========================================================
def is_safe_point(map_img, x, y, r=5):
    """
    該點 (x,y) 及其以 r 為半徑的方形鄰域是否全為 free(>=128)。
    """
    H, W = map_img.shape[:2]
    x, y = int(round(x)), int(round(y))
    for dy in range(-r, r + 1):
        ny = y + dy
        if ny < 0 or ny >= H:
            return False
        for dx in range(-r, r + 1):
            nx = x + dx
            if nx < 0 or nx >= W:
                return False
            if map_img[ny, nx] < 128:
                return False
    return True

def find_all_object_instances(map_path, color_map, target_class):
    """
    (新函數)
    在語意地圖中尋找一個類別的所有獨立實例 (instances)。
    例如，找到地圖上 3 個不同的 'window'。

    返回：
        list[tuple(int, int)]: 一個包含所有實例中心點 (x, y) 座標的清單。
    """
    img = cv2.imread(map_path)
    if img is None:
        raise FileNotFoundError(f"❌ 找不到地圖: {map_path}")
    
    target_class = target_class.lower()
    if target_class not in color_map:
        raise ValueError(f"⚠️ 類別 '{target_class}' 不在語意表中。")
    
    # 1. 取得目標顏色 (BGR)
    bgr_color = tuple(reversed(color_map[target_class]))
    
    # 2. 建立該顏色的二進位遮罩 地圖上所有 'window' 像素都會是 255
    original_mask = cv2.inRange(img, bgr_color, bgr_color)
    
    if cv2.countNonZero(original_mask) == 0:
        print(f"⚠️ 找不到目標類別 '{target_class}' 的任何像素。")
        return [], None

    # ==========================================================
    # 用形態學「閉運算」(Closing) 來合併鄰近的區域
    # ==========================================================
    
    # (1) 定義一個 "核" (Kernel) 決定了要合併多近的物體 如果窗戶還是太多，請嘗試「加大」這個值 (e.g., 25, 25)
    merge_kernel_size = (300, 300) 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, merge_kernel_size)
    
    # (2) 執行閉運算 (Dilation -> Erosion)
    #     Dilation: 膨脹，讓鄰近的白色區域連在一起
    #     Erosion: 侵蝕，把膨脹後多餘的邊界縮回來
    #     iterations=1 代表只做一次
    mask = cv2.morphologyEx(original_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 3. (核心) 現在才對「合併後」的 mask 進行連通元件分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)

    goals_list = []
    
    # 4. 迭代所有找到的標籤 (跳過 0，因為 0 是 background)
    for i in range(1, num_labels):
        # 獲取標籤 i (例如 窗戶A) 的中心點
        center = tuple(map(int, centroids[i]))
        goals_list.append(center)
        
    print(f"[INFO] 找到 {len(goals_list)} 個 '{target_class}' 實例 (已合併鄰近區域)。")
    return goals_list, original_mask


def find_nearest_safe_point_radial(map_img, p, max_radius=30, safe_radius=5, neighbor_check=4):
    # ... (保持不變)
    H, W = map_img.shape[:2]
    gx, gy = int(p[0]), int(p[1])

    for r in range(0, max_radius + 1):
        candidates = []
        candidates += [(gx, gy - r), (gx, gy + r), (gx - r, gy), (gx + r, gy)]
        for dy in range(-r, r + 1):
            candidates.append((gx - r, gy + dy))
            candidates.append((gx + r, gy + dy))
        for dx in range(-r + 1, r):
            candidates.append((gx + dx, gy - r))
            candidates.append((gx + dx, gy + r))

        seen = set()
        uniq = []
        for (x, y) in candidates:
            if (x, y) not in seen:
                seen.add((x, y))
                uniq.append((x, y))

        for (x, y) in uniq:
            if 0 <= x < W and 0 <= y < H and map_img[y, x] >= 128:
                if is_safe_point(map_img, x, y, r=safe_radius):
                    return (x, y)

    print("⚠️ [警告] 找不到附近安全點，回傳原點。")
    return (gx, gy)


def find_safe_goal_along_line(map_img, start, goal, step=1, safe_radius=5):
    # ... (保持不變)
    gx, gy = goal
    sx, sy = start
    H, W = map_img.shape[:2]

    dx, dy = sx - gx, sy - gy
    dist = math.hypot(dx, dy)
    if dist == 0:
        return goal
    dx, dy = dx / dist, dy / dist

    for t in np.arange(0, dist, step):
        x = gx + dx * t
        y = gy + dy * t
        if 0 <= int(x) < W and 0 <= int(y) < H and map_img[int(y), int(x)] >= 128:
            if is_safe_point(map_img, x, y, r=safe_radius):
                return (int(x), int(y))

    print("⚠️ [警告] 找不到沿線安全 goal，回傳原始 goal。")
    return goal

# ==========================================================
# ======= 語意表/互動/視覺化函式 (保持不變) =======
# ==========================================================


def load_semantic_table(excel_path):
    # ... (保持不變)
    df = pd.read_excel(excel_path)
    color_col = None
    name_col = None
    for c in df.columns:
        if "Color_Code" in c and "(R" in c:
            color_col = c
        elif c.strip().lower() in ["name", "class", "object", "color", "color name"]:
            name_col = c
    if color_col is None or name_col is None:
        raise ValueError(f"❌ 無法在 Excel 中找到顏色或名稱欄位，檢查欄名：{list(df.columns)}")
    color_map = {}
    for _, row in df.iterrows():
        id = row[0]
        name = str(row[name_col]).strip().lower()
        color_str = str(row[color_col]).strip()
        nums = [int(v) for v in color_str.replace(
            "(", "").replace(")", "").split(",") if v.strip().isdigit()]
        if len(nums) == 3:
            color_map[name] = tuple(nums)
    print(f"[INFO] 成功載入 {len(color_map)} 個語意分類。")
    return color_map


def load_semantic_ID_table(excel_path):
    # ... (保持不變)
    df = pd.read_excel(excel_path)

    id_col = df.columns[0]
    name_col = None

    for c in df.columns:
        if c.strip().lower() in ["name", "class", "object", "color", "color name"]:
            name_col = c

    if name_col is None:
        raise ValueError(f"❌ 無法在 Excel 中找到名稱欄位，檢查欄名：{list(df.columns)}")

    id_map = {}
    for _, row in df.iterrows():
        obj_id = int(row[id_col]) if not pd.isna(row[id_col]) else None
        name = str(row[name_col]).strip().lower()
        if obj_id is not None and name:
            id_map[name] = obj_id

    print(f"[INFO] 成功載入 {len(id_map)} 個語意分類。")
    return id_map


def find_object_region(map_path, color_map, target_class):
    img = cv2.imread(map_path)
    if img is None:
        raise FileNotFoundError(f"❌ 找不到地圖: {map_path}")
    target_class = target_class.lower()
    if target_class not in color_map:
        raise ValueError(f"⚠️ 類別 '{target_class}' 不在語意表中。")
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
    """
    (修改版)
    讓使用者在看到所有可能的 goal 之後，點選起點。
    """
    img = cv2.imread(map_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.imshow(img_rgb)
    
    x , y = goal
    plt.scatter(x, y, c='red', s=30, marker='*',label="Goals")
    plt.title("點選起點 (Start)")
    plt.legend()
    pts = plt.ginput(1, timeout=0)
    plt.close()
    start = tuple(map(int, pts[0]))
    return start


def visualize_rrt(map_img, nodes, start, goal, path):
    plt.figure(figsize=(8, 8))
    plt.imshow(map_img, cmap='gray')
    plt.plot(start[0], start[1], "go", markersize=8, label="Start")
    plt.plot(goal[0], goal[1], "ro", markersize=8, label="Goal")
    for node in nodes:
        if node.parent is not None:
            plt.plot([node.x, node.parent.x], [
                    node.y, node.parent.y], "b-", linewidth=0.4)
    if path:
        px, py = zip(*path)
        plt.plot(px, py, "r-", linewidth=2.0, label="Path (RRT* + smooth)")
    plt.legend()
    plt.axis("equal")
    plt.show()

def visualize_multiple_goals(map_path, goals_list, target_class):
    """
    (新函數)
    在要求使用者點選起點 *之前*，先單獨顯示所有找到的目標。
    """
    img = cv2.imread(map_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(img_rgb)
    
    for i, (x, y) in enumerate(goals_list):
        # 標記
        plt.scatter(x, y, c='red', s=100, zorder=5, marker='X', 
                    label="Goal" if i == 0 else None) # 只顯示一個圖例
        # 標號
        plt.text(x + 5, y + 5, f"{target_class} #{i+1}", 
                 color='white', 
                 fontsize=10,
                 bbox=dict(facecolor='red', alpha=0.7, pad=0.1))
    
    plt.title(f"系統找到 {len(goals_list)} 個 '{target_class}' 區域")
    plt.legend()
    plt.axis("equal")
    plt.show(block=True) # block=True 確保使用者看到才繼續
    plt.close()

def find_nearest_safe_point_in_region_radial(map_img, labels_map, target_label, p, max_radius=100, safe_radius=5):
    """
    (✅ 新函數：專門用於修正 Goal)
    從點 p (通常是原始 goal) 開始，徑向 (方型) 搜索
    第一個「同時」滿足以下條件的點：
    1. 位於 labels_map 上的 target_label 區域
    2. 滿足 is_safe_point (r=safe_radius) 條件
    """
    H, W = map_img.shape[:2]
    gx, gy = int(p[0]), int(p[1])

    # 0. 先檢查原點 (半徑 0)
    if 0 <= gx < W and 0 <= gy < H:
        if labels_map[gy, gx] == target_label and is_safe_point(map_img, gx, gy, r=safe_radius):
            return (gx, gy)
            
    # 1. 徑向 (方型) 搜索
    for r in range(1, max_radius + 1):
        # (此處使用您原有的方型搜索邏輯)
        candidates = []
        candidates += [(gx, gy - r), (gx, gy + r), (gx - r, gy), (gx + r, gy)]
        for dy in range(-r, r + 1):
            candidates.append((gx - r, gy + dy))
            candidates.append((gx + r, gy + dy))
        for dx in range(-r + 1, r):
            candidates.append((gx + dx, gy - r))
            candidates.append((gx + dx, gy + r))

        seen = set()
        uniq = []
        for (x, y) in candidates:
            if (x, y) not in seen:
                seen.add((x, y))
                uniq.append((x, y))

        # 檢查這一層的所有候選點
        for (x, y) in uniq:
            if 0 <= x < W and 0 <= y < H:
                # 核心修改：
                # 必須先檢查標籤 (labels_map[y, x] == target_label)
                # 然後才檢查安全性 (is_safe_point)
                if labels_map[y, x] == target_label:
                    if is_safe_point(map_img, x, y, r=safe_radius):
                        # 找到了！
                        return (x, y)

    # 迴圈結束，在 max_radius 內都沒找到
    print(f"⚠️ [警告] 在 {max_radius} 像素範圍內，找不到屬於區域 {target_label} 的安全點。")
    return None # 失敗


from collections import deque # 確保在檔案開頭 import deque
def build_bfs_distance_map_downsampled(map_crop, start_local, downsample_factor):
    """
    (✅ BBox + Downsample 優化版)
    1. 接收一個 *已經裁剪* 的地圖 (map_crop)
    2. 接收一個 *相對於* 裁剪地圖的起點 (start_local)
    3. 接收一個降採樣因子 (downsample_factor)
    
    返回一個與 map_crop 同等大小的距離圖。
    """
    
    # 1. 獲取裁剪後地圖的尺寸
    H_crop, W_crop = map_crop.shape
    if H_crop == 0 or W_crop == 0:
        print("❌ BFS 錯誤：BBox 裁剪後地圖為空。")
        return np.full((H_crop, W_crop), np.inf, dtype=np.float32)

    # 2. 降採樣地圖
    H_small = max(1, H_crop // downsample_factor)
    W_small = max(1, W_crop // downsample_factor)
    
    map_small = cv2.resize(map_crop, (W_small, H_small), 
                        interpolation=cv2.INTER_NEAREST)
    
    # 3. 計算降採樣後的起點座標
    sx_small = int(start_local[0] / downsample_factor)
    sy_small = int(start_local[1] / downsample_factor)

    # 4. 建立小地圖的距離圖
    dist_map_small = np.full((H_small, W_small), np.inf, dtype=np.float32)

    # 檢查起點是否有效
    if not (0 <= sx_small < W_small and 0 <= sy_small < H_small and \
            map_small[sy_small, sx_small] >= 128):
        print(f"❌ BFS 錯誤：降採樣後的起點 ({sx_small}, {sy_small}) 不在可行走區域。")
        # 即使起點無效，仍返回放大的 inf 地圖，讓後續邏輯失敗
    else:
        q = deque([(sx_small, sy_small)])
        dist_map_small[sy_small, sx_small] = 0.0

        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), 
                    (1, 1), (1, -1), (-1, 1), (-1, -1)]
        costs = [1.0, 1.0, 1.0, 1.0, 1.415, 1.415, 1.415, 1.415]

        while q:
            cx, cy = q.popleft()
            current_cost = dist_map_small[cy, cx]

            for i, (dx, dy) in enumerate(neighbors):
                nx, ny = cx + dx, cy + dy
                edge_cost = costs[i]
                
                # 檢查小地圖邊界
                if 0 <= nx < W_small and 0 <= ny < H_small:
                    if map_small[ny, nx] >= 128:
                        new_cost = current_cost + edge_cost
                        if new_cost < dist_map_small[ny, nx]:
                            dist_map_small[ny, nx] = new_cost
                            q.append((nx, ny))
    
    print(f"[INFO] 降採樣 BFS ( {H_small}x{W_small} ) 完成。")

    # 5. 放大 (Upscale) 距離圖
    large_val = 1e9
    dist_map_small[dist_map_small == np.inf] = large_val
    
    dist_map_crop_resized = cv2.resize(dist_map_small, (W_crop, H_crop), 
                                    interpolation=cv2.INTER_LINEAR)
    
    # 6. 恢復成本尺度
    dist_map_crop = dist_map_crop_resized * downsample_factor
    
    # 7. 恢復 inf
    dist_map_crop[dist_map_crop > (large_val * 0.9 * downsample_factor)] = np.inf
    
    return dist_map_crop


def find_nearest_safe_point_INSIDE(map_img, start_dist_map, start_point, original_goal, 
                                   max_radius=100, safe_radius=5, 
                                   path_ratio_threshold=2.0):
    """
    (✅ 新函數：專門用於修正 Goal)
    從 original_goal 開始徑向搜索，找到一個「內部」的安全點。
    「內部」定義為：
    1. 該點 P 可從 start 到達 (start_dist_map[P] < inf)
    2. 路徑比率 (L_path / L_euc) < path_ratio_threshold
    在所有滿足條件的點中，返回距離 original_goal 最近的點。
    """
    H, W = map_img.shape[:2]
    gx, gy = int(original_goal[0]), int(original_goal[1])

    inside_candidates = [] # 儲存 ( 距離goal的距離, (x, y) )

    # 0. 先檢查原點 (半徑 0)
    if 0 <= gx < W and 0 <= gy < H:
        path_cost = start_dist_map[gy, gx]
        if path_cost < np.inf and is_safe_point(map_img, gx, gy, r=safe_radius):
            euc_cost = distance(start_point, (gx, gy))
            ratio = path_cost / (euc_cost + 1e-6)
            if ratio < path_ratio_threshold:
                return (gx, gy) # 原點就在內部且安全

    # 1. 徑向 (方型) 搜索
    for r in range(1, max_radius + 1):
        # (此處使用您原有的方型搜索邏輯)
        candidates_at_r = []
        candidates_at_r += [(gx, gy - r), (gx, gy + r), (gx - r, gy), (gx + r, gy)]
        for dy in range(-r, r + 1):
            candidates_at_r.append((gx - r, gy + dy))
            candidates_at_r.append((gx + r, gy + dy))
        for dx in range(-r + 1, r):
            candidates_at_r.append((gx + dx, gy - r))
            candidates_at_r.append((gx + dx, gy + r))
        
        seen = set()
        uniq = [p for p in candidates_at_r if p not in seen and not seen.add(p)]

        # 檢查這一層的所有候選點
        for (x, y) in uniq:
            if 0 <= x < W and 0 <= y < H:
                # 檢查 1: 是否安全
                if is_safe_point(map_img, x, y, r=safe_radius):
                    # 檢查 2: 是否可從 Start 到達
                    path_cost = start_dist_map[y, x]
                    if path_cost < np.inf:
                        # 檢查 3: 是否為 "內部" (路徑比率)
                        euc_cost = distance(start_point, (x, y))
                        ratio = path_cost / (euc_cost + 1e-6)
                        
                        if ratio < path_ratio_threshold:
                            # 這是個 "內部" 候選點
                            dist_to_goal = distance((x, y), original_goal)
                            inside_candidates.append( (dist_to_goal, (x, y)) )

        # 如果在這一層 (半徑 r) 找到了任何內部點，
        # 我們就停止搜索，並從中選取離 goal 最近的
        if inside_candidates:
            # 排序：依據 "距離 goal 的距離" (item[0])
            inside_candidates.sort(key=lambda item: item[0])
            best_point = inside_candidates[0][1]
            return best_point # 找到了！

    # 迴圈結束，在 max_radius 內都沒找到
    print(f"⚠️ [警告] 在 {max_radius} 像素範圍內，找不到滿足路徑比率 (<{path_ratio_threshold}) 的安全點。")
    return None # 失敗


# ==========================================================
# RRT* 主算法
# ==========================================================
def rrt_star_planning(map_img, start, goal, SAFE_WEIGHT=10000.0):
    """
    改良版 RRT*：使用統一的倒數平方懲罰推離牆面。
    ✅ 使用 R-Tree 取代 KD-Tree 以實現 $O(\log N)$ 增量更新
    """
    H, W = map_img.shape[:2]
    # 加一道「安全膨脹」：侵蝕可行區（用來避免鑽窄縫和牆體瑕疵）
    kernel_size = 65
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    map_img = cv2.erode(map_img, np.ones((8, 8), np.uint8), iterations=1)
    map_img = cv2.morphologyEx(map_img, cv2.MORPH_OPEN, kernel)
    
    
    # 計算距離地圖 (Distance Transform)
    dist_map = cv2.distanceTransform(map_img, cv2.DIST_L2, 5)
    d_safe_max = np.percentile(dist_map, 60)

    # TODO: 隔離內部區域

    if d_safe_max < 1e-6:
        d_safe_max = 1.0
    D_SAFE_MAX_FOR_SAMPLING = d_safe_max
    is_start_modified = is_goal_modified = False
    
    original_start, original_goal = start, goal
    is_start_modified = is_goal_modified = False

    # ✅ 步驟 1: 修正 Start 點 (使用舊的 'find_nearest_safe_point_radial')
    # (我們只想找 *任何* 安全點作為 BFS 起點)
    if not (0 <= int(start[1]) < H and 0 <= int(start[0]) < W) or \
       map_img[int(start[1]), int(start[0])] < 128 or \
       not is_safe_point(map_img, start[0], start[1], r=5):
        
        print("⚠️ [提醒] 起點在障礙/不安全，尋找最近的 *任何* 安全點 ...")
        start_old = start
        # 使用 *舊的* 函數，它只關心安不安全
        start = find_nearest_safe_point_radial( 
            map_img, start, max_radius=30, safe_radius=5)
        print(f"Start: {start_old} → {start}")
        is_start_modified = (start_old != start)

    # 檢查修正後的 start
    if not (0 <= int(start[1]) < H and 0 <= int(start[0]) < W and map_img[int(start[1]), int(start[0])] >= 128):
        print(f"❌ 錯誤：修正後的起點 {start} 仍位於障礙物標籤。")
        return None, []
    
    print(f"[INFO] 已確定 Start 位於安全區域: {start}")
# ✅ 步驟 2: 計算 Bounding Box
    buffer = 100 # 您提議的緩衝區 (像素)
    DOWNSAMPLE_FACTOR = 4 # 降採樣 4 倍 (總像素減少 16 倍)
    x_min = max(0, int(min(start[0], original_goal[0]) - buffer))
    y_min = max(0, int(min(start[1], original_goal[1]) - buffer))
    x_max = min(W, int(max(start[0], original_goal[0]) + buffer))
    y_max = min(H, int(max(start[1], original_goal[1]) + buffer))
    bbox = (x_min, y_min, x_max, y_max)

    print(f"[INFO] BBox 範圍: {bbox}")
    # 裁剪地圖
    map_crop = map_img[y_min:y_max, x_min:x_max]
    
    # 計算相對於 "crop" 的起點
    start_local = (start[0] - x_min, start[1] - y_min)

    print(f"[INFO] BBox 範圍: {(x_min, y_min, x_max, y_max)}")
    print(f"[INFO] 裁剪尺寸: {map_crop.shape}, 降採樣因子: {DOWNSAMPLE_FACTOR}")

    # ✅ 步驟 3: 建立 BFS 路徑距離圖 (使用新函數)
    dist_map_crop = build_bfs_distance_map_downsampled(
        map_crop, 
        start_local, 
        DOWNSAMPLE_FACTOR
    )

    # ✅ 步驟 4: 將結果貼回 (Paste) 到全尺寸地圖
    start_dist_map = np.full(map_img.shape, np.inf, dtype=np.float32)
    start_dist_map[y_min:y_max, x_min:x_max] = dist_map_crop


    # ✅ 步驟 3: 修正 Goal 點
    # 檢查：Goal 是否已在 "內部" 且安全？
    gx_i, gy_i = int(goal[0]), int(goal[1])
    is_safe = False
    is_inside = False
    
    if 0 <= gx_i < W and 0 <= gy_i < H:
        path_cost = start_dist_map[gy_i, gx_i]
        if path_cost < np.inf: # < inf 代表可到達
            euc_cost = distance(start, goal)
            ratio = path_cost / (euc_cost + 1e-6)
            if ratio < 2.0: # (使用與新函數相同的預設 threshold)
                is_inside = True
            is_safe = is_safe_point(map_img, goal[0], goal[1], r=5)

    if is_safe and is_inside:
        print("[INFO] Goal 點有效且位於「內部」，無需修正。")
    else:
        # === 否，Goal 需要修正 ===
        if not is_inside:
            print(f"⚠️ [提醒] 目標 {original_goal} 在障礙上或位於「外部」區域。")
        else: # (is_inside = True, is_safe = False)
             print(f"⚠️ [提醒] 目標 {original_goal} 位於「內部」但不夠安全。")
        
        print(f"    ... 正在搜尋 {original_goal} 附近，最接近的「內部」安全點...")
        
        goal_old = goal
        # ✅✅✅ 使用我們新開發的、基於 BFS 的函數！ ✅✅✅
        goal = find_nearest_safe_point_INSIDE(
            map_img, 
            start_dist_map, 
            start,          # 傳入修正後的 start 點
            goal_old, 
            max_radius=180, # 增加搜索半徑
            safe_radius=5,
            path_ratio_threshold=2.0 # 可調整此比率
        )
        
        if goal is None:
            # 如果新函數返回 None，代表在 180px 內找不到替代點
            print(f"❌ 錯誤：在原始目標 {goal_old} 附近 180px 內，找不到「內部」安全替代點。")
            return None, []
            
        print(f"Goal: {goal_old} → {goal}")
        is_goal_modified = (goal_old != goal)
    
    # # === (起點/目標點檢查 結束) ===
    # # ==========================================================
    # if is_start_modified or is_goal_modified:
    #     plt.figure(figsize=(8, 8))
    #     # 顯示處理過的安全地圖 (morphologyEx/erode後的版本)
    #     plt.imshow(map_img, cmap='gray') 
    #     plt.title("Start/Goal 修正結果 (綠色→原始點, 紅色→修正後的安全點)")
        
    #     # 繪製 Start 點
    #     if is_start_modified:
    #         # 原始點 (標記為綠色圓圈)
    #         plt.plot(start_old[0], start_old[1], "go", markersize=10, 
    #                  markerfacecolor='none', markeredgecolor='green', label="Original Start")
    #         # 修正後的點 (標記為綠色X)
    #         plt.plot(start[0], start[1], "gx", markersize=10, label="Safe Start")
    #         # 畫一條線連接
    #         plt.plot([start_old[0], start[0]], [start_old[1], start[1]], "g--", linewidth=1.0)
    #     else:
    #          plt.plot(start[0], start[1], "gx", markersize=10, label="Start (No change)")
        
    #     # 繪製 Goal 點
    #     if is_goal_modified:
    #         # 原始點 (標記為紅色圓圈)
    #         plt.plot(goal_old[0], goal_old[1], "ro", markersize=10, 
    #                  markerfacecolor='none', markeredgecolor='red', label="Original Goal")
    #         plt.plot(goal[0], goal[1], "rx", markersize=10, label="Safe Goal")
    #         plt.plot([goal_old[0], goal[0]], [goal_old[1], goal[1]], "r--", linewidth=1.0)
    #     else:
    #          plt.plot(goal[0], goal[1], "rx", markersize=10, label="Goal (No change)")

    #     plt.legend()
    #     plt.axis("equal")
    #     plt.show() # 暫停執行，等待使用者關閉視窗
    #     plt.close()        

    # =====================================
    # print("[INFO] 正在計算可行走區域的連通元件...")
    # map_img 是二值化地圖 (0=障礙, 255=可行走)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(map_img, 8, cv2.CV_32S)
    # print(f"共有:{num_labels}個標籤")
    
    
    # # ‼️ 【【【新增的除錯視覺化程式碼】】】
    # print("[INFO] 正在產生標籤視覺化地圖 (Label Map)...")
    
    # # 1. 建立一個只顯示「障礙物 (label=0)」的遮罩
    # #    (label == 0) 會產生一個布林陣列，True 的地方就是 label 0
    # #    np.uint8(...) * 255 將 True 轉為 255 (白色)，False 轉為 0 (黑色)
    # obstacle_mask_vis = np.uint8(labels == 0) * 255
    
    # # 2. 建立一個顯示「所有」連通區域的彩色地圖
    # #    標準化 labels 陣列 (從 0~num_labels 映射到 0~255)
    # labels_vis = cv2.normalize(labels, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # #    套用色彩映射 (JET  colormap)
    # labels_color_vis = cv2.applyColorMap(labels_vis, cv2.COLORMAP_JET)
    # #    (重要) 把 label 0 (障礙物) 的地方強制塗成黑色，這樣比較容易看
    # labels_color_vis[labels == 0] = [0, 0, 0]

    # # 3. 使用 Matplotlib 顯示
    # plt.figure(figsize=(12, 6))
    
    # # 圖一：只顯示障礙物 (Label 0)
    # plt.subplot(1, 2, 1)
    # plt.imshow(obstacle_mask_vis, cmap='gray')
    # plt.title(f"Obstacles (Label=0) ONLY\n(侵蝕後的 {kernel_size}x{kernel_size} kernel)")
    # plt.scatter(start[0], start[1], c='lime', marker='x', s=100, label="Start")
    # plt.scatter(goal[0], goal[1], c='red', marker='x', s=100, label="Goal")
    # plt.legend()

    # # 圖二：顯示所有連通元件
    # plt.subplot(1, 2, 2)
    # plt.imshow(cv2.cvtColor(labels_color_vis, cv2.COLOR_BGR2RGB))
    # plt.title(f"All Connected Components\n(Total: {num_labels} labels)")
    
    # # 標出 Start 和 Goal 實際落在哪個標籤上
    # try:
    #     start_label_vis = labels[int(start[1]), int(start[0])]
    #     goal_label_vis = labels[int(goal[1]), int(goal[0])]
    #     plt.scatter(start[0], start[1], c='lime', marker='x', s=100, 
    #                 label=f"Start (落在 Label {start_label_vis})")
    #     plt.scatter(goal[0], goal[1], c='red', marker='x', s=100, 
    #                 label=f"Goal (落在 Label {goal_label_vis})")
    # except IndexError:
    #     print("警告：Start/Goal 座標可能在標籤圖之外。")
        
    # plt.legend()
    
    # print("[INFO] 顯示標籤地圖。請檢查 Start/Goal 是否落在同一個非 0 標籤上。")
    # print("       (關閉彈出視窗後，程式將繼續執行...)")
    # plt.show() # <--- 程式會暫停在這裡，直到你關閉視窗
    # plt.close()
    # # ‼️ 【【【除錯程式碼結束】】】
    
    # 獲取修正後的 start/goal 所在的區域標籤
    # 注意：labels 的索引是 (y, x)
    start_label = labels[int(start[1]), int(start[0])]
    goal_label = labels[int(goal[1]), int(goal[0])]
    # 檢查：
    # 1. start/goal 是否落在障礙物上 (label=0) (雖然前面修正過，但 double check)
    # 2. start/goal 是否在不同的連通區域 (無法到達)
    if start_label == 0:
        print(f"❌ 錯誤：修正後的起點 {start} 位於障礙物標籤 (label=0)。")
        return None, []
    if goal_label == 0:
        print(f"❌ 錯誤：修正後的目標 {goal} 位於障礙物標籤 (label=0)。")
        return None, []
    if start_label != goal_label:
        print(
            f"❌ 錯誤：起點 (區域 {start_label}) 和目標 (區域 {goal_label}) 位於不同的連通區域，路徑不存在。")
        return None, []

    print(f"[INFO] Start/Goal 均位於連通區域 {start_label}。")
    # 建立一個只包含 'start_label' 區域的遮罩 (Mask)
    sampling_mask = np.uint8(labels == start_label) * 255

    # 從遮罩中找出所有可行走點的座標 (x, y)
    # free_coords 是一個 (N, 1, 2) 的陣列, 格式為 (x, y)
    free_coords_cv = cv2.findNonZero(sampling_mask)

    if free_coords_cv is None:
        print(f"❌ 錯誤：在區域 {start_label} 中找不到任何可行走點。")
        return None, []

    # 將 (N, 1, 2) 轉換為 (N, 2) 以便快速索引
    free_coords = free_coords_cv.squeeze(1)
    print(f"[INFO] 已快取 {len(free_coords)} 個有效取樣點。")
    # === 初始化 ===
    rng = random.Random(12345)
    start_node = Node(start[0], start[1], parent=None, cost=0.0)
    nodes = [start_node]  # 我們仍然需要 list 來儲存 Node 物件
    best_goal_node = None
    c_best = float("inf")

    # ✅ 【優化】: 初始化 R-Tree
    # R-Tree 儲存 (id, (x, y, x, y), object)
    # 這裡的 id 我們直接用 node 在 'nodes' list 中的索引 (index)
    p = index.Property()
    p.dimension = 2
    # 建立 R-Tree 索引。R-Tree 中的 'id' 對應 'nodes' list 的索引
    rtree = index.Index(properties=p)
    # 插入起點 (id=0)
    rtree.insert(0, (start_node.x, start_node.y, start_node.x, start_node.y))

    # (參數) 靠近目標多近時 (px)，停止計算安全懲罰
    GOAL_SAFETY_EXEMPT_DIST = 40.0
    MIN_SAFE_DIST = 8.0 * DPI//150  # 絕對最小安全距離（像素）

    for it in range(MAX_ITER):
        is_goal_sample = False
        sample = None
        
        # --- (取樣 ... 保持不變) ---
        if rng.random() < GOAL_SAMPLE_RATE:
            sample = goal
            if not is_free_pixel(map_img, sample[0], sample[1]):
                continue
            is_goal_sample = True
        else:
            if INFORMED_SAMPLING and np.isfinite(c_best):
                sample = sample_informed(
                    start, goal, c_best, rng, (H, W), labels, start_label)
            else:
                rand_idx = rng.randint(0, len(free_coords) - 1)
                sample = tuple(free_coords[rand_idx])

            if sample is None:
                continue
            if not is_free_pixel(map_img, sample[0], sample[1]):
                continue

        # --- (延伸/安全採樣檢查 ... ) ---

        # ✅ 【修改】: 使用 R-Tree 查詢最近鄰 (Nearest)
        # R-Tree 查詢點 (x, y) 必須提供 bounding box (x, y, x, y)
        # 查詢返回的是 generator，用 next() 取第一個 (k=1)
        # R-Tree.nearest 返回的是 id (即 'nodes' list 的索引)
        try:
            nearest_id = next(rtree.nearest(
                (sample[0], sample[1], sample[0], sample[1]), 1))
            nearest_node = nodes[nearest_id]
        except StopIteration:
            continue  # R-Tree 為空？ (理論上不會發生)

        # ... (安全檢查邏輯 ... 保持不變)
        if dist_map is not None and not is_goal_sample:
            iy = int(np.clip(sample[1], 0, dist_map.shape[0]-1))
            ix = int(np.clip(sample[0], 0, dist_map.shape[1]-1))
            d_safe_sample = dist_map[iy, ix]

            # if d_safe_sample < MIN_SAFE_DIST:
            #     continue
            heading = math.atan2(
                sample[1] - nearest_node.y, sample[0] - nearest_node.x)
            sector_vals = get_sector_distances(dist_map, (nearest_node.x, nearest_node.y),
                                            heading_rad=heading, fov_deg=150, radius=25)
            if len(sector_vals) > 10:
                d_safe_Q3 = np.percentile(sector_vals, 25)
            else:
                d_safe_Q3 = D_SAFE_MAX_FOR_SAMPLING
            if d_safe_sample <= d_safe_Q3:
                continue
        # ... (安全檢查結束)

        new_node = steer(nearest_node, sample, STEP_SIZE)
        if new_node is None:
            continue
        if not is_free_pixel(map_img, new_node.x, new_node.y):
            continue
        if not line_collision_free(map_img, nearest_node.pt, new_node.pt):
            continue

        # ---------------------------
        if dist_map is not None:
            # 獲取 new_node 的安全距離
            iy = int(np.clip(new_node.y, 0, dist_map.shape[0]-1))
            ix = int(np.clip(new_node.x, 0, dist_map.shape[1]-1))
            d_safe_new_node = dist_map[iy, ix]

            # 獲取 new_node 到目標的距離
            dist_to_goal = distance(new_node.pt, goal)

            # 僅在離目標一定距離外 (e.g. > 40px) 才強制執行絕對安全距離
            # 允許在接近目標時 (e.g. < 40px) 稍微貼牆
            if dist_to_goal > GOAL_SAFETY_EXEMPT_DIST:
                if d_safe_new_node < MIN_SAFE_DIST:
                            continue
        new_node_penalty = get_safety_penalty(
            new_node.pt, goal, dist_map, SAFE_WEIGHT, GOAL_SAFETY_EXEMPT_DIST)

        # --- 選擇最佳 parent (Choose Parent) ---
        n = len(nodes)
        radius = NEIGHBOR_COEFF * math.sqrt(max(math.log(n) / n, 1e-9))

        # ✅ 【修改】: 使用 R-Tree 查詢鄰居 (Near / Ball Query)
        # R-Tree 範圍查詢 (intersection) 需要 Bounding Box
        bounds = (
            new_node.x - radius, new_node.y - radius,
            new_node.x + radius, new_node.y + radius
        )
        # R-Tree.intersection 返回 id 的 generator
        indices = list(rtree.intersection(bounds))
        neighbors = [nodes[i] for i in indices]

        # 計算初始最佳成本 (parent: nearest_node)
        best_parent = nearest_node
        # 邊緣成本 = 幾何距離 + new_node 的懲罰
        best_edge_cost = distance(
            best_parent.pt, new_node.pt) + new_node_penalty
        best_cost = best_parent.cost + best_edge_cost

        for nb in neighbors:
            # ... (Choose Parent 邏輯 ... 保持不變)
            geometric_edge_cost = distance(nb.pt, new_node.pt)
            cand_cost = nb.cost + geometric_edge_cost + new_node_penalty
            if cand_cost + 1e-6 < best_cost and line_collision_free(map_img, nb.pt, new_node.pt):
                best_parent = nb
                best_cost = cand_cost

        # === 新節點加入 ===
        new_node.parent = best_parent
        new_node.cost = best_cost

        # ✅ 【修改】: 增量更新 R-Tree
        # 1. 先將新節點加入 'nodes' list
        nodes.append(new_node)
        # 2. 獲取新節點的索引 (id)
        new_node_id = len(nodes) - 1
        # 3. 將 (id, bounds) 插入 R-Tree
        rtree.insert(new_node_id, (new_node.x,
                     new_node.y, new_node.x, new_node.y))

        # ❌ 【移除】: 刪除 vstack 和 KDTree 重建
        # node_coords = np.vstack([node_coords, new_node.pt])
        # kdtree = KDTree(node_coords)

        # --- Rewire ---
        for nb in neighbors:
            if nb is new_node or nb is best_parent:
                continue

            # ... (Rewire 邏輯 ... 保持不變)
            geometric_edge_cost = distance(new_node.pt, nb.pt)
            nb_penalty = get_safety_penalty(
                nb.pt, goal, dist_map, SAFE_WEIGHT, GOAL_SAFETY_EXEMPT_DIST)
            segment_cost = geometric_edge_cost + nb_penalty
            new_cost = new_node.cost + segment_cost
            if new_cost + 1e-6 < nb.cost and line_collision_free(map_img, new_node.pt, nb.pt):
                nb.parent = new_node
                nb.cost = new_cost

        # --- (嘗試接通 Goal ... 保持不變) ---
        if distance(new_node.pt, goal) <= GOAL_REACH_THRESH:
            if line_collision_free(map_img, new_node.pt, goal):
                final_cost = new_node.cost + distance(new_node.pt, goal)
                goal_node = Node(goal[0], goal[1],
                                parent=new_node, cost=final_cost)
                if goal_node.cost < c_best:
                    best_goal_node = goal_node
                    c_best = goal_node.cost
        # -----------------------------------

    # === 收尾 ===
    if best_goal_node is None:
        print("❌ 未找到可行路徑。")
        return None, nodes

    raw_path = extract_path(best_goal_node)
    # 啟用平滑化
    # raw_path = smooth_path(map_img, raw_path, iterations=SMOOTH_ITER)

    return raw_path, nodes

# ==========================================================
# 主流程
# ==========================================================
if __name__ == "__main__":
    print("=== HW2 Part2: Semantic-guided RRT* Path Planning (w/ Informed + Smoothing) ===")

    # 讀地圖並二值化（白=free=255，黑=obs=0）
    map_img_gray = cv2.imread(MAP_PATH, cv2.IMREAD_GRAYSCALE)
    if map_img_gray is None:
        raise FileNotFoundError(MAP_PATH)
    _, binary_map = cv2.threshold(map_img_gray, 240, 255, cv2.THRESH_BINARY)

    # Step 1. 載入語意表與地圖 (略)
    color_map = load_semantic_table(EXCEL_PATH)
    id_map = load_semantic_ID_table(EXCEL_PATH)
    map_img = cv2.imread(MAP_PATH)
    if map_img is None:
        raise FileNotFoundError(f"❌ 找不到地圖: {MAP_PATH}")

    # Step 2. 掃描地圖中實際出現的顏色
    unique_colors = np.unique(map_img.reshape(-1, 3), axis=0)
    unique_colors_set = {tuple(color.tolist()) for color in unique_colors}
    available_classes = []
    for name, rgb in color_map.items():
        bgr = tuple(reversed(rgb))
        if bgr in unique_colors_set:
            available_classes.append(name)
    if not available_classes:
        raise RuntimeError("❌ map.png 找不到任何語意類別，請確認顏色與語意表一致。")
    print(f"[INFO] 可用的目標類別（出現在地圖上）：{available_classes}")

    # Step 3. 讓使用者輸入目標並選起點
    target_class = input(f"請輸入目標類別 {available_classes}: ").strip().lower()
    if target_class not in available_classes:
        raise ValueError(f"⚠️ '{target_class}' 不在地圖可用清單中。")
    goals_list,goal_mask = find_all_object_instances(MAP_PATH, color_map, target_class)

    if not goals_list or goal_mask is None: 
        raise ValueError(f"⚠️ 找不到目標類別 '{target_class}' 的任何區域。")
    
    print(f"[INFO] 將 {target_class} 區域 (來自 color map) 合併到可行走地圖 (binary_map) 中...")
    binary_map_with_goal = cv2.bitwise_or(binary_map, goal_mask)
    # 顯示所有找到的窗戶
    # visualize_multiple_goals(MAP_PATH, goals_list, target_class)

    # (3) 目前先自動選擇第一個。
    # TODO: 未來您可以修改這裡，例如讓使用者點選，或自動找最近的。
    while True:
        goal_idx = input(f"請輸入要找第幾個窗戶（1～{len(goals_list)}，只能輸入一個數字）: ")

        # 檢查是否為純數字且範圍正確
        if goal_idx.isdigit():
            goal_idx = int(goal_idx)
            if 1 <= goal_idx <= len(goals_list):
                break
            else:
                print(f"⚠️ 請輸入 1～{len(goals_list)} 之間的數字。")
        else:
            print("⚠️ 無效輸入，請輸入數字。")

    goal = goals_list[goal_idx - 1]
    print(f"[INFO] 已自動選擇 {len(goals_list)} 個目標中的第 1 個: {goal} 作為 RRT* 終點。")

    # 挑選起點
    start = select_start(MAP_PATH, goal)

    # 執行 RRT* (保持不變)
    path, nodes = rrt_star_planning(binary_map_with_goal, start, goal, SAFE_WEIGHT=500000)

    # 顯示結果
    if path:
        print(
            f"[INFO] 路徑長度（像素）: {path_length_px(path):.1f} | 節點數: {len(nodes)}")

        # 繪製最終路徑（基於平滑化後的 path）
        result_img = cv2.cvtColor(binary_map, cv2.COLOR_GRAY2BGR)
        for i in range(len(path) - 1):
            p1 = (int(path[i][0]), int(path[i][1]))
            p2 = (int(path[i + 1][0]), int(path[i + 1][1]))
            cv2.line(result_img, p1, p2, (0, 0, 255), 2)

        out = f"rrt_star_result_{target_class}.png"
        cv2.imwrite(out, result_img)
        print(f"[INFO] 結果已儲存至 {out}")
    else:
        print("[INFO] 無路徑產生")

    visualize_rrt(binary_map, nodes, start, goal, path)
