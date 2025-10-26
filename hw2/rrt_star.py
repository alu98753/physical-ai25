import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
from collections import deque
import os
import scipy
from scipy.spatial import KDTree

# ==========================================================
# 路徑設定
# ==========================================================
currdir = os.path.dirname(os.path.abspath(__file__))
MAP_PATH = os.path.join(currdir, "map.png")
EXCEL_PATH = os.path.join(
    currdir, "color_coding_semantic_segmentation_classes.xlsx")

# ==========================================================
# RRT* 參數（像素座標系）
# ==========================================================
STEP_SIZE = 5      # 每次延伸步長（px）
MAX_ITER = 40000     # 迭代上限
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


def sample_informed(start, goal, c_best, rng, map_shape):
    # 若 c_best 無限大，回傳 None 代表退回 uniform
    c_min = distance(start, goal)
    if not np.isfinite(c_best) or c_best <= c_min + 1e-6:
        return None

    # 橢圓參數
    a = c_best / 2.0       # 橢圓長半軸
    b = math.sqrt(max(a*a - (c_min/2.0)**2, 1e-6))  # 短半軸
    # 旋轉角（start→goal）
    theta = math.atan2(goal[1]-start[1], goal[0]-start[0])

    # 在單位圓內取樣，再放縮成橢圓
    r = math.sqrt(rng.random())
    ang = 2*math.pi*rng.random()
    x_e = r * math.cos(ang) * a
    y_e = r * math.sin(ang) * b

    # 旋轉 & 平移到世界座標（像素）
    c = math.cos(theta)
    s = math.sin(theta)
    x = x_e * c - y_e * s + (start[0] + goal[0]) / 2.0
    y = x_e * s + y_e * c + (start[1] + goal[1]) / 2.0

    # 保底：若落在障礙或越界就返回 None（外層會改用 uniform）
    x_i, y_i = int(round(x)), int(round(y))
    H, W = map_shape
    if 0 <= x_i < W and 0 <= y_i < H:
        return (x_i, y_i)
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
# RRT* 主算法
# ==========================================================
def rrt_star_planning(map_img, start, goal, dist_map=None, SAFE_WEIGHT=10000.0, D_SAFE_MAX_FOR_SAMPLING=1.0):
    """
    改良版 RRT*：使用統一的倒數平方懲罰推離牆面。
    """
    H, W = map_img.shape[:2]

    # === (起點/目標點檢查 ... 保持不變) ===
    if not (0 <= start[0] < W and 0 <= start[1] < H) or map_img[int(start[1]), int(start[0])] < 128 \
            or not is_safe_point(map_img, start[0], start[1], r=5):
        print("⚠️ [提醒] 起點在障礙/不安全，尋找最近安全點 ...")
        start_old = start
        start = find_nearest_safe_point_radial(
            map_img, start, max_radius=30, safe_radius=5)
        print(f"✅ [修正] Start: {start_old} → {start}")

    if not (0 <= goal[0] < W and 0 <= goal[1] < H) or map_img[int(goal[1]), int(goal[0])] < 128 \
            or not is_safe_point(map_img, goal[0], goal[1], r=5):
        print("⚠️ [提醒] 目標在障礙/不安全，沿連線尋找安全替代點 ...")
        goal_old = goal
        goal = find_safe_goal_along_line(
            map_img, start, goal, step=1, safe_radius=5)
        print(f"✅ [修正] Goal: {goal_old} → {goal}")
    # =====================================

    # === 初始化 ===
    rng = random.Random(12345)
    start_node = Node(start[0], start[1], parent=None, cost=0.0)
    nodes = [start_node]
    best_goal_node = None
    c_best = float("inf")

    # (參數) 靠近目標多近時 (px)，停止計算安全懲罰
    GOAL_SAFETY_EXEMPT_DIST = 40.0
    MIN_SAFE_DIST = 8.0  # 絕對最小安全距離（像素） <--- 新增此變量
    
    for it in range(MAX_ITER):
        # --- (取樣 ... 保持不變) ---
        if rng.random() < GOAL_SAMPLE_RATE:
            sample = goal
            if not is_free_pixel(map_img, sample[0], sample[1]):
                continue
        else:
            if INFORMED_SAMPLING and np.isfinite(c_best):
                sample = sample_informed(start, goal, c_best, rng, (H, W))
                if sample is None:
                    sample = (rng.randrange(W), rng.randrange(H))
            else:
                sample = (rng.randrange(W), rng.randrange(H))

            if not is_free_pixel(map_img, sample[0], sample[1]):
                continue

        # --- (延伸/安全採樣檢查 ... 保持不變) ---
        nearest_node = nearest(nodes, sample)


        if dist_map is not None:
            iy = int(np.clip(sample[1], 0, dist_map.shape[0]-1))
            ix = int(np.clip(sample[0], 0, dist_map.shape[1]-1))
            d_safe_sample = dist_map[iy, ix]

            # 絕對最小距離檢查：如果採樣點太貼牆，立即拒絕
            if d_safe_sample < MIN_SAFE_DIST:
                  continue

            # 2. 局部貼牆拒絕（修正後的扇形邏輯）
            # 計算從 nearest_node 指向 sample 的方向
            heading = math.atan2(
                sample[1] - nearest_node.y, sample[0] - nearest_node.x)

            # 獲取前方扇形區域的安全距離分佈
            sector_vals = get_sector_distances(dist_map, (nearest_node.x, nearest_node.y),
                                              heading_rad=heading, fov_deg=150, radius=25)

            if len(sector_vals) > 10:
                # 找出 "最貼牆的 75% 距離" 的上限
                # 亦即：將所有距離由小到大排列，取第 75% 的值。
                # 距離越小代表越貼牆。
                # 如果這個值（d_safe_Q3）很小，表示這個區域整體都很貼牆。
                d_safe_Q3 = np.percentile(sector_vals, 25)
            else:
                d_safe_Q3 = D_SAFE_MAX_FOR_SAMPLING

            # 修正邏輯：如果採樣點的安全距離 d_safe_sample 落在
            # 扇形區域內 "最貼牆的 75% 距離" 範圍內，則拒絕。
            # 換句話說，只接受 d_safe_sample 位於 d_safe_Q3 以外（更寬敞）的採樣。
            if d_safe_sample <= d_safe_Q3:
                continue

        new_node = steer(nearest_node, sample, STEP_SIZE)
        if new_node is None:
            continue
        if not is_free_pixel(map_img, new_node.x, new_node.y):
            continue
        if not line_collision_free(map_img, nearest_node.pt, new_node.pt):
            continue
        # ---------------------------

        new_node_penalty = get_safety_penalty(
            new_node.pt, goal, dist_map, SAFE_WEIGHT, GOAL_SAFETY_EXEMPT_DIST)
        
        # --- 選擇最佳 parent (Choose Parent) ---
        n = len(nodes)
        radius = NEIGHBOR_COEFF * math.sqrt(max(math.log(n) / n, 1e-9))
        neighbors = near(nodes, new_node, radius) or [nearest_node]

        # 計算初始最佳成本 (parent: nearest_node)
        best_parent = nearest_node
        # 邊緣成本 = 幾何距離 + new_node 的懲罰
        best_edge_cost = distance(
            best_parent.pt, new_node.pt) + new_node_penalty
        best_cost = best_parent.cost + best_edge_cost

        for nb in neighbors:
            # 1. 計算幾何邊緣成本
            geometric_edge_cost = distance(nb.pt, new_node.pt)

            # 2. 總成本 = nb.cost + 幾何邊緣成本 + new_node 的懲罰
            cand_cost = nb.cost + geometric_edge_cost + new_node_penalty

            # 3. 檢查是否有更好的 parent
            if cand_cost + 1e-6 < best_cost and line_collision_free(map_img, nb.pt, new_node.pt):
                best_parent = nb
                best_cost = cand_cost

        # === 新節點加入 ===
        new_node.parent = best_parent
        new_node.cost = best_cost
        nodes.append(new_node)


        # --- Rewire ---
        for nb in neighbors:
            if nb is new_node or nb is best_parent:
                  continue

            # 1. 計算幾何邊緣成本 (new_node -> nb)
            geometric_edge_cost = distance(new_node.pt, nb.pt)

            # 2. ✅【優化 1】: 計算鄰居 (nb) 的懲罰
            nb_penalty = get_safety_penalty(
                nb.pt, goal, dist_map, SAFE_WEIGHT, GOAL_SAFETY_EXEMPT_DIST)

            # 3. 邊緣成本 = 幾何距離 + nb 的懲罰
            segment_cost = geometric_edge_cost + nb_penalty

            # 4. 計算新的總成本
            new_cost = new_node.cost + segment_cost

            # 5. 比較成本並更新
            if new_cost + 1e-6 < nb.cost and line_collision_free(map_img, new_node.pt, nb.pt):
                nb.parent = new_node
                nb.cost = new_cost

        # --- (嘗試接通 Goal ... 保持不變) ---
        if distance(new_node.pt, goal) <= GOAL_REACH_THRESH:
            if line_collision_free(map_img, new_node.pt, goal):

                # 抵達 Goal 時，不計算安全懲罰，只用距離
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
    raw_path = smooth_path(map_img, raw_path, iterations=SMOOTH_ITER)

    return raw_path, nodes

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
    # ... (保持不變)
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
    # ... (保持不變)
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


def visualize_rrt(map_img, nodes, start, goal, path):
    # ... (保持不變)
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

    # 加一道「安全膨脹」：侵蝕可行區（用來避免鑽窄縫和牆體瑕疵）
    kernel = np.ones((15, 15), np.uint8)
    binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_OPEN, kernel)
    binary_map = cv2.erode(binary_map, kernel, iterations=1)

    # 計算距離地圖 (Distance Transform)
    dist_map = cv2.distanceTransform(binary_map, cv2.DIST_L2, 5)
    d_safe_max = np.percentile(dist_map, 60)

    if d_safe_max < 1e-6:
        d_safe_max = 1.0

    # Step 1. 載入語意表與地圖 (略)
    color_map = load_semantic_table(EXCEL_PATH)
    id_map = load_semantic_ID_table(EXCEL_PATH)
    map_img = cv2.imread(MAP_PATH)
    if map_img is None:
        raise FileNotFoundError(f"❌ 找不到地圖: {MAP_PATH}")

    # Step 2. 掃描地圖中實際出現的顏色 (略)
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
    goal, mask = find_object_region(MAP_PATH, color_map, target_class)
    start = select_start(MAP_PATH, goal)

    # 執行 RRT*
    # ✅ 調整 SAFE_WEIGHT，採用一個巨大的懲罰係數來推離牆壁
    path, nodes = rrt_star_planning(binary_map, start, goal, dist_map,
                                    SAFE_WEIGHT=500000,
                                    D_SAFE_MAX_FOR_SAMPLING=d_safe_max)
    # 顯示結果
    if path:
        visualize_rrt(binary_map, nodes, start, goal, path)
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
