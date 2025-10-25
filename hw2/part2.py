import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
from collections import deque

# ==========================================================
# 路徑設定
# ==========================================================
MAP_PATH = "/home/clu98753cs13/Desktop/course/phyai/physical-ai25/hw2/map.png"
EXCEL_PATH = "/home/clu98753cs13/Desktop/course/phyai/physical-ai25/hw2/color_coding_semantic_segmentation_classes.xlsx"

# ==========================================================
# RRT* 參數（像素座標系）
# ==========================================================
STEP_SIZE          = 5             # 每次延伸步長（px）
MAX_ITER           = 10000           # 迭代上限
GOAL_SAMPLE_RATE   = 0.05           # 目標偏置機率
NEIGHBOR_COEFF     = 60.0           # 鄰居半徑係數 (r = coeff * sqrt(log(n)/n))
SMOOTH_ITER        = 50            # 路徑平滑化嘗試次數
INFORMED_SAMPLING  = True           # 找到初始路徑後啟用 Informed RRT*
GOAL_REACH_THRESH  = 1.5*STEP_SIZE  # 新節點到目標多少距離內視為可接通
COLLISION_SAMPLES_PER_STEP = 2      # 線段碰撞取樣密度（距離/STEP_SIZE*此係數）

# ==========================================================
# 結構
# ==========================================================
class Node:
    __slots__ = ("x","y","parent","cost")
    def __init__(self, x:float, y:float, parent=None, cost:float=0.0):
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
    支援浮點座標的安全性檢查：
    - 在 (x, y) 周圍取 13 個鄰點
    - 只要其中一個是 free (像素值 >=128) 即視為 free
    - radius 控制取樣的距離 (像素)：若要放大探測區域（像素距離），可乘上 radius
    """
    H, W = map_img.shape[:2]
    cx, cy = int(round(x)), int(round(y))

    # 13 點 pattern（中心 + 第一圈 8 點 + 第二圈 4 點）
    offsets = [
        (0, 0),  # 中心
        (-1, 0), (1, 0), (0, -1), (0, 1),  # 十字
        (-1, -1), (-1, 1), (1, -1), (1, 1),  # 斜角
        (-2, 0), (2, 0), (0, -2), (0, 2)  
    ]

    for dx, dy in offsets:
        xx = int(round(cx + dx * radius))
        yy = int(round(cy + dy * radius))
        if 0 <= xx < W and 0 <= yy < H:
            if map_img[yy, xx] >= 128:
                return True

    return False



def line_collision_free(map_img, p1, p2):
    """支援浮點採樣的線段碰撞"""
    dist = distance(p1, p2)
    n = int(COLLISION_SAMPLES_PER_STEP * dist / max(1.0, STEP_SIZE)) + 2
    xs = np.linspace(p1[0], p2[0], n)
    ys = np.linspace(p1[1], p2[1], n)
    for x, y in zip(xs, ys):
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
            best_d = d; best = n
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
    if not path or len(path) < 2: return 0.0
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
    a = c_best / 2.0               # 橢圓長半軸
    b = math.sqrt(max(a*a - (c_min/2.0)**2, 1e-6))  # 短半軸
    # 旋轉角（start→goal）
    theta = math.atan2(goal[1]-start[1], goal[0]-start[0])

    # 在單位圓內取樣，再放縮成橢圓
    r = math.sqrt(rng.random())
    ang = 2*math.pi*rng.random()
    x_e = r * math.cos(ang) * a
    y_e = r * math.sin(ang) * b

    # 旋轉 & 平移到世界座標（像素）
    c = math.cos(theta); s = math.sin(theta)
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

# ==========================================================
# RRT* 主算法（介面與舊 rrt_planning 相容）
# ==========================================================
# ==========================================================
# RRT* 主算法（支援 goal 在障礙上 → 轉為最近空白點）
# ==========================================================
def rrt_star_planning(map_img, start, goal):
    """
    參數：
      - map_img: 二值地圖 (uint8)，255 可行走、0 障礙
      - start, goal: (x,y) 像素座標
    回傳：
      - path: [(x,y), ...] 或 None
      - nodes: [Node, ...]（包含 parent/cost）
    """
    H, W = map_img.shape[:2]

    # === ✅ 起點合法性檢查 ===
    if not is_free_pixel(map_img, start[0], start[1]):
        plt.figure(figsize=(8, 8))
        plt.imshow(map_img, cmap='gray')
        plt.scatter(start[0], start[1], c='red', s=80, label='Start (在障礙上!)')
        plt.scatter(goal[0], goal[1], c='green', s=60, label='Goal')
        plt.title("⚠️ 起點落在障礙上 - 請重新選擇起點")
        plt.legend()
        plt.axis("equal")
        plt.savefig("start_on_obstacle.png", bbox_inches="tight", dpi=200)
        plt.close()
        raise ValueError("❌ 起點 (Start) 落在障礙區，請重新選擇位置。")

    # === ✅ 目標檢查：允許在障礙上，但會自動修正 ===
    if not is_free_pixel(map_img, goal[0], goal[1]):
        print("⚠️ [提醒] 目標 (Goal) 位於障礙區，嘗試尋找最近可行走點 ...")
        goal_old = goal
        goal = find_safe_goal_along_line(map_img, start, goal, step=1, safe_radius=5)
        print(f"✅ [修正] Goal: {goal_old} → {goal}")
        if goal == goal_old:
            print("❌ [警告] 找不到任何可行走點，將仍以原目標執行。")
        else:
            print(f"✅ [修正] 已將目標從 {goal_old} → {goal}")
        plt.imshow(map_img, cmap='gray')
        plt.scatter(start[0], start[1], c='blue', label='Start')
        plt.scatter(goal_old[0], goal_old[1], c='red', label='Original Goal')
        plt.scatter(goal[0], goal[1], c='green', label='Safe Goal')
        plt.legend()
        plt.axis('equal')
        plt.savefig("debug_safe_goal_line.png", dpi=200)

    # === 初始化 ===
    rng = random.Random(12345)
    start_node = Node(start[0], start[1], parent=None, cost=0.0)
    nodes = [start_node]
    best_goal_node = None
    c_best = float("inf")

    for it in range(MAX_ITER):
        # --- 取樣 ---
        if rng.random() < GOAL_SAMPLE_RATE:
            sample = goal
        else:
            if INFORMED_SAMPLING and np.isfinite(c_best):
                sample = sample_informed(start, goal, c_best, rng, (H, W))
                if sample is None:
                    sample = (rng.randrange(W), rng.randrange(H))
            else:
                sample = (rng.randrange(W), rng.randrange(H))

        # 無效取樣直接跳過
        if not is_free_pixel(map_img, sample[0], sample[1]):
            continue

        # --- 延伸 ---
        nearest_node = nearest(nodes, sample)
        new_node = steer(nearest_node, sample, STEP_SIZE)
        if new_node is None:
            continue
        if not is_free_pixel(map_img, new_node.x, new_node.y):
            continue
        if not line_collision_free(map_img, nearest_node.pt, new_node.pt):
            continue

        # --- 選擇最佳 parent ---
        n = len(nodes)
        radius = NEIGHBOR_COEFF * math.sqrt(max(math.log(n) / n, 1e-9))
        neighbors = near(nodes, new_node, radius) or [nearest_node]
        best_parent = nearest_node
        best_cost = best_parent.cost + distance(best_parent.pt, new_node.pt)

        for nb in neighbors:
            cand_cost = nb.cost + distance(nb.pt, new_node.pt)
            if cand_cost + 1e-6 < best_cost and line_collision_free(map_img, nb.pt, new_node.pt):
                best_parent = nb
                best_cost = cand_cost

        new_node.parent = best_parent
        new_node.cost = best_cost
        nodes.append(new_node)

        # --- Rewire ---
        for nb in neighbors:
            if nb is new_node or nb is best_parent:
                continue
            new_cost = new_node.cost + distance(new_node.pt, nb.pt)
            if new_cost + 1e-6 < nb.cost and line_collision_free(map_img, new_node.pt, nb.pt):
                nb.parent = new_node
                nb.cost = new_cost

        # --- 嘗試接通 Goal ---
        if distance(new_node.pt, goal) <= GOAL_REACH_THRESH:
            if line_collision_free(map_img, new_node.pt, goal):
                goal_node = Node(goal[0], goal[1], parent=new_node,
                                 cost=new_node.cost + distance(new_node.pt, goal))
                if goal_node.cost < c_best:
                    best_goal_node = goal_node
                    c_best = goal_node.cost

    # === 收尾 ===
    if best_goal_node is None:
        print("❌ 未找到可行路徑。")
        return None, nodes

    raw_path = extract_path(best_goal_node)
    smooth = smooth_path(map_img, raw_path, iterations=SMOOTH_ITER)
    return smooth, nodes


# ==========================================================
# 🔍 工具函式：找最近可行走點
# ==========================================================
def find_safe_goal_along_line(map_img, start, goal, step=1, safe_radius=5):
    """
    沿著 start→goal 連線尋找安全可行的 goal 替代點。
    條件：
      - 該點及其半徑 safe_radius 內皆為 free (255)
    參數：
      - map_img: 二值地圖 (0=障礙, 255=free)
      - start, goal: (x, y)
      - step: 沿線反向搜尋步長（像素）
      - safe_radius: 檢查半徑（像素）
    回傳：
      - (x, y): 最近的安全 goal 點；若找不到，回傳原始 goal
    """
    gx, gy = goal
    sx, sy = start
    H, W = map_img.shape[:2]

    # 計算連線方向（goal → start）
    dx, dy = sx - gx, sy - gy
    dist = math.hypot(dx, dy)
    if dist == 0:
        return goal
    dx, dy = dx / dist, dy / dist  # 單位向量

    def is_safe_point(x, y, r):
        """檢查該點 r 半徑內是否全為 free"""
        x, y = int(x), int(y)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H:
                    if map_img[ny, nx] < 128:
                        return False
                else:
                    return False
        return True

    # 反向搜尋：從 goal 往 start 方向走
    for t in np.arange(0, dist, step):
        x = gx + dx * t
        y = gy + dy * t
        if is_safe_point(x, y, safe_radius):
            return (int(x), int(y))

    print("⚠️ [警告] 找不到沿線安全 goal，回傳原始 goal。")
    return goal

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
# ======= 以下部份保留你原本的語意表/互動/視覺化 =======
# ==========================================================
def load_semantic_table(excel_path):
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
        nums = [int(v) for v in color_str.replace("(", "").replace(")", "").split(",") if v.strip().isdigit()]
        if len(nums) == 3:
            color_map[name] = tuple(nums)
    print(f"[INFO] 成功載入 {len(color_map)} 個語意分類。")
    return color_map

def load_semantic_ID_table(excel_path):
    df = pd.read_excel(excel_path)

    id_col = df.columns[0]  # 假設 ID 在第 0 欄
    name_col = None

    # 找出名稱欄
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
    bgr_color = tuple(reversed(color_map[target_class]))  # Excel 是 RGB，OpenCV BGR
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
        plt.plot(px, py, "r-", linewidth=2.0, label="Path (RRT* + smooth)")
    plt.legend()
    plt.axis("equal")
    plt.show()

# ==========================================================
# 主流程（與你原本一致，唯改用 RRT*）
# ==========================================================
if __name__ == "__main__":
    print("=== HW2 Part2: Semantic-guided RRT* Path Planning (w/ Informed + Smoothing) ===")

    # 讀地圖並二值化（白=free=255，黑=obs=0）
    map_img_gray = cv2.imread(MAP_PATH, cv2.IMREAD_GRAYSCALE)
    if map_img_gray is None:
        raise FileNotFoundError(MAP_PATH)
    _, binary_map = cv2.threshold(map_img_gray, 240, 255, cv2.THRESH_BINARY)

    # 加一道「安全膨脹」：侵蝕可行區（可選，用來避免鑽窄縫）
    # kernel = np.ones((11, 11), np.uint8)
    # binary_map = cv2.erode(binary_map, kernel, iterations=1)

    # Step 1. 載入語意表與地圖
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
    goal, mask = find_object_region(MAP_PATH, color_map, target_class)
    start = select_start(MAP_PATH, goal)

    # 執行 RRT*（介面與舊版相同）
    path, nodes = rrt_star_planning(binary_map, start, goal)

    # 顯示結果
    if path:
        visualize_rrt(binary_map, nodes, start, goal, path)
        print(f"[INFO] 路徑長度（像素）: {path_length_px(path):.1f} | 節點數: {len(nodes)}")
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
