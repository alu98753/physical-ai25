import cv2
import math
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from part2 import rrt_planning
from rrt_star import rrt_star_planning, path_length_px

# ==========================================================
# 通用設定
# ==========================================================
MAP_PATH = "/home/clu98753cs13/Desktop/course/phyai/physical-ai25/hw2/map.png"
SAVE_DIR = "/home/clu98753cs13/Desktop/course/phyai/physical-ai25/hw2/313554044_HW2/result/3_bonus"
os.makedirs(SAVE_DIR, exist_ok=True)

SAFE_WEIGHT = 500000
N_RUNS = 10

# ==========================================================
# 目標列表
# ==========================================================
START = (464, 167)
targets = {
    "BIKE": (324.0, 550.0),
    "WINDOW": (199.0, 404.0),
    "BLINDS": (187, 458),
    "CABINET": (491.0, 185.0),
    "CHAIR": (373.0, 496.0),
    "CUP": (306.0, 551.0),
    "COOKTOP": (306.0, 551.0)
}

# ==========================================================
# 輔助函式
# ==========================================================
def compute_path_cost(path):
    if not path or len(path) < 2:
        return float("inf")
    return sum(math.hypot(path[i+1][0]-path[i][0],
                          path[i+1][1]-path[i][1])
               for i in range(len(path)-1))

def safe_stats(arr):
    arr = np.array(arr)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return None, None, None, None, None
    return np.mean(arr), np.min(arr), np.max(arr), np.var(arr, ddof=1), np.std(arr, ddof=1)

def save_rrt(map_img, nodes, start, goal, path, title, save_dir=SAVE_DIR):
    """將 RRT or RRT* 結果畫圖存檔"""
    plt.figure(figsize=(8, 8))
    plt.imshow(map_img, cmap="gray")
    plt.plot(start[0], start[1], "go", markersize=8, label="Start")
    plt.plot(goal[0], goal[1], "ro", markersize=8, label="Goal")

    for node in nodes:
        if getattr(node, "parent", None) is not None:
            plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "b-", linewidth=0.4)

    if path:
        px, py = zip(*path)
        plt.plot(px, py, "r-", linewidth=2.0, label="Path")

    plt.legend()
    plt.axis("equal")
    plt.title(title)
    save_path = os.path.join(save_dir, f"{title.replace(' ', '_')}.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[IMG] Saved → {save_path}")

# ==========================================================
# 主程式
# ==========================================================
if __name__ == "__main__":
    print(f"=== Path Cost & Time Comparison (Average over {N_RUNS} runs per target) ===")

    map_img_gray = cv2.imread(MAP_PATH, cv2.IMREAD_GRAYSCALE)
    if map_img_gray is None:
        raise FileNotFoundError(MAP_PATH)
    _, binary_map = cv2.threshold(map_img_gray, 240, 255, cv2.THRESH_BINARY)

    output_path = os.path.join(SAVE_DIR, "path_comparison_results.txt")
    with open(output_path, "w") as f:
        f.write("=== Path Cost & Time Comparison (Average over 10 runs per target) ===\n\n")

        for target_name, GOAL in targets.items():
            print(f"\n🚩 Target: {target_name} ({GOAL})")
            f.write(f"\n=== Target: {target_name} ({GOAL}) ===\n")

            cost_rrt_list, cost_rrtstar_list = [], []
            time_rrt_list, time_rrtstar_list = [], []

            for run in range(1, N_RUNS + 1):
                print(f"  → Run {run}/{N_RUNS}")

                # --- RRT ---
                t0 = time.time()
                path_rrt, nodes_rrt = rrt_planning(binary_map, START, GOAL)
                t1 = time.time()
                elapsed_rrt = t1 - t0
                if path_rrt:
                    cost_rrt = compute_path_cost(path_rrt)
                    cost_rrt_list.append(cost_rrt)
                    if run == 1:  # 只存第一次圖
                        save_rrt(binary_map, nodes_rrt, START, GOAL, path_rrt,
                                title=f"{target_name}_RRT")
                else:
                    cost_rrt_list.append(float("inf"))
                time_rrt_list.append(elapsed_rrt)

                # --- RRT* ---
                t0 = time.time()
                path_rrtstar, nodes_rrtstar = rrt_star_planning(binary_map, START, GOAL, MIN_SAFE_DIST=0, SAFE_WEIGHT=SAFE_WEIGHT)
                t1 = time.time()
                elapsed_rrtstar = t1 - t0
                if path_rrtstar:
                    cost_rrtstar = path_length_px(path_rrtstar)
                    cost_rrtstar_list.append(cost_rrtstar)
                    if run == 1:  # 只存第一次圖
                        save_rrt(binary_map, nodes_rrtstar, START, GOAL, path_rrtstar,
                                title=f"{target_name}_RRTStar")
                else:
                    cost_rrtstar_list.append(float("inf"))
                time_rrtstar_list.append(elapsed_rrtstar)

            # 統計
            mean_rrt, min_rrt, max_rrt, var_rrt, std_rrt = safe_stats(cost_rrt_list)
            mean_rrtstar, min_rrtstar, max_rrtstar, var_rrtstar, std_rrtstar = safe_stats(cost_rrtstar_list)
            mean_t_rrt, min_t_rrt, max_t_rrt, _, std_t_rrt = safe_stats(time_rrt_list)
            mean_t_rrtstar, min_t_rrtstar, max_t_rrtstar, _, std_t_rrtstar = safe_stats(time_rrtstar_list)

            if mean_rrt is None or mean_rrtstar is None:
                summary = "⚠️ 無有效路徑。\n"
                print(summary)
                f.write(summary)
                continue

            improvement = (mean_rrt - mean_rrtstar) / mean_rrt * 100
            time_ratio = mean_t_rrtstar / mean_t_rrt if mean_t_rrt and mean_t_rrtstar else float("nan")

            # 印出與寫入
            summary = (
                f"[RRT]\n"
                f"  cost_avg = {mean_rrt:.2f}, min = {min_rrt:.2f}, max = {max_rrt:.2f}\n"
                f"  var = {var_rrt:.2f}, std = {std_rrt:.2f}\n"
                f"  time_avg = {mean_t_rrt:.2f}s, min = {min_t_rrt:.2f}s, max = {max_t_rrt:.2f}s, std = {std_t_rrt:.2f}s\n"
                f"[RRT*]\n"
                f"  cost_avg = {mean_rrtstar:.2f}, min = {min_rrtstar:.2f}, max = {max_rrtstar:.2f}\n"
                f"  var = {var_rrtstar:.2f}, std = {std_rrtstar:.2f}\n"
                f"  time_avg = {mean_t_rrtstar:.2f}s, min = {min_t_rrtstar:.2f}s, max = {max_t_rrtstar:.2f}s, std = {std_t_rrtstar:.2f}s\n"
                f"✅ 改善率: {improvement:.2f}% | ⏱️ 時間比 RRT*/RRT = {time_ratio:.2f}x\n"
            )

            print(summary)
            f.write(summary)

    print(f"\n📄 所有結果與圖片已輸出到：{SAVE_DIR}")