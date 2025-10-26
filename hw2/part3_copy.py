import os
import cv2
import math
import json
import numpy as np
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb,d3_40_colors_hex
from PIL import Image
import matplotlib.pyplot as plt
from part2 import rrt_star_planning, load_semantic_ID_table ,  load_semantic_table, find_object_region , rrt_star_planning

# ==========================================================
# Habitat 環境封裝
# ==========================================================

# ==========================================================
# 動作生成（安全版本）
# ==========================================================
FORWARD_STEP = 0.05
ARRIVAL_JUDGE = FORWARD_STEP*1.5
TURN_ANGLE = 1
ARRIVAL_THRESH = 0.15
MAX_ACTIONS = 5000
FPS = 120
class HabitatEnvWrapper:
    def __init__(self, sim_settings, floor=1):
        self.cfg = self.make_simple_cfg(sim_settings)
        self.sim = habitat_sim.Simulator(self.cfg)
        self.agent = self.sim.initialize_agent(sim_settings["default_agent"])

        # 初始化位置
        state = habitat_sim.AgentState()
        state.position = np.array([0.0, 0.0, 0.0]) if floor == 1 else np.array([0.0, 1.0, -1.0])
        self.agent.set_state(state)
        print("[DEBUG] action space keys:", list(self.cfg.agents[0].action_space.keys()))

    def make_simple_cfg(self, settings):
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = settings["scene"]

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=FORWARD_STEP)
            ),
            "move_backward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=-FORWARD_STEP)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=TURN_ANGLE)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=TURN_ANGLE)
            ),
        }



        def make_sensor(uuid, stype):
            spec = habitat_sim.CameraSensorSpec()
            spec.uuid = uuid
            spec.sensor_type = stype
            spec.resolution = [settings["height"], settings["width"]]
            spec.position = [0.0, settings["sensor_height"], 0.0]
            return spec

        agent_cfg.sensor_specifications = [
            make_sensor("color_sensor", habitat_sim.SensorType.COLOR),
            make_sensor("depth_sensor", habitat_sim.SensorType.DEPTH),
            make_sensor("semantic_sensor", habitat_sim.SensorType.SEMANTIC)
        ]
        return habitat_sim.Configuration(sim_cfg, [agent_cfg])

    def step(self, action: str):
        obs = self.sim.step(action)
        color_data = obs["color_sensor"] 
        if color_data.shape[2] == 4:
            rgb_img = color_data[:, :, :3]
        else:
            rgb_img = color_data

        return {
            "rgb": rgb_img, # 現在確保是 (H, W, 3)
            "depth": obs["depth_sensor"],
            "semantic": self._decode_semantic(obs["semantic_sensor"]),
        }

    @staticmethod
    def _decode_semantic(semantic_obs):
        img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        img.putpalette(d3_40_colors_rgb.flatten())
        img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        return np.asarray(img.convert("RGB"))

# util
def hex_to_rgb(hex_color):
    """支援 '#', '0x' 等格式的 hex → (R, G, B)"""
    hex_color = hex_color.strip().lower().replace('0x', '').replace('#', '')
    if len(hex_color) != 6:
        raise ValueError(f"❌ 無效的 hex 色碼: {hex_color}")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def target_mask(obs):
    semantic_img = obs["semantic"]  # RGB 格式
    target_id = id_map[TARGET_CLASS.lower()] % 40
    target_rgb = hex_to_rgb(d3_40_colors_hex[target_id])
    
    # print(f"[DEBUG] colormap (from excel): {color_map[TARGET_CLASS.lower()]}, used target_rgb: {target_rgb}")
    mask = cv2.inRange(semantic_img, target_rgb, target_rgb)
    # print(f"[DEBUG] mask pixels: {np.sum(mask > 0)}")  # 看有沒有非0像素

    cv2.imwrite("debug_semantic_rgb.png", cv2.cvtColor(semantic_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite("debug_mask.png", mask )

    return (mask > 0).astype(np.uint8)

def load_bounds(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["xmin"], data["xmax"], data["zmin"], data["zmax"]

def show_rrt_path(map_path, path_pixel, start, goal,count, title="RRT Path Preview"):
    """
    顯示 RRT 結果路徑圖，直到使用者關閉視窗後才繼續執行。
    """
    map_img = cv2.imread(map_path)
    if map_img is None:
        print(f"❌ 找不到地圖檔案: {map_path}")
        return

    map_img_rgb = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 8))
    plt.imshow(map_img_rgb)
    plt.title(title)
    plt.axis("off")

    # 畫出路徑線
    if path_pixel:
        xs, ys = zip(*path_pixel)
        plt.plot(xs, ys, color="red", linewidth=2, label="RRT Path")

    # 起點與終點
    plt.scatter(start[0], start[1], c="lime", s=60, label="Start", edgecolors="black")
    plt.scatter(goal[0], goal[1], c="cyan", s=60, label="Goal", edgecolors="black")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"./part3OUTPUT/rrt_show_round_{count}.png")

    plt.show(block=True)  # 等待視窗關閉後才繼續

def distance_to_path(current_pos, path):
    """回傳 agent 到當前路徑上最近點的距離"""
    if not path:
        return np.inf
    pxz = np.array([[px, pz] for px, pz in path])
    pos_xz = np.array([current_pos[0], current_pos[2]])
    dists = np.linalg.norm(pxz - pos_xz, axis=1)
    return np.min(dists)

def pixel_to_world(u, v, w, h, bounds):
    SCALE_FACTOR = 10000 / 255  # 
    # 假設 bounds 是點雲的 min/max (例如 20, 230, 15, 240)
    xmin_pt, xmax_pt, zmin_pt, zmax_pt = bounds 

    # 步驟 1: Pixel (u, v) 轉換到 Point Cloud 座標 (x_pt, z_pt)
    # (標準線性內插)
    x_pt = xmin_pt + (u / w) * (xmax_pt - xmin_pt)
    
    # (1.0 - v/h) 是因為 pixel 的 v 軸 (向下) 和 Habitat 的 z 軸 (通常向上或向前) 是反的
    z_pt = zmin_pt + (1.0 - (v / h)) * (zmax_pt - zmin_pt)

    # 步驟 2: Point Cloud 座標 (x_pt, z_pt) 轉換到 Habitat 世界座標
    # 根據 spec  套用縮放
    x_world = x_pt * SCALE_FACTOR
    z_world = z_pt * SCALE_FACTOR

    return float(x_world), float(z_world)

def world_to_pixel(x, z, w, h, bounds):
    SCALE_FACTOR = 10000 / 255  # 
    xmin_pt, xmax_pt, zmin_pt, zmax_pt = bounds

    # 步驟 1: World 座標 -> Point Cloud 座標
    x_pt = x / SCALE_FACTOR
    z_pt = z / SCALE_FACTOR
    
    # 步驟 2: Point Cloud 座標 -> Pixel 座標 (u, v)
    # (反向線性內插)
    u_ratio = (x_pt - xmin_pt) / (xmax_pt - xmin_pt)
    v_ratio = 1.0 - (z_pt - zmin_pt) / (zmax_pt - zmin_pt) # 反轉 v 軸

    u = u_ratio * w
    v = v_ratio * h

    return u,v

def calculate_path_distance(p1, p2):
    """Calculates Euclidean distance between two (x, z) points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def segment_world_path(world_path, segment_length=2.0):
    """
    Segments a world path into milestones approximately segment_length apart.

    Args:
        world_path: List of (x, z) world coordinates.
        segment_length: Desired distance between milestones (in meters).

    Returns:
        List of milestone (x, z) coordinates, including start and end.
    """
    if not world_path or len(world_path) < 2:
        return world_path # Return original if too short

    milestones = [world_path[0]]
    cumulative_dist_since_last_milestone = 0.0

    for i in range(1, len(world_path)):
        dist_segment = calculate_path_distance(world_path[i], world_path[i-1])
        cumulative_dist_since_last_milestone += dist_segment

        # If we have traveled far enough along the path for a new milestone
        if cumulative_dist_since_last_milestone >= segment_length:
            milestones.append(world_path[i])
            cumulative_dist_since_last_milestone = 0.0 # Reset distance counter

    # Always include the final goal if it wasn't the last milestone added
    if calculate_path_distance(milestones[-1], world_path[-1]) > 1e-3: # Check if last points are different
        milestones.append(world_path[-1])

    print(f"[INFO] Path segmented into {len(milestones)} milestones (approx. every {segment_length}m).")
    # Print first few milestones for debugging
    for i, m in enumerate(milestones[:min(5, len(milestones))]):
         print(f"  Milestone {i}: ({m[0]:.2f}, {m[1]:.2f})")
    if len(milestones) > 5:
         print("  ...")
         m = milestones[-1]
         print(f"  Milestone {len(milestones)-1}: ({m[0]:.2f}, {m[1]:.2f})")

    return milestones
def wrap_to_pi(a): return (a + math.pi) % (2 * math.pi) - math.pi

def generate_actions_from_world_path(world_path_xz,
                                    current_yaw_rad = 0.0,
                                    forward_step_m=FORWARD_STEP,
                                    turn_step_deg=TURN_ANGLE,
                                    arrival_thresh_m=ARRIVAL_THRESH):
    actions = []
    sim_x, sim_z = world_path_xz[0]
    yaw = current_yaw_rad

    def turn_to(desired_yaw):
        nonlocal yaw
        step = math.radians(turn_step_deg)
        d = wrap_to_pi(desired_yaw - yaw)
        angle_deg = math.degrees(abs(d))

        # 若角度太小，就不轉（防止左右切換）
        if angle_deg < 2:
            return

        # 大角度先快速轉，小角度平滑
        if angle_deg > 15:
            step_mult = 1.0
        elif angle_deg > 5:
            step_mult = 0.5
        else:
            step_mult = 0.25

        if d > 0:
            actions.append("turn_left")
            yaw += step * step_mult
        else:
            actions.append("turn_right")
            yaw -= step * step_mult



    def forward_to(tx, tz):
        nonlocal sim_x, sim_z
        dx, dz = tx - sim_x, tz - sim_z
        dist = math.hypot(dx, dz)
        # print("dist:",dist)
        if not np.isfinite(dist) or dist > 10:
            print(f"⚠️ [WARN] Invalid dist={dist:.3f}, skipping")
            return
        n = max(1, int((dist - arrival_thresh_m) / forward_step_m))
        n = min(n, 50)  # 安全上限
        actions.extend(["move_forward"] * n)
        sim_x, sim_z = tx, tz
        # print(f"[move] from ({sim_x:.2f},{sim_z:.2f}) → ({tx:.2f},{tz:.2f}), dist={dist:.2f}, yaw={math.degrees(desired_yaw):.1f}")

    for k in range(1, len(world_path_xz)):
        tx, tz = world_path_xz[k]
        dx, dz = tx - sim_x, tz - sim_z
        dist = math.hypot(dx, dz)
        desired_yaw = math.atan2(-dx, -dz)  # 注意這裡反轉

        # print(f"Segment {k}: dist={dist:.4f}, yaw_diff={math.degrees(wrap_to_pi(desired_yaw - yaw)):.2f}")


        turn_to(desired_yaw)
        forward_to(tx, tz)
        if len(actions) > MAX_ACTIONS:
            print("⚠️ [WARN] Too many actions generated, truncating.")
            break

    # print(f"[INFO] Total generated actions: {len(actions)}")
    # print(f"[INFO] first 5 actions:{actions[:5]}")
    return actions

# ==========================================================
# 影片錄製
# ==========================================================
def overlay_mask(rgb, mask, color=(255, 0, 0), alpha=0.15):
    """
    將 target mask 疊加到 RGB 影像上（半透明顏色）
    color: RGB 顏色，例如 (255,0,0) 為紅色
    alpha: 疊加透明度 (0~1)
    """
    # === Step 1. 確保尺寸一致 ===
    if rgb.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]))

    # === Step 2. 確保 mask 為二值 0/1 ===
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8)
    else:
        mask = (mask > 0).astype(np.uint8)

    # === Step 3. 建立彩色遮罩圖層 ===
    color_mask = np.zeros_like(rgb, dtype=np.uint8)
    color_mask[mask > 0] = color  # 只在 mask 區域上上色

    # === Step 4. 混合圖像 ===
    dst = cv2.addWeighted(color_mask, alpha, rgb, 1 - alpha, 0)

    return dst

def face_goal(env, goal, bounds, w, h, vw, target_mask, FPS=120, TURN_ANGLE=1):
    """
    當抵達目標後，慢慢旋轉面向指定 goal pixel 的世界座標，
    並在結尾顯示目標 overlay 與「Reached Goal」畫面。
    """
    pos = env.agent.get_state().position.copy()
    print(f"[SUCCESS] Reached goal ✅ ({pos[0]:.2f}, {pos[2]:.2f})")

    # === 對準指定目標 pixel 的世界座標 ===
    gx, gz = pixel_to_world(goal[0], goal[1], w, h, bounds)
    dx, dz = gx - pos[0], gz - pos[2]
    desired_yaw = math.atan2(-dx, -dz)

    # 取目前 agent 的 yaw
    q = env.agent.get_state().rotation
    current_yaw = 2 * math.atan2(q.imag[1], q.real)

    # 計算角度差
    yaw_diff = wrap_to_pi(desired_yaw - current_yaw)
    print(f"[TURN] current_yaw={math.degrees(current_yaw):.1f}°, desired_yaw={math.degrees(desired_yaw):.1f}°, diff={math.degrees(yaw_diff):.1f}°")

    # 左為負 → turn_left
    turn_action = "turn_left" if yaw_diff < 0 else "turn_right"
    turn_steps = int(abs(math.degrees(yaw_diff)) // TURN_ANGLE)
    print(f"[INFO] Turning {turn_steps} steps to face goal...")

    if turn_steps > 0:
        # === 有需要旋轉時 ===
        for i in range(turn_steps):
            obs = env.step(turn_action)
            rgb = obs["rgb"]
            mask = target_mask(obs)
            vis = overlay_mask(rgb, mask)
            # 控制轉動時每步的錄影幀數
            for _ in range(FPS * 2 // max(turn_steps, 1)):
                vw.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    else:
        # === 不需要旋轉時（保持目前畫面） ===
        obs = env.step("turn_left")  # 小動作確保拿到新畫面
        rgb = obs["rgb"]
        mask = target_mask(obs)
        vis = overlay_mask(rgb, mask)
        for _ in range(FPS // 10):
            vw.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    # 再確認最終朝向
    q2 = env.agent.get_state().rotation
    final_yaw = 2 * math.atan2(q2.imag[1], q2.real)
    print(f"[DONE] Final yaw aligned: {math.degrees(final_yaw):.1f}°")

    # === 結尾畫面：保持面向目標並塗色 ===
    rgb = obs["rgb"].copy()
    mask = target_mask(obs)
    vis = overlay_mask(rgb, mask)
    vis = end_anime(vis)
    for _ in range(FPS * 3):
        vw.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    vw.release()
    print(f"[END] Navigation finished with facing target.")

def end_anime(rgb):
    # === 結尾畫面 (居中文字 + 停留) ===
    text = "Reached Goal"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (rgb.shape[1] - text_size[0]) // 2
    text_y = (rgb.shape[0] + text_size[1]) // 2

    vis = rgb.copy()
    cv2.putText(vis, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
    return vis
# ==========================================================
# 導航
# ==========================================================

def align_to_first_waypoint(env, start_pixel, goal, safe_binary_map, bounds, vw, target_mask, w, h, FPS=120, TURN_ANGLE=1, BACKWARD_DIST=0.75):
    """
    Replan 後的動作調整：
    回傳新的 world_path 給外層使用
    """
    backward_steps = max(1, int(BACKWARD_DIST / FORWARD_STEP))
    print(f"[BACKWARD] 後退 {backward_steps} 步以避免貼牆旋轉")
    for _ in range(backward_steps):
        obs = env.step("move_backward")
        rgb = obs["rgb"]; mask = target_mask(obs)
        vis = overlay_mask(rgb, mask)
        for _ in range(FPS // 20):
            vw.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    # === 重新規劃 ===
    path_pixel, _ = rrt_star_planning(safe_binary_map, start_pixel, goal)
    counter = 0
    while path_pixel is None:
        print(f"❌ [REPLAN FAIL] 無法找到可行路徑，again。{start_pixel}")
        path_pixel, _ = rrt_star_planning(safe_binary_map, start_pixel, goal)
        counter += 1
        if path_pixel or counter >= 10:
            break

    if not path_pixel:
        print("[FAIL] 無法重新規劃出路徑")
        return None

    # === 轉換成世界座標 ===
    world_path = [pixel_to_world(u, v, w, h, bounds) for (u, v) in path_pixel]

    # === 角度校正 ===
    pos = env.agent.get_state().position.copy()
    q = env.agent.get_state().rotation
    current_yaw = 2 * math.atan2(q.imag[1], q.real)

    dx = world_path[1][0] - pos[0]
    dz = world_path[1][1] - pos[2]
    desired_yaw = math.atan2(-dx, -dz)
    yaw_diff = wrap_to_pi(desired_yaw - current_yaw)
    angle_diff_deg = abs(math.degrees(yaw_diff))

    print(f"[ALIGN] 當前 yaw={math.degrees(current_yaw):.1f}°, 目標 yaw={math.degrees(desired_yaw):.1f}°, 差={angle_diff_deg:.1f}°")

    if angle_diff_deg > 5:
        turn_action = "turn_left" if yaw_diff < 0 else "turn_right"
        turn_steps = int(angle_diff_deg // TURN_ANGLE)
        print(f"[TURN] 對準新路徑方向 ({turn_steps} 步)")
        for _ in range(turn_steps):
            obs = env.step(turn_action)
            rgb = obs["rgb"]; mask = target_mask(obs)
            vis = overlay_mask(rgb, mask)
            for _ in range(FPS // 30):
                vw.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    print("[ALIGN DONE] ✅ 已完成後退 + 對準 + 路徑更新")
    return world_path,desired_yaw

# ==========================================================
# 新增：分而治之 - 創建子目標點
# ==========================================================
def create_subgoals(world_path, segment_distance_m=2.0):
    """
    從完整的 world_path 中，每隔約 segment_distance_m 選取一個點作為子目標。
    """
    if not world_path or len(world_path) < 2:
        return world_path # 如果路徑很短，直接返回

    subgoals = [world_path[0]] # 包含起點
    accumulated_dist = 0.0
    
    for i in range(len(world_path) - 1):
        p1 = world_path[i]
        p2 = world_path[i+1]
        segment_dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        accumulated_dist += segment_dist

        # 如果累積距離超過閾值，且不是最後一段
        if accumulated_dist >= segment_distance_m and i < len(world_path) - 2:
            subgoals.append(p2)
            accumulated_dist = 0.0 # 重置累積距離

    # 確保最終目標點一定在列表中
    if subgoals[-1] != world_path[-1]:
        subgoals.append(world_path[-1])
        
    print(f"[INFO] 分而治之：從 {len(world_path)} 個路徑點中，生成 {len(subgoals)} 個子目標點。")
    # 打印子目標點以供調試
    for idx, sg in enumerate(subgoals):
        print(f"  Subgoal {idx}: ({sg[0]:.2f}, {sg[1]:.2f})")
        
    return subgoals

# ==========================================================
# 修正後的 correct_course
# ==========================================================
def correct_course(env, current_segment_world_path, bounds, w, h, vw, target_mask,
                   progress_idx=0, max_angle_diff=30.0):
    """
    修正版 v3：Pure Pursuit + 角度與距離篩選 + 無 fallback + 無前進。
    1.  只執行轉彎，不執行前進。
    2.  選擇目標點：
        - 從 progress_idx 開始搜尋。
        - 必須在前方視野內 (角度差 < max_angle_diff)。
        - 選擇符合角度條件下，距離機器人最近的點作為修正目標。
    3.  如果找不到有效目標點，則不執行任何修正。
    4.  返回計算出的最近點的全域索引 (nearest_idx)。
    """
    # 確保不會發生 NameError
    nearest_idx = progress_idx

    state = env.agent.get_state()
    pos = state.position.copy()
    q = state.rotation
    current_yaw = 2 * math.atan2(q.imag[1], q.real)
    print(f"[INFO] correct_course: current_yaw={math.degrees(current_yaw):.1f}°") # 保留 Print

    # === Step 1: 從進度點開始挑選候選路徑點 ===
    # 使用 current_segment_world_path 進行校正
    candidate_path = current_segment_world_path[progress_idx:]
    if not candidate_path:
        print("[WARN] correct_course: candidate_path is empty.") # 保留 Print
        # 如果候選路徑為空，返回最後索引和 0 距離
        last_idx = len(current_segment_world_path) - 1 if current_segment_world_path else 0
        return last_idx, 0.0

    # === Step 2: 篩選有效目標點 ===
    valid_targets = [] # 儲存 (全域索引 in segment, 距離, 點座標)
    max_angle_diff_rad = math.radians(max_angle_diff)

    # 計算實際最近點的距離 (基於 segment)
    dists_all = [math.hypot(px - pos[0], pz - pos[2]) for (px, pz) in candidate_path]
    actual_nearest_local_idx = int(np.argmin(dists_all))
    # important: nearest_idx is relative to the *current_segment_world_path*
    actual_nearest_idx_in_segment = progress_idx + actual_nearest_local_idx
    actual_dist_to_path = dists_all[actual_nearest_local_idx]


    for local_idx, (px, pz) in enumerate(candidate_path):
        current_idx_in_segment = progress_idx + local_idx # 當前點在 segment 中的索引
        dx, dz = px - pos[0], pz - pos[2]
        dist = math.hypot(dx, dz)

        # 忽略太近的點
        if dist < FORWARD_STEP * 3: # 放寬一點，避免選到自己腳下
            continue

        # 計算指向該點的理想 Yaw
        desired_yaw_to_pt = math.atan2(dx, -dz) # 標準 Habitat Yaw
        yaw_diff_rad = abs(wrap_to_pi(desired_yaw_to_pt - current_yaw))

        # 篩選：角度必須在容忍範圍內
        if yaw_diff_rad <= max_angle_diff_rad:
            valid_targets.append((current_idx_in_segment, dist, (px, pz)))

    # === Step 3: 選擇最佳修正目標點 ===
    target_for_correction = None
    target_global_idx_in_segment = actual_nearest_idx_in_segment # 預設

    if valid_targets:
        # 在所有有效點中，選擇距離最近的那個點作為修正目標
        best_target_info = min(valid_targets, key=lambda item: item[1])
        target_global_idx_in_segment, target_dist, target_for_correction = best_target_info
        print(f"[CORRECT] 找到有效目標點 (idx_in_segment={target_global_idx_in_segment}, dist={target_dist:.2f}m) 符合角度 < {max_angle_diff}°") # 保留 Print
    else:
        # 如果前方 30 度內沒有任何路徑點，**則不進行修正**
        print(f"[CORRECT] 前方 {max_angle_diff}° 無有效點，本次不進行角度修正") # 保留 Print
        # target_for_correction 保持為 None


    # === Step 4: 角度修正 (只轉彎) ===
    if target_for_correction:
        dx, dz = target_for_correction[0] - pos[0], target_for_correction[1] - pos[2]
        desired_yaw = math.atan2(dx, -dz)
        yaw_diff = wrap_to_pi(desired_yaw - current_yaw)
        angle_diff_deg = abs(math.degrees(yaw_diff))

        # 僅在偏離角度足夠大時才轉彎
        if angle_diff_deg > 5: # 最小轉角閾值
            print(f"[CORRECT] 修正角度 {angle_diff_deg:.1f}° → 向 ({target_for_correction[0]:.2f}, {target_for_correction[1]:.2f})") # 保留 Print

            turn_action = "turn_left" if yaw_diff < 0 else "turn_right"
            # 執行固定的小幅度轉彎 (例如 1 步)
            turn_steps_to_exec = 1

            for _ in range(turn_steps_to_exec):
                obs = env.step(turn_action) # 執行轉彎
                # --- 轉彎時也需要更新狀態和錄影 ---
                pos = env.agent.get_state().position.copy() # 更新 pos
                q = env.agent.get_state().rotation
                current_yaw = 2 * math.atan2(q.imag[1], q.real) # 更新 yaw
                # --- 錄影 ---
                rgb = obs["rgb"]; mask = target_mask(obs)
                vis = overlay_mask(rgb, mask)
                vw.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)) # 寫入一幀
        else:
            # 角度夠準，pass
            print(f"[CORRECT] 角度偏差 {angle_diff_deg:.1f}° (<5°)，無需修正") # 保留 Print
            pass
    # else: (找不到目標點時不打印)
    #    print("[WARN] correct_course: 未找到修正目標點")


    # ⚠️ 確保移除了所有 move_forward 動作！

    # 返回：當前離機器人最近點在 *當前路徑段* 中的索引，以及該距離
    # 注意：返回的是 actual_nearest_idx_in_segment
    return actual_nearest_idx_in_segment, actual_dist_to_path


# ==========================================================
# 修正後的 run_navigation_replan
# ==========================================================
def run_navigation_replan(env, binary_map, safe_binary_map, color_map, bounds, start_pixel_orig, goal_pixel_orig, target_mask,
                          output_video="result_replan.mp4", replan_thresh=0.3, segment_distance_m=2.0,
                          # --- 新增避障參數 ---
                          proactive_avoidance_threshold=5000, # 觸發主動避障的像素閾值
                          escape_backward_dist=0.5,           # 後退距離 (米)
                          escape_turn_angle=45):              # 轉向角度 (度)
    """
    主導航流程（分而治之 v3 - 逐段規劃 + 主動避障/脫困）：
      - 依序導航至一系列子目標點。
      - 每次前往下一個子目標前，規劃該路徑段。
      - 加入主動避障和脫困邏輯。
      - 整合修正後的 correct_course()。
    """
    vw = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (512, 512))
    frame_count = 0
    h, w = binary_map.shape

    # === 初始化 ===
    current_state = env.agent.get_state()
    current_pos = current_state.position.copy() # 初始 agent 位置 (世界座標)

    # === 預先規劃完整路徑以生成子目標 ===
    print("[INFO] 正在進行初始完整路徑規劃 (僅用於生成子目標)...") # 保留 Print
    initial_path_pixel, _ = rrt_star_planning(safe_binary_map, start_pixel_orig, goal_pixel_orig)
    if initial_path_pixel is None:
        print("❌ 初始 RRT* 規劃失敗 (無法生成子目標)，無法開始導航。") # 保留 Print
        vw.release()
        return

    initial_world_path = [pixel_to_world(u, v, w, h, bounds) for (u, v) in initial_path_pixel]
    if not initial_world_path:
        print("❌ 初始世界路徑為空，無法開始導航。") # 保留 Print
        vw.release()
        return

    # === 分而治之：創建子目標點 ===
    subgoals = create_subgoals(initial_world_path, segment_distance_m)
    subgoals = subgoals[1:]
    if len(subgoals) <= 1:
        print("[WARN] 子目標點列表過短，將直接導航至最終目標。") # 保留 Print
        if not subgoals: subgoals = [initial_world_path[-1]]


    # === 主迴圈：依序導航至子目標 ===
    current_subgoal_index = 0
    global_replan_count = 0

    try:
        while current_subgoal_index < len(subgoals):

            # --- 設定當前子目標 ---
            current_target_subgoal_world = subgoals[current_subgoal_index]
            current_target_subgoal_pixel = world_to_pixel(current_target_subgoal_world[0], current_target_subgoal_world[1], w, h, bounds)

            print(f"\n========== [導航至子目標 {current_subgoal_index}/{len(subgoals)-1}] ==========") # 保留 Print
            print(f"目標世界座標: ({current_target_subgoal_world[0]:.2f}, {current_target_subgoal_world[1]:.2f})") # 保留 Print

            # --- 獲取當前狀態 ---
            current_state = env.agent.get_state()
            current_pos = current_state.position.copy()
            current_pixel = world_to_pixel(current_pos[0], current_pos[2], w, h, bounds)
            q = current_state.rotation
            current_yaw = 2 * math.atan2(q.imag[1], q.real)

            # --- ✅ 核心修改：規劃 *當前路徑段* ---
            print(f"[INFO] 規劃路徑段：從 ({current_pos[0]:.2f}, {current_pos[2]:.2f}) -> 子目標 {current_subgoal_index}") # 保留 Print
            path_pixel_segment, _ = rrt_star_planning(safe_binary_map, current_pixel, current_target_subgoal_pixel)
            retry_segment = 0
            while path_pixel_segment is None and retry_segment < 3:
                 print(f"❌ [WARN] 無法規劃到子目標 {current_subgoal_index} 的路徑段，重試 ({retry_segment+1}/3)...") # 保留 Print
                 path_pixel_segment, _ = rrt_star_planning(safe_binary_map, current_pixel, current_target_subgoal_pixel)
                 retry_segment += 1

            if path_pixel_segment is None:
                 print(f"[FAIL] 連續多次無法規劃到子目標 {current_subgoal_index} 的路徑，導航終止。") # 保留 Print
                 return

            current_segment_world_path = [pixel_to_world(u, v, w, h, bounds) for (u, v) in path_pixel_segment]
            if not current_segment_world_path:
                 print(f"[FAIL] 規劃出的路徑段為空 (子目標 {current_subgoal_index})，導航終止。") # 保留 Print
                 return

            # --- 生成 *當前路徑段* 的動作 ---
            actions = generate_actions_from_world_path(current_segment_world_path, current_yaw_rad=current_yaw)
            print(f"[INFO] 為子目標 {current_subgoal_index} 生成 {len(actions)} 個動作 (路徑段長度 {len(current_segment_world_path)} 點)") # 保留 Print

            # --- 內層迴圈：執行動作 ---
            subgoal_reached = False
            local_replan_count_this_segment = 0
            progress_idx = 0 # ✅ 重置 *當前路徑段* 的進度索引
            stuck_counter_val = 0 # 重置卡住計數器

            # --- 獲取進入內層迴圈前的觀測 ---
            # 我們需要上一步的 obs 來做主動避障判斷
            # 可以在規劃完路徑後，執行一個空動作或小轉彎來獲取
            try:
                obs = env.step("turn_right") # 獲取初始觀測
                current_pos = env.agent.get_state().position.copy() # 更新一下位置
            except Exception as e_init_obs:
                 print(f"❌ [ERROR] 無法獲取初始觀測: {e_init_obs}")
                 return


            while not subgoal_reached:

                print(f"--- [子目標 {current_subgoal_index} 導航中 | Segment Replan: {local_replan_count_this_segment}] ---") # 保留 Print

                for idx, act in enumerate(actions):

                    # ========== ✅ NEW: 主動避障/脫困 ==========
                    depth = obs["depth"]
                    near_mask = (depth < 0.45)
                    near_pixels = np.sum(near_mask)
                    print(f"[DEPTH] near<0.45m: {near_pixels}, pos=({current_pos[0]:.2f},{current_pos[2]:.2f}), yaw={math.degrees(current_yaw):.1f}°") # 保留 Print (使用 current_pos)

                    if near_pixels > proactive_avoidance_threshold and act == "move_forward":
                        print(f"[AVOID] 前方障礙觸發主動避障 (near={near_pixels} > {proactive_avoidance_threshold})") # 保留 Print

                        # --- 判斷左右逃脫方向 ---
                        h_depth, w_depth = depth.shape
                        left_slice = depth[:, :w_depth//3] # 左側 1/3
                        right_slice = depth[:, w_depth*2//3:] # 右側 1/3
                        left_near_pixels = np.sum(left_slice < 0.6) # 稍微放寬檢查距離
                        right_near_pixels = np.sum(right_slice < 0.6)
                        escape_action = "turn_right" # 預設向右轉
                        if left_near_pixels < right_near_pixels:
                            escape_action = "turn_left" # 如果左邊比較開闊，向左轉
                        print(f"[AVOID] 左側 near (<0.6m): {left_near_pixels}, 右側 near (<0.6m): {right_near_pixels}. 選擇逃脫方向: {escape_action}") # 保留 Print

                        # --- 執行脫困：後退 + 轉向 ---
                        print(f"[AVOID] 執行脫困：後退 {escape_backward_dist}m") # 保留 Print
                        backward_steps = max(1, int(escape_backward_dist / FORWARD_STEP))
                        for _ in range(backward_steps):
                            obs = env.step("move_backward")
                            # 錄影
                            rgb = obs["rgb"]; mask = target_mask(obs); vis = overlay_mask(rgb, mask)
                            vw.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                            frame_count += 1

                        print(f"[AVOID] 執行脫困：轉向 {escape_action} {escape_turn_angle}°") # 保留 Print
                        turn_steps = int(escape_turn_angle // TURN_ANGLE)
                        for _ in range(turn_steps):
                            obs = env.step(escape_action)
                            # 錄影
                            rgb = obs["rgb"]; mask = target_mask(obs); vis = overlay_mask(rgb, mask)
                            vw.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                            frame_count += 1

                        # --- 脫困後強制 Replan ---
                        print("[AVOID] 脫困完成，強制重新規劃...") # 保留 Print
                        current_pos = env.agent.get_state().position.copy() # 更新位置
                        stuck_counter_val = 0 # 重置卡住計數 (因為我們是主動避開)
                        break # 跳出內層 for 迴圈，觸發 Replan
                    # ========== END: 主動避障/脫困 ==========


                    # --- 角度校正 (如果不是因避障跳出) ---
                    if act == "move_forward":
                        # ✅ correct_course 使用 current_segment_world_path
                        progress_idx, current_dist_to_path = correct_course(env, current_segment_world_path, bounds, w, h, vw, target_mask,
                                progress_idx=progress_idx, max_angle_diff=30.0)
                        # print(f"[DEBUG] correct_course 返回 segment progress_idx={progress_idx}, dist_to_path={current_dist_to_path:.3f}m") # 保留 Print
                    else:
                        current_dist_to_path = distance_to_path(current_pos, current_segment_world_path)


                    # --- 執行動作 (如果不是因避障跳出) ---
                    obs = env.step(act)

                    # --- 更新狀態和計算距離 ---
                    pos = env.agent.get_state().position.copy()
                    q = env.agent.get_state().rotation
                    yaw = math.degrees(2 * math.atan2(q.imag[1], q.real))
                    print(f"[TRACE] {act}, pos=({pos[0]:.2f},{pos[2]:.2f}), yaw={yaw:.1f}°") # 保留 Print

                    dist_to_current_subgoal = math.hypot(pos[0] - current_target_subgoal_world[0], pos[2] - current_target_subgoal_world[1])
                    dist_to_final_goal = np.linalg.norm(pos[[0, 2]] - np.array(subgoals[-1]))

                    # 保留 Debug 輸出
                    print(f"[DEBUG] pos=({pos[0]:.2f},{pos[2]:.2f}), "
                          f"subgoal=({current_target_subgoal_world[0]:.2f},{current_target_subgoal_world[1]:.2f}), "
                          f"final_goal=({subgoals[-1][0]:.2f},{subgoals[-1][1]:.2f}), "
                          f"dist_to_subgoal={dist_to_current_subgoal:.3f}, dist_to_final_goal={dist_to_final_goal:.3f}, dist_to_path={current_dist_to_path:.3f}") # 保留 Print

                    # ======= 事件偵測 =======
                    # 1. 到達 *當前子目標*
                    if dist_to_current_subgoal < ARRIVAL_JUDGE * 4:
                        print(f"[INFO] 到達子目標 {current_subgoal_index} ({current_target_subgoal_world[0]:.2f}, {current_target_subgoal_world[1]:.2f})") # 保留 Print
                        subgoal_reached = True
                        current_pos = pos.copy()
                        break # 跳出內層 for 迴圈

                    # 2. 嚴重偏離 -> 觸發 Replan
                    if current_dist_to_path > replan_thresh:
                        print(f"[REPLAN] Deviated from segment path ({current_dist_to_path:.2f} m > {replan_thresh}m) → 重新規劃 (目標: 子目標 {current_subgoal_index})") # 保留 Print
                        current_pos = pos.copy()
                        break # 跳出內層 for 迴圈

                    # 3. 卡住 -> 觸發 Replan
                    move_dist = np.linalg.norm(pos - current_pos)
                    if move_dist < 0.01 and act == "move_forward":
                        stuck_counter_val += 1
                    else:
                        stuck_counter_val = 0

                    if stuck_counter_val > 20:
                        print(f"[REPLAN] Agent stuck (連續 {stuck_counter_val} 次 move < 0.01m) → 重新規劃 (目標: 子目標 {current_subgoal_index})") # 保留 Print
                        current_pos = pos.copy()
                        # stuck_counter_val 在 Replan 後會重置
                        break # 跳出內層 for 迴圈

                    current_pos = pos.copy() # 更新位置

                    # === 繪製畫面 ===
                    rgb = obs["rgb"]
                    mask = target_mask(obs)
                    vis = overlay_mask(rgb, mask)
                    for sub in range(FPS//15):
                        vw.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                        frame_count += 1
                    # if frame_count % 40 == 0:
                    #     print(f"[VIDEO] 已寫入 {frame_count} 幀") # 保留 Print (降低頻率)

                # --- 內層 for 迴圈結束處理 ---
                if subgoal_reached:
                    break # 跳出內層 while 迴圈

                elif not subgoal_reached:
                    if idx == len(actions) - 1 and len(actions)>0: # 檢查是否是 actions 耗盡 (且 actions 不是空的)
                        print("[WARN] Action list exhausted for segment, but subgoal not reached. Forcing replan.") # 保留 Print
                    # else: (是因為 break 跳出，Replan 訊息已打印)

                    current_pos = env.agent.get_state().position.copy()

                # ======= 重新規劃 (Re-plan - 目標是當前子目標) =======
                print(f"--- [觸發重新規劃 - 目標: 子目標 {current_subgoal_index}] ---") # 保留 Print
                global_replan_count += 1
                local_replan_count_this_segment += 1
                if local_replan_count_this_segment > 10:
                    print(f"[FAIL] 在導航至子目標 {current_subgoal_index} 時重新規劃次數過多，放棄。") # 保留 Print
                    return

                start_pixel = world_to_pixel(current_pos[0], current_pos[2], w, h, bounds)
                # is_stuck_replan 判斷移到下面，因為避障也會 break
                is_stuck_replan = stuck_counter_val > 0 # stuck_counter_val 在 break 前不會重置

                new_segment_world_path = None
                desired_yaw = 0

                if is_stuck_replan: # 因卡住而 Replan
                    print("[REPLAN] Stuck case: Performing align_to_first_waypoint...") # 保留 Print
                    try:
                        align_result = align_to_first_waypoint(
                            env, start_pixel, current_target_subgoal_pixel,
                            safe_binary_map, bounds, vw, target_mask, w, h, FPS, TURN_ANGLE
                        )
                        if align_result is None:
                            print("[FAIL] align_to_first_waypoint failed during replan.") # 保留 Print
                            return
                        new_segment_world_path, desired_yaw = align_result
                    except Exception as e:
                        print(f"❌ [ERROR] align_to_first_waypoint failed: {e}") # 保留 Print
                        raise
                else: # 因偏離、避障或 action 耗盡而 Replan
                    print("[REPLAN] Deviation/Avoidance/Exhausted case: Recalculating RRT* path for segment...") # 保留 Print
                    path_pixel_segment, _ = rrt_star_planning(safe_binary_map, start_pixel, current_target_subgoal_pixel)
                    retry = 0
                    while path_pixel_segment is None and retry < 5:
                        print(f"❌ [REPLAN FAIL] RRT* failed for segment, retrying... ({retry+1}/5)") # 保留 Print
                        path_pixel_segment, _ = rrt_star_planning(safe_binary_map, start_pixel, current_target_subgoal_pixel)
                        retry += 1
                    if not path_pixel_segment:
                        print("[FAIL] RRT* failed for segment after retries.") # 保留 Print
                        return

                    new_segment_world_path = [pixel_to_world(u, v, w, h, bounds) for (u, v) in path_pixel_segment]
                    q_replan = env.agent.get_state().rotation
                    desired_yaw = 2 * math.atan2(q_replan.imag[1], q_replan.real)

                # --- 更新 *當前路徑段* 和動作 ---
                if new_segment_world_path:
                    show_rrt_path(MAP_PATH, path_pixel_segment, start, goal,local_replan_count_this_segment, title="RRT Path Preview")
                    current_segment_world_path = new_segment_world_path
                    actions = generate_actions_from_world_path(current_segment_world_path, current_yaw_rad=desired_yaw)
                    progress_idx = 0 # ✅ 重置 *路徑段* 的進度索引
                    stuck_counter_val = 0 # ✅ 重置卡住計數器

                    print(f"[INFO] Replan successful. New segment path has {len(current_segment_world_path)} points. New action list has {len(actions)} steps.") # 保留 Print
                    # 繼續內層 while 迴圈
                else:
                    print("[FAIL] Replan failed to generate a new path segment.") # 保留 Print
                    return

            # --- 子目標到達，準備下一個 ---
            current_subgoal_index += 1
            progress_idx = 0 # 為下一個 segment 重置 progress_idx

        # === 所有子目標完成 ===
        final_pos = env.agent.get_state().position
        dist_to_final = math.hypot(final_pos[0]-subgoals[-1][0], final_pos[2]-subgoals[-1][1])
        if dist_to_final < ARRIVAL_JUDGE * 2:
            print("[INFO] 所有子目標完成，執行最終朝向...") # 保留 Print
            face_goal(env, goal_pixel_orig, bounds, w, h, vw, target_mask, FPS)
        else:
            print(f"[WARN] 完成所有子目標，但離最終目標仍有 {dist_to_final:.2f}m。") # 保留 Print
            face_goal(env, goal_pixel_orig, bounds, w, h, vw, target_mask, FPS)

    # --- Exception Handling ---
    except KeyboardInterrupt:
        print("\n🛑 [INTERRUPT] User interrupted. Saving video...") # 保留 Print
    except Exception as e:
        import traceback
        print(f"\n❌ [UNCAUGHT ERROR] {type(e).__name__}: {e}") # 保留 Print
        print(traceback.format_exc()) # 保留 Print
        print("🚨 Saving current video and exiting.") # 保留 Print
    finally:
        # --- Cleanup ---
        if vw is not None and vw.isOpened():
             vw.release()
             print(f"[SAVE] Video safely saved ({frame_count} frames)") # 保留 Print
        try:
             env.sim.close()
             print("[CLEANUP] Simulator closed ✅") # 保留 Print
        except Exception as e_close:
             print(f"[WARN] Error closing simulator: {e_close}") # 保留 Print

    print(f"[END] Navigation finished after {global_replan_count} total replan rounds.") # 保留 Print

# ==========================================================
# 導航結果地圖輸出
# ==========================================================
def visualize_path_on_map(map_path, path, goal, start, target_class, output_dir, save_prefix="rrt_result_part3"):
    map_img = cv2.imread(map_path)
    if map_img is None:
        print(f"❌ 找不到地圖檔: {map_path}")
        return

    # 路徑線
    for i in range(len(path) - 1):
        p1, p2 = path[i], path[i + 1]
        cv2.line(map_img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 0, 255), 2)

    # 起點與目標
    cv2.circle(map_img, (int(start[0]), int(start[1])), 6, (0, 255, 0), -1)
    cv2.circle(map_img, (int(goal[0]), int(goal[1])), 6, (255, 0, 0), -1)

    # 標籤
    cv2.putText(map_img, "Start", (int(start[0]) + 10, int(start[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(map_img, "Goal", (int(goal[0]) + 10, int(goal[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    out_path = f"{output_dir}/{save_prefix}_{target_class}.png"
    cv2.imwrite(out_path, map_img)
    print(f"[INFO] 導航路徑已輸出至 {out_path}")
def simplify_path(path, min_step=0.2):
    """移除相鄰太近的點，確保距離至少 min_step。"""
    if not path:
        return []
    simplified = [path[0]]
    for p in path[1:]:
        if math.hypot(p[0] - simplified[-1][0], p[1] - simplified[-1][1]) >= min_step:
            simplified.append(p)
    return simplified

# ==========================================================
# 主程式
# ==========================================================
if __name__ == "__main__":
    
    print("=== HW2 Part3 (Safe Final Version) ===")
    MAP_PATH = "/home/clu98753cs13/Desktop/course/phyai/physical-ai25/hw2/map.png"
    EXCEL_PATH = "/home/clu98753cs13/Desktop/course/phyai/physical-ai25/hw2/color_coding_semantic_segmentation_classes.xlsx"
    BOUNDS_PATH = "/home/clu98753cs13/Desktop/course/phyai/physical-ai25/hw2/coordinate_bounds.json"
    OUTPUT_PATH = "./part3OUTPUT"
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    color_map = load_semantic_table(EXCEL_PATH)
    id_map = load_semantic_ID_table(EXCEL_PATH)
    
    TARGET_CLASS = "window"
    goal, mask = find_object_region(MAP_PATH, color_map, TARGET_CLASS)
    start = (335, 240) # window test final: 要繞過椅子等 還要左右轉彎
    # start = (236, 457) # window test1: 直線行走左右有障礙 可能卡牆 pass
    # start = (382, 256) # sofa test1: 直線走 pass
    # start = (212, 428) # base-cabinet test1: 直線走

    map_gray = cv2.imread(MAP_PATH, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(map_gray, 240, 255, cv2.THRESH_BINARY)
    # === 解決方案：侵蝕可行走區域 (建立安全緩衝區) ===
    print("[INFO] 正在侵蝕可行走區域以建立安全緩衝區...")
    
    # 決定緩衝區的大小 (kernel 越大，緩衝區越寬，路徑越保守)
    # 5x5 或 7x7 通常是個好的開始
    kernel_size = 25
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # 'binary' 中可行走區域是 255 (白色)，障礙物是 0 (黑色)
    # 'erode' (侵蝕) 會縮小白**色區域，使其遠離黑色邊界
    # 這等同於讓障礙物 (牆壁) "膨脹" 了 (kernel_size / 2) 個像素
    safe_binary_map = cv2.erode(binary, kernel, iterations=1)
    
    print("[INFO] 安全緩衝區建立完畢。")
    # === 解決方案結束 ===
    path, _ = rrt_star_planning(safe_binary_map, start, goal)
    if path is None:
        print("❌ 無法找到可行路徑。")
        exit()

    # pixel → world
    bounds = load_bounds(BOUNDS_PATH)
    h, w, _ = cv2.imread(MAP_PATH).shape
    
    world_path = [pixel_to_world(u, v, w, h, bounds) for (u, v) in path]
    # world_path = [(x * 40, z * 40) for (x, z) in world_path]  # 保留縮放
    world_path = simplify_path(world_path, min_step=0.25)
    print(f"[INFO] Simplified path from {len(path)} → {len(world_path)} points")
    # for i in range(5):
    #     (x1, z1), (x2, z2) = world_path[i], world_path[i+1]
    #     print(f"Segment {i}: dist={math.hypot(x2 - x1, z2 - z1):.3f}")

    # print("[DEBUG] First few world_path coords:")
    # for i, p in enumerate(world_path[:5]):
    #     print(f"  {i}: {p}")
    # Habitat 環境
    sim_settings = {
        "scene": "/home/clu98753cs13/Desktop/course/phyai/physical-ai25/hw0/replica_v1/apartment_0/habitat/mesh_semantic.ply",
        "default_agent": 0,
        "sensor_height": 1.5,
        "width": 512,
        "height": 512,
        "sensor_pitch": 0,
    }

    # === 導航 + 輸出 ===
    env = HabitatEnvWrapper(sim_settings)
    
    # TODO: 1.Which code snippet affects my position and angle?
    
    # TODO: 2.set init position 
    start_x, start_z = pixel_to_world(start[0], start[1], w, h, bounds)
    print(f"[INFO] Start world coord: ({start_x:.3f}, {start_z:.3f})")
    state = habitat_sim.AgentState()
    state.position = np.array([start_x, 0, start_z])  # y=高度 =0
    env.agent.set_state(state)
    print(f"[INFO] Agent placed at world position: {state.position}")

    # TODO: 3.set rotate
    # === 設定初始面向 ===
    dx = world_path[1][0] - world_path[0][0]
    dz = world_path[1][1] - world_path[0][1]
    init_yaw = math.atan2(-dx, -dz)  # 注意：dx, dz 的順序
    quat = np.array([0.0, math.sin(init_yaw / 2.0), 0.0, math.cos(init_yaw / 2.0)], dtype=np.float32)

    state.rotation = quat
    env.agent.set_state(state)

    print(f"[INFO] Agent initial yaw = {math.degrees(init_yaw):.2f}°")
    print(f"[INFO] Agent placed at {state.position} with rotation {state.rotation}")


    # TODO: 4. generate action and move
    """
    I think the robot’s step size and angle have a big impact.
    Also, for the motion generation part`generate_actions_from_world_path`, 
    I simplified it — it only searches "once" ,
    and doesn’t fit the path, so the path got simplified.
    """
    
    video_path = f"{OUTPUT_PATH}/{TARGET_CLASS}_safe_final.mp4"
    run_navigation_replan(env, binary, safe_binary_map, color_map, bounds,
                        start, goal, # 傳入像素座標
                        target_mask, output_video=video_path,
                        segment_distance_m=1.5)
    # run_navigation(env, world_path, target_mask, output_video=video_path)
    visualize_path_on_map(MAP_PATH, path, goal, start, TARGET_CLASS, OUTPUT_PATH)
    env.sim.close()
    print(f"[✅ DONE] 導航影片與地圖已輸出至 {OUTPUT_PATH}/")
