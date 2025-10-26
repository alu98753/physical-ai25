import os
import cv2
import math
import json
import numpy as np
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb, d3_40_colors_hex
from PIL import Image
import matplotlib.pyplot as plt
# 確保 rrt_star.py 就在旁邊
from rrt_star import rrt_star_planning
# 確保 part2.py 就在旁邊 (或您已將 rrt_star.py 獨立出來)
from part2 import load_semantic_ID_table, load_semantic_table, find_object_region

# ==========================================================
# Habitat 環境封裝
# ==========================================================
FORWARD_STEP = 0.05
ARRIVAL_JUDGE = FORWARD_STEP*1.5
TURN_ANGLE = 1
MAX_ACTIONS = 5000
FPS = 120
ARRIVAL_THRESH = 0.15 # 抵達目標的距離閾值 (米)

# === V3 控制器參數 ===
LOOKAHEAD_DISTANCE = 0.5        # (米) Pure Pursuit "胡蘿蔔" 的前瞻距離
OBSTACLE_CLEARANCE_DIST = 0.2  # (米) 認定為障礙物的深度閾值
PROACTIVE_THRESH_PIXELS = 3000  # (像素) 觸發主動避障的像素數量閾值
CORRECTION_ANGLE_THRESH = 5.0   # (度) 循跡時的角度容忍範圍
STUCK_LIMIT = 100               # (幀) 卡住/震盪多少幀後觸發「最後手段」
ESCAPE_BACKWARD_DIST = 0.5      # (米) 最後手段：後退距離
ESCAPE_TURN_ANGLE = 45          # (度) 最後手段：轉向角度

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
        # ✅ 【V2 洞見】: 整合 'move_backward' 動作
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=FORWARD_STEP)
            ),
            "move_backward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=-FORWARD_STEP) # 負值
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

# ==========================================================
# 座標轉換 (保持不變)
# ==========================================================
def load_bounds(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["xmin"], data["xmax"], data["zmin"], data["zmax"]

# ( ... distance_to_path, pixel_to_world, world_to_pixel 保持不變 ... )
def distance_to_path(current_pos, path):
    """回傳 agent 到當前路徑上最近點的距離"""
    if not path:
        return np.inf
    pxz = np.array([[px, pz] for px, pz in path])
    pos_xz = np.array([current_pos[0], current_pos[2]])
    dists = np.linalg.norm(pxz - pos_xz, axis=1)
    return np.min(dists)

def pixel_to_world(u, v, w, h, bounds):
    SCALE_FACTOR = 10000 / 255
    xmin_pt, xmax_pt, zmin_pt, zmax_pt = bounds
    x_pt = xmin_pt + (u / w) * (xmax_pt - xmin_pt)
    z_pt = zmin_pt + (1.0 - (v / h)) * (zmax_pt - zmin_pt)
    x_world = x_pt * SCALE_FACTOR
    z_world = z_pt * SCALE_FACTOR
    return float(x_world), float(z_world)

def world_to_pixel(x, z, w, h, bounds):
    SCALE_FACTOR = 10000 / 255
    xmin_pt, xmax_pt, zmin_pt, zmax_pt = bounds
    x_pt = x / SCALE_FACTOR
    z_pt = z / SCALE_FACTOR
    u_ratio = (x_pt - xmin_pt) / (xmax_pt - xmin_pt)
    v_ratio = 1.0 - (z_pt - zmin_pt) / (zmax_pt - zmin_pt)
    u = u_ratio * w
    v = v_ratio * h
    return u,v

def wrap_to_pi(a): return (a + math.pi) % (2 * math.pi) - math.pi

def find_escape_route(depth, hfov_deg=90.0, num_sectors=5):
    """
    分析深度圖 (depth map)，找出最開闊的逃脫方向。

    Args:
        depth: 深度攝影機
        hfov_deg: 相機的水平可視角度 (Horizontal FoV) (預設 90 度)
        num_sectors: 要將視野切成多少個扇區 (奇數)

    Returns:
        (action, angle_deg): 
        - action: "turn_left" 或 "turn_right"
        - angle_deg: 建議轉向的角度 (度)
        - 如果找不到好的路徑 (例如全都很近)，返回 (None, 0)
    """
    if num_sectors % 2 == 0:
        print("WARN: num_sectors 應為奇數，自動 +1")
        num_sectors += 1
        
    h, w = depth.shape
    
    # 僅取畫面中央 1/3 的垂直高度來分析，避免地板或天花板的干擾
    middle_slice = depth[h // 3 : h * 2 // 3, :]
    
    sector_width = w // num_sectors
    sector_avg_depths = []
    
    # 1. (Scan & Evaluate) 掃描並評估每個扇區
    for i in range(num_sectors):
        start_col = i * sector_width
        end_col = (i + 1) * sector_width if i < num_sectors - 1 else w
        sector = middle_slice[:, start_col:end_col]
        
        # 清理深度數據：
        # - 將 無窮大 (Inf) 和 無效值 (NaN) 替換為 0 (視為障礙)
        # - 將 0 (通常也是無效讀數) 忽略
        # - 將大於 10 公尺的視為 10 公尺 (上限)
        valid_depths = sector[(sector > 0.01) & np.isfinite(sector)]
        
        if len(valid_depths) == 0:
            sector_avg_depths.append(0.0) # 這個扇區完全被遮擋或無效
        else:
            valid_depths[valid_depths > 10.0] = 10.0
            avg_depth = np.mean(valid_depths)
            sector_avg_depths.append(avg_depth)

    print(f"[ESCAPE_SCAN] 扇區平均深度 (m): {[round(d, 2) for d in sector_avg_depths]}")

    # 2. (Act) 決策
    center_sector_idx = num_sectors // 2
    
    # 我們在這裡是因為「中間」被擋住了，所以我們只比較「中間以外」的扇區
    left_sectors_avg = np.mean(sector_avg_depths[:center_sector_idx])
    right_sectors_avg = np.mean(sector_avg_depths[center_sector_idx+1:])

    # 找出「最佳」扇區的索引 (不是中間的那個)
    # 建立一個副本，並把中間扇區的深度設為 0，這樣 argmax 就不會選到它
    depths_no_center = list(sector_avg_depths)
    depths_no_center[center_sector_idx] = 0.0
    
    best_sector_idx = np.argmax(depths_no_center)
    best_depth = sector_avg_depths[best_sector_idx]

    # 如果最好的逃脫路徑平均深度仍然小於 1 公尺，代表太窄，使用 45 度固定轉向
    if best_depth < 1.0:
        print(f"[ESCAPE_SCAN] 找不到開闊路徑 (最佳僅 {best_depth:.2f}m)，使用預設 45 度轉向。")
        action = "turn_left" if left_sectors_avg > right_sectors_avg else "turn_right"
        return action, 45.0 # Fallback

    # 3. 計算轉向角度
    # 找到最佳扇區的「中心像素」
    best_sector_center_pixel = (best_sector_idx * sector_width) + (sector_width // 2)
    
    # 將「像素」轉換為「角度」
    # (pixel / width) -> 範圍 [0.0, 1.0]
    # ((pixel / width) - 0.5) -> 範圍 [-0.5, 0.5]
    # ((pixel / width) - 0.5) * hfov_deg -> 範圍 [-45, 45] (假設 hfov=90)
    
    target_angle_deg = ((best_sector_center_pixel / w) - 0.5) * hfov_deg
    
    print(f"[ESCAPE_SCAN] 最佳扇區: {best_sector_idx}. 建議轉向角度: {target_angle_deg:.1f}°")

    if target_angle_deg < 0:
        return "turn_left", abs(target_angle_deg)
    else:
        return "turn_right", abs(target_angle_deg)
    
# ==========================================================
# 導航與影片錄製 (V1/V2 函數保持不變, V3 控制器會呼叫它們)
# ==========================================================
def overlay_mask(rgb, mask, color=(255, 0, 0), alpha=0.15):
    if rgb.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]))
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8)
    else:
        mask = (mask > 0).astype(np.uint8)
    color_mask = np.zeros_like(rgb, dtype=np.uint8)
    color_mask[mask > 0] = color
    dst = cv2.addWeighted(color_mask, alpha, rgb, 1 - alpha, 0)
    return dst

def face_goal(env, goal, bounds, w, h, vw, target_mask, FPS=120, TURN_ANGLE=1):
    """
    抵達目標後, 旋轉朝向目標並結束錄影。
    """
    pos = env.agent.get_state().position.copy()
    print(f"[SUCCESS] Reached goal ✅ ({pos[0]:.2f}, {pos[2]:.2f})")
    gx, gz = pixel_to_world(goal[0], goal[1], w, h, bounds)
    dx, dz = gx - pos[0], gz - pos[2]
    desired_yaw = math.atan2(-dx, -dz)
    q = env.agent.get_state().rotation
    current_yaw = 2 * math.atan2(q.imag[1], q.real)
    yaw_diff = wrap_to_pi(desired_yaw - current_yaw)
    print(f"[TURN] current_yaw={math.degrees(current_yaw):.1f}°, desired_yaw={math.degrees(desired_yaw):.1f}°, diff={math.degrees(yaw_diff):.1f}°")
    
    turn_action = "turn_left" if yaw_diff < 0 else "turn_right"
    turn_steps = int(abs(math.degrees(yaw_diff)) // TURN_ANGLE)
    print(f"[INFO] Turning {turn_steps} steps to face goal...")

    if turn_steps > 0:
        for i in range(turn_steps):
            obs = env.step(turn_action)
            rgb = obs["rgb"]
            mask = target_mask(obs)
            vis = overlay_mask(rgb, mask)
            for _ in range(FPS * 2 // max(turn_steps, 1)):
                vw.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    else:
        obs = env.step("turn_left")
        rgb = obs["rgb"]
        mask = target_mask(obs)
        vis = overlay_mask(rgb, mask)
        for _ in range(FPS // 10):
            vw.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    # ( ... 結尾動畫 ... )
    rgb = obs["rgb"].copy()
    mask = target_mask(obs)
    vis = overlay_mask(rgb, mask)
    vis = end_anime(vis)
    for _ in range(FPS * 3):
        vw.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    vw.release()
    print(f"[END] Navigation finished with facing target.")

def end_anime(rgb):
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

def simplify_path(path, min_step=0.2):
    if not path: return []
    simplified = [path[0]]
    for p in path[1:]:
        if math.hypot(p[0] - simplified[-1][0], p[1] - simplified[-1][1]) >= min_step:
            simplified.append(p)
    return simplified

def visualize_path_on_map(map_path, path, goal, start, target_class, output_dir, save_prefix="rrt_result_part3"):
    map_img = cv2.imread(map_path)
    if map_img is None: return
    for i in range(len(path) - 1):
        p1, p2 = path[i], path[i + 1]
        cv2.line(map_img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 0, 255), 2)
    cv2.circle(map_img, (int(start[0]), int(start[1])), 6, (0, 255, 0), -1)
    cv2.circle(map_img, (int(goal[0]), int(goal[1])), 6, (255, 0, 0), -1)
    cv2.putText(map_img, "Start", (int(start[0]) + 10, int(start[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(map_img, "Goal", (int(goal[0]) + 10, int(goal[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    out_path = f"{output_dir}/{save_prefix}_{target_class}.png"
    cv2.imwrite(out_path, map_img)
    print(f"[INFO] 導航路徑已輸出至 {out_path}")



# #########################################
def _execute_smart_escape(env, escape_turn_angle, escape_backward_dist, proactive_avoidance_threshold,
                        vw, target_mask, frame_count, FPS, FORWARD_STEP, TURN_ANGLE,commit_distance_m=0.5):
    """
    執行智能脫困程序 (V6 "Peek" 邏輯)。
    - Peek 左/右 尋找最佳方向。
    - 後退。
    - 轉向最佳方向。
    - 向前移動直到脫離障礙物。
    - 返回: (final_obs, updated_frame_count)
    """
    
    print(f"[ESCAPE] Smart escape maneuver triggered. Peeking for escape route...")
    
    # --- 1. Get original state ---
    original_state = env.agent.get_state()
    original_pos = original_state.position
    original_rot = original_state.rotation
    
    # --- 2. Define turn quaternions (for peeking) ---
    turn_angle_rad = math.radians(escape_turn_angle)
    y_axis = np.array([0, 1, 0])
    
    q_left = habitat_sim.utils.common.quat_from_angle_axis(
        turn_angle_rad, y_axis
    )
    q_right = habitat_sim.utils.common.quat_from_angle_axis(
        -turn_angle_rad, y_axis
    )

    # --- 3. Peek Left ---
    left_rot = original_rot * q_left
    left_state = habitat_sim.AgentState()
    left_state.position = original_pos
    left_state.rotation = left_rot
    
    env.agent.set_state(left_state)
    obs_left = env.sim.get_sensor_observations()
    depth_left = obs_left["depth_sensor"]
    near_pixels_left = np.sum(depth_left < 0.2)
    print(f"[ESCAPE] Peek Left ({escape_turn_angle} deg): {near_pixels_left} near pixels.")

    # --- 4. Peek Right ---
    right_rot = original_rot * q_right
    right_state = habitat_sim.AgentState()
    right_state.position = original_pos
    right_state.rotation = right_rot
    
    env.agent.set_state(right_state)
    obs_right = env.sim.get_sensor_observations()
    depth_right = obs_right["depth_sensor"]
    near_pixels_right = np.sum(depth_right < 0.2)
    print(f"[ESCAPE] Peek Right (-{escape_turn_angle} deg): {near_pixels_right} near pixels.")

    # --- 5. Return to Original State (CRITICAL) ---
    env.agent.set_state(original_state)
    print("[ESCAPE] Returned to original orientation for decision.")

    # --- 6. Decide Best Escape Direction ---
    if near_pixels_left <= near_pixels_right:
        escape_action = "turn_left"
        print(f"[ESCAPE] Decision: Left is clearer ({near_pixels_left} vs {near_pixels_right}). Escaping left.")
    else:
        escape_action = "turn_right"
        print(f"[ESCAPE] Decision: Right is clearer ({near_pixels_right} vs {near_pixels_left}). Escaping right.")

    # --- 7. Execute Escape: Backward ---
    print(f"[ESCAPE] Executing: Backward {escape_backward_dist}m")
    backward_steps = max(1, int(escape_backward_dist / FORWARD_STEP))
    obs = None # 確保 obs 至少被定義
    for _ in range(backward_steps):
        obs = env.step("move_backward")
        rgb = obs["rgb"]; mask = target_mask(obs); vis = overlay_mask(rgb, mask)
        vw.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)); frame_count += 1
    
    # --- 8. Execute Escape: Initial Turn ---
    print(f"[ESCAPE] Executing: Initial Turn {escape_action} {escape_turn_angle} deg")
    turn_steps = int(escape_turn_angle // TURN_ANGLE)
    for _ in range(turn_steps):
        obs = env.step(escape_action)
        rgb = obs["rgb"]; mask = target_mask(obs); vis = overlay_mask(rgb, mask)
        vw.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)); frame_count += 1
    
# ========== [!!! V9 FIX !!!] ==========
    # --- 9. V9 Logic: Escape "Commit" Sub-Loop ---
    # 強制向前移動 commit_distance_m，以鞏固脫困方向
    # 除非在此過程中又撞到 *新* 的障礙物
    print(f"[ESCAPE] Entering escape sub-loop: Forcing move forward {commit_distance_m}m to commit...")
    escape_steps = 0
    
    # 'obs' 來自上一個轉彎動作
    if obs is None: # 以防萬一
         obs = env.step("turn_left"); obs = env.step("turn_right")

    max_escape_steps = max(1, int(commit_distance_m / FORWARD_STEP))
    stuck_pos_during_escape = env.agent.get_state().position.copy()

    while escape_steps < max_escape_steps:
        print(f"[ESCAPE] Commit Step {escape_steps+1}/{max_escape_steps}: Moving forward...")
        obs = env.step("move_forward")
        
        # 檢查新觀測
        current_depth = obs["depth"]
        current_near_pixels = np.sum(current_depth < 0.2)
        
        # 錄影
        rgb = obs["rgb"]; mask = target_mask(obs); vis = overlay_mask(rgb, mask)
        vw.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)); frame_count += 1
        
        escape_steps += 1
        
        # 檢查在「鞏固時」是否又卡住了 (撞到 *新* 障礙物)
        if current_near_pixels > proactive_avoidance_threshold:
            print(f"[ESCAPE] Hit a *new* obstacle during escape commit (near={current_near_pixels})! Aborting commit.")
            break # 停止 "commit"
            
        # 檢查是否物理上卡住
        new_pos = env.agent.get_state().position
        if escape_steps > 5 and np.linalg.norm(new_pos - stuck_pos_during_escape) < 0.01:
            print("[ESCAPE] Physically stuck during escape move! Aborting commit.")
            break
        stuck_pos_during_escape = new_pos
        
    if escape_steps >= max_escape_steps:
        print(f"[ESCAPE] Escape commit complete ({commit_distance_m}m).")
    else:
        print(f"[ESCAPE] Escape commit cut short after {escape_steps} steps.")
    # ========== [!!! END OF V9 FIX !!!] ==========

    # --- 10. Return final state ---
    return obs, frame_count

def run_navigation_replan(env, binary_map, safe_binary_map, color_map, bounds, start_pixel_orig, goal_pixel_orig, target_mask,
                        output_video="result_replan.mp4", replan_thresh=0.3, segment_distance_m=2.0,
                        # --- Avoidance Params ---
                        proactive_avoidance_threshold=5000,
                        escape_backward_dist=0.5,
                        escape_turn_angle=30,         # 'Peek' 和 'Turn' 都使用這個角度
                        escape_commit_distance_m=1.25,
                        # --- Controller Params ---
                        look_ahead_points=5,
                        turn_threshold_deg=5.0,
                        stuck_threshold=5):
    
    """
    Main navigation loop (V8 - Refactored Smart Escape):
    - Implements "peek left/right" logic in a helper function.
    - Uses new oscillation detection logic.
    - Calls the *same* helper function for *both* proactive avoidance and stuck/oscillation.
    """
    vw = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (512, 512))
    frame_count = 0
    h, w = binary_map.shape

    # === 1. Initialization & Subgoal Generation (Unchanged) ===
    print("[INFO] Starting initial path planning (for subgoal generation)...")
    initial_path_pixel, _ = rrt_star_planning(safe_binary_map, start_pixel_orig, goal_pixel_orig)
    if initial_path_pixel is None:
        print("[FAIL] Initial RRT* planning failed. Cannot start navigation.")
        vw.release()
        return
    initial_world_path = [pixel_to_world(u, v, w, h, bounds) for (u, v) in initial_path_pixel]
    if not initial_world_path:
        print("[FAIL] Initial world path is empty. Cannot start navigation.")
        vw.release()
        return
    subgoals = create_subgoals(initial_world_path, segment_distance_m)
    subgoals = subgoals[1:] # Remove start point
    if not subgoals: 
        subgoals = [initial_world_path[-1]]

    # === 2. Main Loop (Outer) ===
    current_subgoal_index = 0
    global_replan_count = 0
    
    # Get initial observation (obs)
    try:
        obs = env.step("turn_right")
        obs = env.step("turn_left") # Turn back, get current view
    except Exception as e_init_obs:
        print(f"[ERROR] Could not get initial observation: {e_init_obs}")
        return

    try:
        while current_subgoal_index < len(subgoals):

            # --- 2a. Set Current Segment Target ---
            current_target_subgoal_world = subgoals[current_subgoal_index]
            current_target_subgoal_pixel = world_to_pixel(current_target_subgoal_world[0], current_target_subgoal_world[1], w, h, bounds)
            print(f"\n========== [Navigating to Subgoal {current_subgoal_index}/{len(subgoals)-1}] ==========")
            print(f"Target World Coords: ({current_target_subgoal_world[0]:.2f}, {current_target_subgoal_world[1]:.2f})")

            # --- 2b. Plan Current Path Segment ---
            current_state = env.agent.get_state()
            current_pos = current_state.position.copy()
            current_pixel = world_to_pixel(current_pos[0], current_pos[2], w, h, bounds)
            
            path_pixel_segment, _ = rrt_star_planning(safe_binary_map, current_pixel, current_target_subgoal_pixel)
            retry_segment = 0
            while path_pixel_segment is None and retry_segment < 3:
                 print(f"[WARN] Cannot plan path to subgoal {current_subgoal_index}, retrying ({retry_segment+1}/3)...")
                 path_pixel_segment, _ = rrt_star_planning(safe_binary_map, current_pixel, current_target_subgoal_pixel)
                 retry_segment += 1
            if path_pixel_segment is None:
                 print(f"[FAIL] Failed to plan path to subgoal {current_subgoal_index} after retries. Aborting.")
                 return

            current_segment_world_path = [pixel_to_world(u, v, w, h, bounds) for (u, v) in path_pixel_segment]
            if not current_segment_world_path:
                 print(f"[FAIL] Path segment for subgoal {current_subgoal_index} is empty. Aborting.")
                 return
            
            # --- 2c. Inner Loop: Reactive Navigation ---
            subgoal_reached = False
            progress_idx = 0
            stuck_check_counter = 0 # 這是新的計時器
            stuck_check_pos_start = current_pos.copy() # 這是新的起始位置

            local_replan_count_this_segment = 0

            while not subgoal_reached:
                
                # --- (1) Get Current State ---
                state = env.agent.get_state()
                pos = state.position.copy()
                q = state.rotation
                current_yaw = 2 * math.atan2(q.imag[1], q.real)
                
                # --- (2) Check Arrival ---
                dist_to_current_subgoal = math.hypot(pos[0] - current_target_subgoal_world[0], pos[2] - current_target_subgoal_world[1])
                if dist_to_current_subgoal < ARRIVAL_JUDGE * 4: # Reached subgoal
                    print(f"[INFO] Reached Subgoal {current_subgoal_index} ({current_target_subgoal_world[0]:.2f}, {current_target_subgoal_world[1]:.2f})")
                    subgoal_reached = True
                    break # Exit inner loop, move to next subgoal

                # ========== [!!! V9 FIX !!!] ==========
                # --- (3) Check for Proactive Obstacle Avoidance ---
                # ** 關鍵修正: **
                # 我們現在使用來自 *上一個迴圈* (L247) 的 `obs` 變數
                # 不再呼叫 env.step("turn_left") / env.step("turn_right")
                
                depth = obs["depth"]
                near_mask = (depth < 0.2)
                near_pixels = np.sum(near_mask)
                print(f"[DEPTH] near<0.2m: {near_pixels}, pos=({pos[0]:.2f},{pos[2]:.2f}), yaw={math.degrees(current_yaw):.1f} deg")

                if near_pixels > proactive_avoidance_threshold:
                    print(f"[AVOID] Proactive obstacle detected (near={near_pixels} > {proactive_avoidance_threshold}).")
                    
                    # 呼叫 *更新後* 的脫困函式
                    obs, frame_count = _execute_smart_escape(
                        env, escape_turn_angle, escape_backward_dist, proactive_avoidance_threshold,
                        vw, target_mask, frame_count, FPS, FORWARD_STEP, TURN_ANGLE,
                        commit_distance_m=escape_commit_distance_m # 傳入新參數
                    )
                    
                    print("[AVOID] Maneuver complete. Forcing replan...")
                    break # 退出內層 while 迴圈，觸發 RRT* 重新規劃
                # ========== END OF V8 AVOIDANCE LOGIC ==========


                # --- (4) Check Deviation ---
                dists_all = [math.hypot(px - pos[0], pz - pos[2]) for (px, pz) in current_segment_world_path[progress_idx:]]
                if not dists_all:
                     print("[WARN] Path points exhausted but subgoal not reached. Forcing replan.")
                     break 
                
                nearest_local_idx = int(np.argmin(dists_all))
                current_dist_to_path = dists_all[nearest_local_idx]
                progress_idx += nearest_local_idx # Update progress along path

                if current_dist_to_path > replan_thresh:
                    print(f"[REPLAN] Deviated from path ({current_dist_to_path:.2f} m > {replan_thresh}m). Forcing replan.")
                    break # Exit inner loop to trigger replan

                # --- (5) Decision: Turn or Move (Pure Pursuit) ---
                look_ahead_idx = min(progress_idx + look_ahead_points, len(current_segment_world_path) - 1)
                target_point = current_segment_world_path[look_ahead_idx]

                dx, dz = target_point[0] - pos[0], target_point[1] - pos[2]
                desired_yaw = math.atan2(-dx, -dz) # Use consistent yaw calculation
                
                yaw_diff_rad = wrap_to_pi(desired_yaw - current_yaw)
                angle_diff_deg = abs(math.degrees(yaw_diff_rad))

                action_to_take = ""
                if angle_diff_deg > turn_threshold_deg:
                    # Turn to align
                    action_to_take = "turn_right" if yaw_diff_rad < 0 else "turn_left"
                    print(f"[NAV] Aligning. Diff: {angle_diff_deg:.1f} deg > {turn_threshold_deg} deg. Action: {action_to_take}")
                    stuck_check_counter = 0
                    stuck_check_pos_start = pos.copy()
                else:
                    # Alignmenet is good, move forward
                    action_to_take = "move_forward"
                    print(f"[NAV] Moving forward. Diff: {angle_diff_deg:.1f} deg")
                    stuck_check_counter += 1

                # ========== [!!! MODIFIED V8 LOGIC !!!] ==========
                
                if stuck_check_counter > stuck_threshold:
                    # 時間到了，檢查一下
                    move_dist_since_check = np.linalg.norm(pos - stuck_check_pos_start)
                    
                    if move_dist_since_check < 0.1: # 在 N 幀內移動不到 10 公分
                        print(f"[REPLAN] Agent stuck (oscillation or unseen obstacle).")
                        print(f"[REPLAN] Moved < 0.1m in {stuck_threshold} frames. Initiating smart escape...")

                        # 呼叫 *相同* 的脫困函式
                        obs, frame_count = _execute_smart_escape(
                            env, escape_turn_angle, escape_backward_dist, proactive_avoidance_threshold,
                            vw, target_mask, frame_count, FPS, FORWARD_STEP, TURN_ANGLE,
                            commit_distance_m=escape_commit_distance_m
                        )

                        print("[REPLAN] (Stuck) Escape complete. Forcing replan from new position.")
                        break # 退出內層迴圈以觸發 replan
                    
                    # 如果 *有* 移動，則重置計數器
                    stuck_check_counter = 0
                    stuck_check_pos_start = pos.copy()
                # ========== [!!! END OF V9 FIX !!!] ==========

                # --- (7) Execute Action & Record Video ---
                obs = env.step(action_to_take) # CRITICAL: Get obs for *next* loop's check

                rgb = obs["rgb"]
                mask = target_mask(obs)
                vis = overlay_mask(rgb, mask)
                for sub in range(FPS//15):
                    vw.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                    frame_count += 1

            # --- Inner loop finished ---

            if subgoal_reached:
                # Success, move to next subgoal
                current_subgoal_index += 1
                progress_idx = 0 
            else:
                # Replan triggered (Avoidance, Deviation, Stuck, or Path End)
                print(f"--- [Triggering Replan for Subgoal {current_subgoal_index}] ---")
                global_replan_count += 1
                local_replan_count_this_segment += 1
                if local_replan_count_this_segment > 10:
                     print(f"[FAIL] Replan limit exceeded for subgoal {current_subgoal_index}. Aborting.")
                     return

        # --- Outer loop finished (all subgoals reached) ---
        
        print("[INFO] All subgoals completed. Executing final turn to goal...")
        face_goal(env, goal_pixel_orig, bounds, w, h, vw, target_mask, FPS)

    except KeyboardInterrupt:
        print("\n[INTERRUPT] User interrupted. Saving video...")
    except Exception as e:
        import traceback
        print(f"\n[UNCAUGHT ERROR] {type(e).__name__}: {e}")
        print(traceback.format_exc())
        print("[CRITICAL] Saving current video and exiting.")
    finally:
        # --- Cleanup ---
        if vw is not None and vw.isOpened():
             vw.release()
             print(f"[SAVE] Video safely saved ({frame_count} frames)")
        try:
             env.sim.close()
             print("[CLEANUP] Simulator closed.")
        except Exception as e_close:
             print(f"[WARN] Error closing simulator: {e_close}")

    print(f"[END] Navigation finished after {global_replan_count} total replan rounds.")



# ==========================================================
# 主程式 (基於 V1 修改)
# ==========================================================
if __name__ == "__main__":
    
    print("=== HW2 Part3 (V3 - 統一控制器版本) ===")

    currdir = os.path.dirname(os.path.abspath(__file__))
    MAP_PATH = os.path.join(currdir, "map.png")
    EXCEL_PATH = os.path.join(
        currdir, "color_coding_semantic_segmentation_classes.xlsx")
    BOUNDS_PATH = os.path.join(currdir, "coordinate_bounds.json")
    OUTPUT_PATH = "./part3OUTPUT"
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    color_map = load_semantic_table(EXCEL_PATH)
    id_map = load_semantic_ID_table(EXCEL_PATH)
    
    # 測試用例
    TARGET_CLASS = "window"
    goal, mask = find_object_region(MAP_PATH, color_map, TARGET_CLASS)
    start = (335, 240) # 複雜繞行
    # start = (212, 428) # 直線

    map_gray = cv2.imread(MAP_PATH, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(map_gray, 240, 255, cv2.THRESH_BINARY)
    
    # --- 安全地圖 (用於 RRT* 規劃) ---
    print("[INFO] 正在建立 RRT* 安全緩衝區...")
    kernel_size = 15 # 保留 V1 的 15x15
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    safe_binary_map = cv2.erode(binary, kernel, iterations=1)
    print("[INFO] 安全緩衝區建立完畢。")
    
    # --- ✅ 步驟 1: 全域路徑規劃 (只執行一次) ---
    path, _ = rrt_star_planning(safe_binary_map, start, goal)
    if path is None:
        print("❌ 無法找到初始可行路徑。")
        exit()

    bounds = load_bounds(BOUNDS_PATH)
    h, w, _ = cv2.imread(MAP_PATH).shape
    
    world_path = [pixel_to_world(u, v, w, h, bounds) for (u, v) in path]
    world_path = simplify_path(world_path, min_step=0.25) # 簡化路徑點
    print(f"[INFO] 初始全域路徑: {len(path)} 點 -> 簡化為 {len(world_path)} 點")
    
    # --- Habitat 環境 ---
    sim_settings = {
        "scene": "/home/clu98753cs13/Desktop/course/phyai/physical-ai25/hw0/replica_v1/apartment_0/habitat/mesh_semantic.ply",
        "default_agent": 0,
        "sensor_height": 1.5,
        "width": 512,
        "height": 512,
        "sensor_pitch": 0,
    }

    # --- Target Mask 函數 (保持不變) ---
    def hex_to_rgb(hex_color):
        hex_color = hex_color.strip().lower().replace('0x', '').replace('#', '')
        if len(hex_color) != 6: raise ValueError(f"❌ 無效的 hex 色碼: {hex_color}")
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def target_mask(obs):
        semantic_img = obs["semantic"]
        target_id = id_map[TARGET_CLASS.lower()] % 40
        target_rgb = hex_to_rgb(d3_40_colors_hex[target_id])
        mask = cv2.inRange(semantic_img, target_rgb, target_rgb)
        return (mask > 0).astype(np.uint8)

    # === ✅ 步驟 2: 啟動環境與 Agent ===
    env = HabitatEnvWrapper(sim_settings)
    
    # 設置初始位置
    start_x, start_z = pixel_to_world(start[0], start[1], w, h, bounds)
    state = habitat_sim.AgentState()
    state.position = np.array([start_x, 0, start_z])
    
    # 設置初始朝向 (面向路徑的第二個點)
    dx = world_path[1][0] - world_path[0][0]
    dz = world_path[1][1] - world_path[0][1]
    init_yaw = math.atan2(-dx, -dz) 
    quat = np.array([0.0, math.sin(init_yaw / 2.0), 0.0, math.cos(init_yaw / 2.0)], dtype=np.float32)
    state.rotation = quat
    
    env.agent.set_state(state)
    print(f"[INFO] Agent placed at {state.position}, yaw = {math.degrees(init_yaw):.2f}°")

    # === ✅ 步驟 3: 執行 V3 統一控制器 ===
    video_path = f"{OUTPUT_PATH}/{TARGET_CLASS}_V3_controller.mp4"
    run_navigation_replan(env, binary, safe_binary_map, color_map, bounds, start, goal, target_mask,
                output_video="result_replan.mp4", replan_thresh=0.3, segment_distance_m=2.0,
                # --- 避障參數 ---
                proactive_avoidance_threshold=5000,
                escape_backward_dist=0.5,
                escape_turn_angle=45,
                # --- 新增：控制器參數 ---
                look_ahead_points=5,       # Pure Pursuit: 預瞄路徑上未來第幾個點
                turn_threshold_deg=5.0,    # 角度偏差 > 5° 才轉彎
                stuck_threshold=20)

    visualize_path_on_map(MAP_PATH, path, goal, start, TARGET_CLASS, OUTPUT_PATH, save_prefix="rrt_result_V3_initial")
    env.sim.close()
    print(f"[✅ DONE] V3 導航影片與地圖已輸出至 {OUTPUT_PATH}/")