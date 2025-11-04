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
from rrt_star import *
# 確保 part2.py 就在旁邊 (或您已將 rrt_star.py 獨立出來)
# from part2 import *
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
LOOKAHEAD_DISTANCE = 0.2        # (米) Pure Pursuit "胡蘿蔔" 的前瞻距離
OBSTACLE_CLEARANCE_DIST = 0.2  # (米) 認定為障礙物的深度閾值
PROACTIVE_THRESH_PIXELS = 3000  # (像素) 觸發主動避障的像素數量閾值
CORRECTION_ANGLE_THRESH = 2.0   # (度) 循跡時的角度容忍範圍
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
        def make_foot_sensor(uuid, stype):
            spec = habitat_sim.CameraSensorSpec()
            spec.uuid = uuid
            spec.sensor_type = stype
            spec.resolution = [settings["height"], settings["width"]]
            spec.position = [0.0, settings["sensor_foot_height"], 0.0]
            # spec.orientation = [-math.pi / 4, 0, 0]            

            return spec
        agent_cfg.sensor_specifications = [
            make_sensor("color_sensor", habitat_sim.SensorType.COLOR),
            make_sensor("depth", habitat_sim.SensorType.DEPTH),
            make_sensor("semantic_sensor", habitat_sim.SensorType.SEMANTIC),
            make_foot_sensor("foot_depth", habitat_sim.SensorType.DEPTH)
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
            "depth": obs["depth"],
            "foot_depth": obs["foot_depth"],
            "semantic": self._decode_semantic(obs["semantic_sensor"]),
        }

    @staticmethod
    def _decode_semantic(semantic_obs):
        img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        img.putpalette(d3_40_colors_rgb.flatten())
        img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        return np.asarray(img.convert("RGB"))


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def transform_depth(image):
    depth_img = (image / 10 * 255).astype(np.uint8)
    return depth_img

def transform_semantic(semantic_obs):
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    semantic_img = cv2.cvtColor(np.asarray(semantic_img), cv2.COLOR_RGB2BGR)
    return semantic_img

# --- Target Mask 函數 ---
def hex_to_rgb(hex_color):
    hex_color = hex_color.strip().lower().replace('0x', '').replace('#', '')
    if len(hex_color) != 6: raise ValueError(f"❌ 無效的 hex 色碼: {hex_color}")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def target_mask(obs):
    semantic_img = obs["semantic"]
    target_id = id_map[target_class.lower()] % 40
    target_rgb = hex_to_rgb(d3_40_colors_hex[target_id])
    mask = cv2.inRange(semantic_img, target_rgb, target_rgb)
    return (mask > 0).astype(np.uint8)


def navigateAndSee(env_instance, action, target_mask_func):
    """
    執行一步、即時顯示觀測，並返回觀測值。
    """
    # 1. 執行 Wrapper 的 step
    # 這會返回處理過的 {"rgb":..., "depth":..., "semantic":...}
    observations = env_instance.step(action)

    # 2. 準備主顯示 (使用 V3 的 overlay_mask 函數)
    rgb_img = observations["rgb"]
    mask = target_mask_func(observations) # 呼叫 target_mask
    vis_img = overlay_mask(rgb_img, mask) # 合成遮罩

    # 3. 顯示所有觀測
    cv2.imshow("Navigation (Overlay)", transform_rgb_bgr(vis_img))
    # cv2.imshow("Depth (Debug)", transform_depth(observations["depth"]))
    # cv2.imshow("Semantic (Debug)", transform_rgb_bgr(observations["semantic"]))

    # 4. 顯示攝影機姿態 (來自您的程式碼)
    agent_state = env_instance.agent.get_state()
    sensor_state = agent_state.sensor_states['color_sensor']
    # print("camera pose: x y z rw rx ry rz")
    # print(sensor_state.position[0],...) # (建議註解掉, 否則 log 會爆炸)

    # 5. 刷新視窗 (!!!! 關鍵中的關鍵 !!!!)
    # 沒有這行, 圖片不會更新
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): # 允許按 'q' 提早結束
        raise KeyboardInterrupt("User pressed 'q' to quit.")

    # 6. 返回觀測值 (!!!! 關鍵 !!!!)
    # 主導航邏輯 (run_navigation_replan) 需要這個 obs 來檢查深度
    return observations


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

def face_goal(env, goal, bounds, w, h, target_mask, FPS=120, TURN_ANGLE=1):
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
            obs = navigateAndSee(env, turn_action, target_mask)
    else:
        obs = navigateAndSee(env, "turn_left", target_mask)

    # ( ... 結尾動畫 ... )
    rgb = obs["rgb"].copy()
    mask = target_mask(obs)
    vis = overlay_mask(rgb, mask)
    vis = end_anime(vis)
    vis_bgr = transform_rgb_bgr(vis)
    cv2.imshow("Navigation (Overlay)", vis_bgr)
    print(f"[END] Navigation finished with facing target. Displaying final frame...")
    cv2.waitKey(3000)

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

def create_subgoals(world_path, segment_distance_m=1.0):
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

def count_near_pixels(depth, near_m=0.25):
    h, w = depth.shape
    r0, r1 = int(h * 0.55), int(h * 0.95)   # 下半部
    c0, c1 = int(w * 0.35), int(w * 0.65)   # 中央窄窗
    roi = depth[r0:r1, c0:c1]
    valid = np.isfinite(roi) & (roi > 0.01)
    if not np.any(valid):
        return 0
    return int(np.sum(roi[valid] < near_m)), roi.size


def get_yaw_from_quat(q):
    # Habitat: yaw around Y，跟你先前用法一致
    return float(2.0 * math.atan2(q.imag[1], q.real))

def calibrate_turn_sign(env):
    """檢查 'turn_left' 會讓 yaw 增加或減少；回傳 +1 或 -1。"""
    s0 = env.agent.get_state()
    yaw0 = get_yaw_from_quat(s0.rotation)
    env.sim.step("turn_left")
    yaw1 = get_yaw_from_quat(env.agent.get_state().rotation)
    env.agent.set_state(s0)
    dyaw = wrap_to_pi(yaw1 - yaw0)
    return 1.0 if dyaw > 0 else -1.0  # 左轉若讓 yaw 變大 → +1；否則 -1


# =========================
# 極簡循跡控制器（無避障）
# =========================
def get_current_yaw_rad(env):
    q = env.agent.get_state().rotation  # [x,y,z,w] 對應 imag={x,y,z}, real=w
    return 2.0 * math.atan2(q.imag[1], q.real)  # Habitat y 軸為上，採用 yaw around +y

def yaw_to_target_rad(curr_pos, target_x, target_z):
    # Habitat 座標 -> 你先前用法：desired = atan2(-dx, -dz)
    dx = target_x - curr_pos[0]
    dz = target_z - curr_pos[2]
    return math.atan2(-dx, -dz)

def find_lookahead_point(world_path, pos_xz, search_start_idx, L):
    """
    從 search_start_idx 之後往前掃，找第一個與當前位置距離 >= L 的點。
    若沒找到，就回傳最終目標。
    回傳：(tx, tz, new_start_idx)
    """
    px, pz = pos_xz
    n = len(world_path)
    best_j = None
    for j in range(search_start_idx, n):
        d = math.hypot(world_path[j][0] - px, world_path[j][1] - pz)
        if d >= L:
            best_j = j
            break
    if best_j is None:
        best_j = n - 1
    # 為了避免回頭，讓起始索引單調不減
    new_start = max(search_start_idx, min(best_j, n - 1))
    return world_path[best_j][0], world_path[best_j][1], new_start

def run_navigation_simple(env, world_path, target_mask_func,
                          lookahead=LOOKAHEAD_DISTANCE,
                          angle_thresh_deg=CORRECTION_ANGLE_THRESH,
                          max_actions=MAX_ACTIONS):
    """
    無腦循跡：對齊->前進->重複。無任何避障/卡住處理。
    world_path: [(x_world, z_world), ...]
    """
    assert len(world_path) >= 1, "world_path 不可為空"
    actions = 0
    carrot_idx = 0  # 從路徑開頭往前找胡蘿蔔點

    # 先確保朝向路徑第二點（如果有）
    if len(world_path) >= 2:
        st = env.agent.get_state()
        dx = world_path[1][0] - world_path[0][0]
        dz = world_path[1][1] - world_path[0][1]
        init_yaw = math.atan2(-dx, -dz)
        quat = np.array([0.0, math.sin(init_yaw / 2.0), 0.0, math.cos(init_yaw / 2.0)], dtype=np.float32)
        st.rotation = quat
        env.agent.set_state(st)

    while actions < max_actions:
        st = env.agent.get_state()
        pos = st.position.copy()   # [x, y, z]
        curr_yaw = get_current_yaw_rad(env)

        # 抵達判定（對世界座標的終點）
        gx, gz = world_path[-1]
        if math.hypot(gx - pos[0], gz - pos[2]) <= ARRIVAL_THRESH:
            # 顯示最後一幀讓你看
            _ = navigateAndSee(env, "turn_left", target_mask_func)  # 隨便觸發一次刷新
            break

        # 選胡蘿蔔點（前視 lookahead 公尺）
        tx, tz, carrot_idx = find_lookahead_point(world_path, (pos[0], pos[2]), carrot_idx, lookahead)

        # 角度誤差
        desired_yaw = yaw_to_target_rad(pos, tx, tz)
        yaw_err = wrap_to_pi(desired_yaw - curr_yaw)
        yaw_err_deg = math.degrees(yaw_err)

        # 轉 or 走
        if abs(yaw_err_deg) > angle_thresh_deg:
            action = "turn_left" if yaw_err < 0 else "turn_right"
        else:
            action = "move_forward"

        _ = navigateAndSee(env, action, target_mask_func)
        actions += 1

    # 抵達後轉正對終點（世界座標版）
    face_world_goal(env, world_path[-1], target_mask_func)

def face_world_goal(env, world_goal_xz, target_mask_func, TURN_ANGLE=TURN_ANGLE):
    """
    終點朝向校正（世界座標版，不需要像素/bounds）
    """
    pos = env.agent.get_state().position.copy()
    dx = world_goal_xz[0] - pos[0]
    dz = world_goal_xz[1] - pos[2]
    desired_yaw = math.atan2(-dx, -dz)

    q = env.agent.get_state().rotation
    current_yaw = 2 * math.atan2(q.imag[1], q.real)
    yaw_diff = wrap_to_pi(desired_yaw - current_yaw)
    steps = int(abs(math.degrees(yaw_diff)) // TURN_ANGLE)

    turn_action = "turn_left" if yaw_diff < 0 else "turn_right"
    for _ in range(max(1, steps)):
        _ = navigateAndSee(env, turn_action, target_mask_func)

# ==========================================================
# 主程式
# ==========================================================
if __name__ == "__main__":
    
    print("=== HW2 Part3 (V3 - 統一控制器版本) ===")
    DPI = 300
    currdir = os.path.dirname(os.path.abspath(__file__))
    MAP_PATH = os.path.join(currdir, f"map{DPI}.png")
    EXCEL_PATH = os.path.join(
        currdir, "color_coding_semantic_segmentation_classes.xlsx")
    BOUNDS_PATH = os.path.join(currdir, "coordinate_bounds.json")
    OUTPUT_PATH = "./part3OUTPUT"
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
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
    # visualize_multiple_goals(MAP_PATH, goals_list, target_class)     # 顯示所有找到的窗戶

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

    # 挑選起點 執行 RRT* 
    start = select_start(MAP_PATH, goal)
    path, nodes = rrt_star_planning(binary_map_with_goal, start, goal, SAFE_WEIGHT=500000)

    # 顯示結果
    if path:
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

    visualize_rrt(binary_map, nodes, start, goal, path)

    bounds = load_bounds(BOUNDS_PATH)
    h, w, _ = cv2.imread(MAP_PATH).shape
    
    world_path = [pixel_to_world(u, v, w, h, bounds) for (u, v) in path]
    # world_path = simplify_path(world_path, min_step=0.25) # 簡化路徑點
    print(f"[INFO] 初始全域路徑: {len(path)} 點 -> 簡化為 {len(world_path)} 點")
    
    # --- Habitat 環境 ---
    sim_settings = {
        "scene": "/home/clu98753cs13/Desktop/course/phyai/physical-ai25/hw2/replica_v1/apartment_0/habitat/mesh_semantic.ply",
        "default_agent": 0,
        "sensor_height": 1.5,
        "sensor_foot_height": 0.2,
        "width": 512,
        "height": 512,
        "sensor_pitch": 0,
    }

    # === ✅ 步驟 2: 啟動環境與 Agent ===
    env = HabitatEnvWrapper(sim_settings)
    
    # 建立視窗
    cv2.namedWindow("Navigation (Overlay)", cv2.WINDOW_AUTOSIZE)
    # cv2.namedWindow("Depth (Debug)", cv2.WINDOW_AUTOSIZE)
    # cv2.namedWindow("Semantic (Debug)", cv2.WINDOW_AUTOSIZE) # 語義通常太雜亂，可選


    # 設置初始位置 初始朝向 (面向路徑的第二個點)
    start_x, start_z = pixel_to_world(start[0], start[1], w, h, bounds)
    state = habitat_sim.AgentState()
    state.position = np.array([start_x, 0, start_z])
    
    dx = world_path[1][0] - world_path[0][0]
    dz = world_path[1][1] - world_path[0][1]
    init_yaw = math.atan2(-dx, -dz) 
    quat = np.array([0.0, math.sin(init_yaw / 2.0), 0.0, math.cos(init_yaw / 2.0)], dtype=np.float32)
    state.rotation = quat
    
    env.agent.set_state(state)
    print(f"[INFO] Agent placed at {state.position}, yaw = {math.degrees(init_yaw):.2f}°")

    # === ✅ 步驟 3: 執行 V3 統一控制器 ===
    video_path = f"{OUTPUT_PATH}/{target_class}_V3_controller.mp4"
    run_navigation_simple(env, world_path, target_mask)


    visualize_path_on_map(MAP_PATH, path, goal, start, target_class, OUTPUT_PATH, save_prefix="rrt_result_V3_initial")
    env.sim.close()
    print(f"[✅ DONE] V3 導航影片與地圖已輸出至 {OUTPUT_PATH}/")