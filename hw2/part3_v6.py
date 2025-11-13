import os
import cv2
import math
import json
import numpy as np
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb, d3_40_colors_hex
from PIL import Image
import matplotlib.pyplot as plt
from habitat_sim.utils.common import quat_from_angle_axis

# 確保 rrt_star.py 就在旁邊
from rrt_star import *
# 確保 part2.py 就在旁邊 (或您已將 rrt_star.py 獨立出來)
# from part2 import *
# ==========================================================
# Habitat 環境封裝
# ==========================================================
FORWARD_STEP = 0.01
ARRIVAL_JUDGE = FORWARD_STEP*1.5
TURN_ANGLE = 1
MAX_ACTIONS = 5000
FPS = 120
ARRIVAL_THRESH = 1 # 抵達目標的距離閾值 (米)

# === V3 控制器參數 ===
LOOKAHEAD_DISTANCE = 0.2        # (米) Pure Pursuit "胡蘿蔔" 的前瞻距離
OBSTACLE_CLEARANCE_DIST = 0.2  # (米) 認定為障礙物的深度閾值
PROACTIVE_THRESH_PIXELS = 3000  # (像素) 觸發主動避障的像素數量閾值
CORRECTION_ANGLE_THRESH = 2.0   # (度) 循跡時的角度容忍範圍
STUCK_LIMIT = 100               # (幀) 卡住/震盪多少幀後觸發「最後手段」
ESCAPE_BACKWARD_DIST = 0.2      # (米) 最後手段：後退距離
ESCAPE_TURN_ANGLE = 45          # (度) 最後手段：轉向角度

class HabitatEnvWrapper:
    def __init__(self, sim_settings, floor=1):
        self.cfg = self.make_simple_cfg(sim_settings)
        self.sim = habitat_sim.Simulator(self.cfg)
        self.agent = self.sim.initialize_agent(sim_settings["default_agent"])
        print("[DEBUG] action space keys:", list(self.cfg.agents[0].action_space.keys()))

    def reset_to(self, position, yaw_rad=0.0):
        state = habitat_sim.AgentState()
        snapped = self.sim.pathfinder.snap_point(np.array(position, dtype=np.float32))
        state.position = snapped
        state.rotation = quat_from_angle_axis(yaw_rad, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        self.agent.set_state(state)
        print(f"[RESET] Agent snapped to navmesh: {snapped}, yaw={math.degrees(yaw_rad):.1f}°")

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


# ==========================================================
# Util
# ==========================================================

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


def navigateAndSee(env_instance, action, target_mask_func, video_writer=None):
    """
    執行一步、即時顯示觀測，並返回觀測值。
    """
    observations = env_instance.step(action)

    # 準備合成顯示 (使用 overlay_mask 函數)
    rgb_img = observations["rgb"]
    mask = target_mask_func(observations) # 呼叫 target_mask
    vis_img = overlay_mask(rgb_img, mask) # 合成遮罩
    cv2.imshow("Navigation (Overlay)", transform_rgb_bgr(vis_img))
    # cv2.imshow("Depth (Debug)", transform_depth(observations["depth"]))
    # cv2.imshow("Semantic (Debug)", transform_rgb_bgr(observations["semantic"]))

    # 顯示攝影機姿態 (來自您的程式碼)
    # agent_state = env_instance.agent.get_state()
    # sensor_state = agent_state.sensor_states['color_sensor']
    # print("camera pose: x y z rw rx ry rz")
    # print(sensor_state.position[0],...) # (建議註解掉, 否則 log 會爆炸)

    if video_writer is not None:
        frame_resized = cv2.resize(transform_rgb_bgr(vis_img), (512, 512))
        video_writer.write(frame_resized)

    # 5. 刷新視窗 (!!!! 關鍵中的關鍵 !!!!)沒有這行, 圖片不會更新
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): # 允許按 'q' 提早結束
        raise KeyboardInterrupt("User pressed 'q' to quit.")

    return observations


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
        print(f"[ESCAPE_SCAN] 找不到開闊路徑 (最佳僅 {best_depth:.2f}m)，使用預設 60 度轉向。")
        action = "turn_left" if left_sectors_avg > right_sectors_avg else "turn_right"
        return action, 60.0 # Fallback

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
    
# 導航與影片錄製 
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

def face_goal(video_writer,env, goal, bounds, w, h, target_mask, FPS=120, TURN_ANGLE=1):
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
            obs = navigateAndSee(env, turn_action, target_mask,video_writer)
    else:
        obs = navigateAndSee(env, "turn_left", target_mask,video_writer)

    # ( ... 結尾動畫 ... )
    rgb = obs["rgb"].copy()
    mask = target_mask(obs)
    vis = overlay_mask(rgb, mask)
    vis = end_anime(vis)
    vis_bgr = transform_rgb_bgr(vis)
    cv2.imshow("Navigation (Overlay)", vis_bgr)
    if video_writer is not None:
        # 寫入約 3 秒動畫（FPS x 3 幀）
        for _ in range(int(FPS * 3)):
            video_writer.write(cv2.resize(vis_bgr, (512, 512)))

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

# === 在做 rrt_star_planning 之前，先放這些工具函式 ===
def compute_pixels_per_meter(bounds, w, h):
    SCALE_FACTOR = 10000 / 255.0
    xmin_pt, xmax_pt, zmin_pt, zmax_pt = bounds
    dxw_per_px = ((xmax_pt - xmin_pt) / float(w)) * SCALE_FACTOR
    dzw_per_px = ((zmax_pt - zmin_pt) / float(h)) * SCALE_FACTOR
    px_per_m_x = 1.0 / max(1e-9, dxw_per_px)
    px_per_m_z = 1.0 / max(1e-9, dzw_per_px)
    return 0.5 * (px_per_m_x + px_per_m_z)

def compute_clearance_map(binary_map):
    # binary_map: 255=free, 0=obs
    free = (binary_map > 0).astype(np.uint8)
    edt = cv2.distanceTransform(free, cv2.DIST_L2, 5).astype(np.float32)  # 單位：像素
    return edt

def inflate_free_space(binary_map, clearance_px):
    # 以「縮小可行區」來達成「障礙物膨脹」
    k = max(1, int(2*clearance_px + 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    free = (binary_map > 0).astype(np.uint8) * 255
    free_inflated = cv2.erode(free, kernel, iterations=1)
    return free_inflated

def find_nearest_safe_pixel(edt, start_xy, min_clear_px, rmax=120):
    x0, y0 = int(round(start_xy[0])), int(round(start_xy[1]))
    H, W = edt.shape
    if 0 <= x0 < W and 0 <= y0 < H and edt[y0, x0] >= min_clear_px:
        return (float(x0), float(y0))
    # 簡單的方形同心層掃描
    for r in range(1, rmax+1):
        for x in range(x0 - r, x0 + r + 1):
            for y in (y0 - r, y0 + r):
                if 0 <= x < W and 0 <= y < H and edt[y, x] >= min_clear_px:
                    return (float(x), float(y))
        for y in range(y0 - r + 1, y0 + r):
            for x in (x0 - r, x0 + r):
                if 0 <= x < W and 0 <= y < H and edt[y, x] >= min_clear_px:
                    return (float(x), float(y))
    return (float(x0), float(y0))

def count_near_pixels(depth, near_m=0.27, min_blob_area=200):
    """
    健壯版近物像素計數（替換舊版）：
      - 對 foot_depth 的 ROI 做中值濾波
      - 只計入面積 >= min_blob_area 的近物連通元件
    回傳:
      near_cnt: ROI 內接近物體的像素總數（已去小雜點）
      roi_area: ROI 總像素數
    """
    d = depth.astype(np.float32)
    h, w = d.shape
    # 下半部 + 中央窄窗（與你原本一致）
    r0, r1 = int(h * 0.55), int(h * 0.95)
    c0, c1 = int(w * 0.35), int(w * 0.65)

    # 邊界保護
    r0 = max(0, min(r0, h - 1)); r1 = max(0, min(r1, h))
    c0 = max(0, min(c0, w - 1)); c1 = max(0, min(c1, w))
    if r1 <= r0 or c1 <= c0:
        return 0, 1  # 避免除以 0

    roi = d[r0:r1, c0:c1].copy()
    roi_area = roi.size if roi.size > 0 else 1

    # 中值濾波（抑制鹽胡椒雜訊）
    roi = cv2.medianBlur(roi, 3)

    # 有效深度：去 NaN / Inf / 0
    valid = np.isfinite(roi) & (roi > 0.01)
    if not np.any(valid):
        return 0, roi_area

    # 近距二值圖
    near_mask = np.zeros_like(roi, dtype=np.uint8)
    near_mask[valid & (roi < float(near_m))] = 1

    # 去除小連通元件（雜點）
    num, labels, stats, _ = cv2.connectedComponentsWithStats(near_mask, connectivity=8)
    clean = np.zeros_like(near_mask, dtype=np.uint8)
    for i in range(1, num):  # 0 是背景
        if stats[i, cv2.CC_STAT_AREA] >= int(min_blob_area):
            clean[labels == i] = 1

    near_cnt = int(clean.sum())
    return near_cnt, roi_area


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

def pick_lookahead_index(path_world, start_idx, Ld_m=0.5):
    """從 start_idx 沿著路徑前進，累積到 Ld_m 的索引；不夠就回最後一點。"""
    d = 0.0
    for i in range(start_idx, len(path_world)-1):
        d += math.hypot(path_world[i+1][0]-path_world[i][0],
                        path_world[i+1][1]-path_world[i][1])
        if d >= Ld_m:
            return i+1
    return len(path_world)-1


from collections import deque

def closest_projection_on_polyline(P, poly):
    """回傳 (s, e_ct, seg_i, t, Q)：
       s: 從 poly[0] 起的弧長；e_ct: 橫向誤差(帶正負)；Q: 投影點座標
    """
    def dot(a,b): return a[0]*b[0] + a[1]*b[1]
    def sub(a,b): return (a[0]-b[0], a[1]-b[1])
    def norm(a):  return math.hypot(a[0], a[1])

    s_acc = 0.0
    best_d2, best_payload = float('inf'), None
    for i in range(len(poly)-1):
        A, B = poly[i], poly[i+1]
        AB = (B[0]-A[0], B[1]-A[1])
        AB2 = max(1e-9, dot(AB, AB))
        t  = max(0.0, min(1.0, dot((P[0]-A[0], P[1]-A[1]), AB)/AB2))
        Q  = (A[0] + AB[0]*t, A[1] + AB[1]*t)
        d2 = (P[0]-Q[0])**2 + (P[1]-Q[1])**2
        if d2 < best_d2:
            AP = (P[0]-A[0], P[1]-A[1])
            cross = AB[0]*AP[1] - AB[1]*AP[0]  # 決定左右符號
            e_ct = math.copysign(math.sqrt(max(0.0, d2)), cross)
            best_d2, best_payload = d2, (s_acc + norm(AB)*t, e_ct, i, t, Q)
        s_acc += norm(AB)
    return best_payload  # (s, e_ct, seg_i, t, Q)

def pure_pursuit_target(poly, seg_i, t, Ld):
    """從投影點起沿折線前進 Ld，回傳 (T, true_seg, t2)。"""
    # 以投影點為新起點建立 tail
    A, B = poly[seg_i], poly[seg_i+1]
    P0 = (A[0] + (B[0]-A[0])*t, A[1] + (B[1]-A[1])*t)
    tail = [P0] + poly[seg_i+1:]

    d = 0.0
    for j in range(len(tail)-1):
        P, Q = tail[j], tail[j+1]
        seg_len = math.hypot(Q[0]-P[0], Q[1]-P[1])
        if d + seg_len >= Ld:
            r = (Ld - d) / max(1e-9, seg_len)
            T = (P[0] + (Q[0]-P[0])*r, P[1] + (Q[1]-P[1])*r)
            true_seg = seg_i + j
            if j == 0:
                AB = (B[0]-A[0], B[1]-A[1])
                ABlen = max(1e-9, math.hypot(AB[0], AB[1]))
                t2 = t + (r*seg_len)/ABlen
            else:
                t2 = r
            return T, true_seg, max(0.0, min(1.0, t2))
        d += seg_len
    return poly[-1], len(poly)-2, 1.0  # 超過終點

def curvature_from(yaw, P, T):
    """近似曲率 for pure pursuit：回傳 (curv, alpha, Ld)"""
    dx, dz = T[0]-P[0], T[1]-P[1]
    desired_yaw = math.atan2(-dx, -dz)
    alpha = (desired_yaw - yaw + math.pi)%(2*math.pi) - math.pi
    Ld = max(1e-3, math.hypot(dx, dz))
    curv = 2.0*math.sin(alpha)/Ld
    return curv, alpha, Ld

def dynamic_lookahead(curv_abs, near_ratio, Lmin=0.12, Lmax=0.45, base=0.25, k1=0.35, k2=0.8):
    """彎越急/越擁擠 → Ld 越小"""
    Ld = base - k1*curv_abs - k2*near_ratio
    return max(Lmin, min(Lmax, Ld))

def polyline_length(poly):
    return sum(math.hypot(poly[i+1][0]-poly[i][0], poly[i+1][1]-poly[i][1]) for i in range(len(poly)-1))

def point_at_arclength(poly, s_target):
    """弧長取點：回傳 poly 上弧長 s_target 的點（夾到終點）"""
    s = 0.0
    for i in range(len(poly)-1):
        A, B = poly[i], poly[i+1]
        seg = math.hypot(B[0]-A[0], B[1]-A[1])
        if s + seg >= s_target:
            r = (s_target - s) / max(1e-9, seg)
            return (A[0] + (B[0]-A[0])*r, A[1] + (B[1]-A[1])*r)
        s += seg
    return poly[-1]

def has_line_of_sight(env, P, Q, step=0.1):
    """採樣直線，確認每點在 navmesh 上（簡易 LOS）。"""
    dx, dz = Q[0]-P[0], Q[1]-P[1]
    dist = max(1e-6, math.hypot(dx, dz))
    n = max(2, int(dist/step))
    for k in range(n+1):
        r = k / n
        x = P[0] + dx*r
        z = P[1] + dz*r
        p = np.array([x, 0.0, z], dtype=np.float32)
        if not env.sim.pathfinder.is_navigable(p):
            return False
    return True

def count_near_pixels_robust(depth, near_m=0.27, min_blob_area=200):
    """對 foot_depth 的 ROI 做中值濾波 + 去小雜點；回傳 (near_cnt, roi_area, near_ratio)。"""
    d = depth.astype(np.float32)
    h, w = d.shape
    r0, r1 = int(h*0.55), int(h*0.95)  # 下半部
    c0, c1 = int(w*0.35), int(w*0.65)  # 中央窄窗
    roi = d[r0:r1, c0:c1].copy()
    if roi.size == 0:
        return 0, 1, 0.0
    roi = cv2.medianBlur(roi, 3)
    valid = np.isfinite(roi) & (roi > 0.01)
    if not np.any(valid):
        return 0, roi.size, 0.0
    near_mask = np.zeros_like(roi, dtype=np.uint8)
    near_mask[valid & (roi < near_m)] = 1
    # 去除小雜點
    num, labels, stats, _ = cv2.connectedComponentsWithStats(near_mask, connectivity=8)
    clean = np.zeros_like(near_mask, dtype=np.uint8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_blob_area:
            clean[labels == i] = 1
    cnt = int(clean.sum())
    return cnt, roi.size, cnt / max(1, roi.size)


def navigate_simple_turn_move(
    env,                    # HabitatEnvWrapper
    world_path,            # [(x, z), ...] 世界座標路徑
    target_mask_func,      # target_mask 函數
    goal_pixel,           # 最終目標像素座標
    bounds,               # 座標邊界
    w, h,                 # 地圖寬高
    video_writer=None,    # 錄影器
    dist_tol=0.40,        # 抵達航點的距離閾值（米）
    yaw_tol_deg=4.0,      # 對準的角度容忍度（度）
    max_actions=5000      # 最大動作數
):
    """
    簡單的 Turn-then-Move 導航策略，逐個航點導航。
    類似 main.py 的 navigate_with_world_coords，但適配 part3_v6.py 的環境。
    """
    assert len(world_path) >= 2, "world_path 至少要有 2 個點"
    
    # 計算目標世界座標（用於最終抵達檢查）
    goalx, goalz = pixel_to_world(goal_pixel[0], goal_pixel[1], w, h, bounds)
    
    # 計算初始狀態
    astate = env.agent.get_state()
    pos = astate.position.copy()
    current_pos_xz = (pos[0], pos[2])
    current_yaw = get_yaw_from_quat(astate.rotation)
    
    # 計算期望 yaw 的輔助函數
    def get_desired_yaw(current_pos_xz, target_pos_xz):
        dx = target_pos_xz[0] - current_pos_xz[0]
        dz = target_pos_xz[1] - current_pos_xz[1]
        return math.atan2(-dx, -dz)
    
    yaw_tol = math.radians(yaw_tol_deg)
    
    # 初始化：從第二個航點開始（第一個是起點）
    path = world_path
    N = len(path)
    i = 1
    actions = 0
    
    # 獲取初始觀測（在設置初始狀態後）
    # 注意：初始狀態應該在調用此函數之前已經設置好
    obs = env.sim.get_sensor_observations()
    
    while i <= N and actions < max_actions:
        astate = env.agent.get_state()
        pos = astate.position.copy()
        current_pos_xz = (pos[0], pos[2])
        current_yaw = get_yaw_from_quat(astate.rotation)
        
        # 檢查是否已到達最終目標（優先檢查）
        final_dist = math.hypot(pos[0] - goalx, pos[2] - goalz)
        if final_dist <= dist_tol:
            print(f"Final goal reached! Distance: {final_dist:.2f}m")
            face_goal(video_writer, env, goal_pixel, bounds, w, h, target_mask_func)
            print("[DONE] Arrived goal.")
            return
        
        # 如果已抵達所有航點，朝向最終目標導航
        if i >= N:
            # 朝向最終目標
            desired_yaw = get_desired_yaw(current_pos_xz, (goalx, goalz))
            dyaw = wrap_to_pi(desired_yaw - current_yaw)
            
            print(f"[nav] pos=({pos[0]:.2f},{pos[2]:.2f}) -> final_goal=({goalx:.2f},{goalz:.2f}). dist={final_dist:.2f}, dyaw={math.degrees(dyaw):.1f}°")
            
            if abs(dyaw) > yaw_tol:
                # 尚未對準最終目標，轉向
                action_to_take = "turn_left" if dyaw > 0 else "turn_right"
            else:
                # 已對準，前進
                action_to_take = "move_forward"
        else:
            # 正常航點導航
            target_pos_xz = path[i]
            
            # 計算距離和角度差
            dx_world = target_pos_xz[0] - current_pos_xz[0]
            dz_world = target_pos_xz[1] - current_pos_xz[1]
            dist_to_target = math.hypot(dx_world, dz_world)
            desired_yaw = get_desired_yaw(current_pos_xz, target_pos_xz)
            dyaw = wrap_to_pi(desired_yaw - current_yaw)
            
            print(f"[nav] pos=({pos[0]:.2f},{pos[2]:.2f}) -> target[{i}]=({target_pos_xz[0]:.2f},{target_pos_xz[1]:.2f}). dist={dist_to_target:.2f}, dyaw={math.degrees(dyaw):.1f}°")
            
            # 決策邏輯
            action_to_take = None
            
            if dist_to_target <= dist_tol:
                # 情況 A: 已抵達航點
                print(f"Reached waypoint {i}. Moving to next.")
                i += 1
                continue  # 繼續下一個循環，檢查是否到達最終目標
            
            elif abs(dyaw) > yaw_tol:
                # 情況 B: 尚未對準
                action_to_take = "turn_left" if dyaw > 0 else "turn_right"
            
            else:
                # 情況 C: 已對準，尚未抵達
                action_to_take = "move_forward"
        
        # 執行動作
        if action_to_take:
            obs = navigateAndSee(env, action_to_take, target_mask_func, video_writer)
            actions += 1
            
            # 碰撞檢測：檢查是否卡住
            if action_to_take == "move_forward":
                new_pos = env.agent.get_state().position
                if np.linalg.norm(new_pos - pos) < 1e-4:
                    print("[WARN] No progress, possibly stuck. Trying to turn...")
                    obs = navigateAndSee(env, "turn_right", target_mask_func, video_writer)
                    obs = navigateAndSee(env, "turn_right", target_mask_func, video_writer)
                    actions += 2
    
    if actions >= max_actions:
        print("[TIMEOUT] Max actions reached; stopping.")
    else:
        print("[WARN] Navigation loop ended unexpectedly.")


def run_navigation(env,
                world_path,           # [(x,z), ...]
                bounds,               # 給 face_goal 用
                target_mask_func,     # target_mask
                goal_pixel,           # 最終目標（像素）
                w, h,                 # 地圖寬高
                video_writer=None,
                lookahead_m=0.2,      # 已改為動態 Ld，這個參數不再直接使用
                turn_on_deg=8.0,
                turn_off_deg=3.0,
                near_m=0.27,
                near_ratio_thr=0.02,  # 依你地圖噪聲可調
                arrival_thresh=ARRIVAL_THRESH,
                max_actions=MAX_ACTIONS):

    assert len(world_path) >= 2, "world_path 至少要有 2 個點"
    goalx, goalz = pixel_to_world(goal_pixel[0], goal_pixel[1], w, h, bounds)

    # 方向校正
    TURN_LEFT_SIGN = calibrate_turn_sign(env)
    def pick_turn_action(yaw_err_rad):
        going_left = (yaw_err_rad > 0)
        if TURN_LEFT_SIGN < 0:
            going_left = not going_left
        return "turn_left" if going_left else "turn_right"

    # 啟動：取一次觀測，避免第一幀無資料
    obs = env.step("turn_right"); obs = env.step("turn_left")

    actions = 0
    turn_state = None
    progress_hist = deque()   # 存 s（沿路弧長）
    total_len = polyline_length(world_path)

    # rejoin 目標（卡住時暫時指定）
    force_target = None
    FORCE_T_STEPS = 0

    # 轉向/前進「批處理」步數
    TURN_BATCH = 3
    MOVE_BATCH = 3

    # 進度卡住偵測
    STUCK_WINDOW = 50       # 看最近 30 步
    STUCK_DELTA_S = 0.05    # 弧長進度 < 5 cm → 視為卡住
    REJOIN_AHEAD = 0.8      # 往前 rejoin 0.8 m

    # 統一的門檻（稍微比參數高一點，因為動態 Ld 會穩定很多）
    TURN_ON  = max(10.0, turn_on_deg)
    TURN_OFF = max(4.0,  turn_off_deg)

    # === 逃脫模式狀態 ===
    escape_plan = None  # {"stage": "turn"/"back"/"fwd", "turn_action":..., "turn_steps_left":..., "back_steps_left":..., "fwd_steps_left":...}
    ESCAPE_TURN_OVERSHOOT_DEG = 5.0

    # 依全域常數換算步數
    ESCAPE_BACK_STEPS = max(2, int(ESCAPE_BACKWARD_DIST / float(FORWARD_STEP)))
    ESCAPE_FWD_PROBE_STEPS = 0 # max(4, int(0.35 / float(FORWARD_STEP)))  # 探走 ~0.35m
    ESCAPE_TURN_BATCH = 20  # 連續轉幾步再刷新一次畫面

    # 轉向鎖（避免剛轉完又反向抖動）
    ESCAPE_LOCK_STEPS = 25
    escape_lock_until = -1  # actions index

    # 小工具：連續執行 N 步 action
    def do_steps(act, n):
        nonlocal actions, obs
        for _ in range(max(0, int(n))):
            obs = navigateAndSee(env, act, target_mask_func,video_writer)
            actions += 1

    while actions < max_actions:
        # 狀態
        astate = env.agent.get_state()
        pos = astate.position.copy()
        yaw = get_yaw_from_quat(astate.rotation)
        P = (pos[0], pos[2])

        # 抵達檢查（朝最終世界座標）
        if math.hypot(P[0]-goalx, P[1]-goalz) < arrival_thresh:
            face_goal(video_writer,env, goal_pixel, bounds, w, h, target_mask_func)
            print("[DONE] Arrived goal.")
            return

        # 若正在執行逃脫計畫，優先完成（避免被其他邏輯打斷）
        if escape_plan is not None:
            stage = escape_plan["stage"]

            if stage == "turn":
                # 大角度連續轉，直到轉完
                left = min(ESCAPE_TURN_BATCH, escape_plan["turn_steps_left"])
                do_steps(escape_plan["turn_action"], left)
                escape_plan["turn_steps_left"] -= left
                if escape_plan["turn_steps_left"] <= 0:
                    escape_plan["stage"] = "back"
                continue

            elif stage == "back":
                left = min(5, escape_plan["back_steps_left"])
                do_steps("move_backward", left)
                escape_plan["back_steps_left"] -= left
                if escape_plan["back_steps_left"] <= 0:
                    escape_plan["stage"] = "fwd"
                continue

            elif stage == "fwd":
                # 向前探走，但要確認下一步可走
                left = min(5, escape_plan["fwd_steps_left"])
                for _ in range(left):
                    if actions >= max_actions: break
                    astate = env.agent.get_state()
                    pos = astate.position.copy()
                    yaw = get_yaw_from_quat(astate.rotation)
                    forward_dir = np.array([-math.sin(yaw), 0.0, -math.cos(yaw)], dtype=np.float32)
                    candidate = pos + forward_dir * float(FORWARD_STEP)
                    if env.sim.pathfinder.is_navigable(candidate):
                        obs = navigateAndSee(env, "move_forward", target_mask_func,video_writer)
                    else:
                        # 前面真的走不了，就先結束逃脫，交回主控
                        break
                    actions += 1
                    escape_plan["fwd_steps_left"] -= 1

                if escape_plan["fwd_steps_left"] <= 0:
                    escape_plan = None
                    # 設定轉向冷卻鎖，避免馬上反向
                    escape_lock_until = actions + ESCAPE_LOCK_STEPS
                    turn_state = None
                    progress_hist.clear()
                continue

        # 近距 ROI（健壯化）
        near_cnt, roi_area, near_ratio = count_near_pixels_robust(obs["foot_depth"], near_m=near_m)
        FRONT_BLOCK = (near_ratio > near_ratio_thr)

        # 投影 + 弧長進度
        s, e_ct, seg_i, t, Q = closest_projection_on_polyline(P, world_path)
        progress_hist.append(s)
        if len(progress_hist) > STUCK_WINDOW:
            progress_hist.popleft()

        # 目標點（動態 Ld）
        # 先粗估曲率
        T0, _, _ = pure_pursuit_target(world_path, seg_i, t, 0.25)
        curv0, alpha0, _ = curvature_from(yaw, P, T0)
        Ld = dynamic_lookahead(abs(curv0), near_ratio)

        # 如果有 rejoin 強制目標，優先用它幾回合
        if force_target is not None and FORCE_T_STEPS > 0:
            T = force_target
            curv, alpha, _ = curvature_from(yaw, P, T)
            FORCE_T_STEPS -= 1
        else:
            T, _, _ = pure_pursuit_target(world_path, seg_i, t, Ld)
            curv, alpha, _ = curvature_from(yaw, P, T)
            force_target = None
            FORCE_T_STEPS = 0

        angle_deg = abs(math.degrees(alpha))

        # 轉向狀態（雙門檻 + 轉向鎖）
        if actions < escape_lock_until:
            if turn_state is None and angle_deg > max(TURN_ON, 25.0):
                turn_state = 'turning'
        else:
            if turn_state is None and angle_deg > TURN_ON:
                turn_state = 'turning'

        # === 決策（分離轉向 / 前進；採批處理以抑制抖動） ===
        step_consumed = False
        if turn_state == 'turning':
            act = pick_turn_action(alpha)
            for _ in range(TURN_BATCH):
                if actions >= max_actions: break
                obs = navigateAndSee(env, act, target_mask_func,video_writer)
                actions += 1
            if angle_deg < TURN_OFF:
                turn_state = None
            step_consumed = True
        else:
            if FRONT_BLOCK:
                # 不貿然前進，僅微轉。動態 Ld 已縮短。
                act = pick_turn_action(alpha)
                if actions < max_actions:
                    obs = navigateAndSee(env, act, target_mask_func,video_writer)
                    actions += 1
                step_consumed = True
            else:
                # 連續前進以避免走走停停
                for _ in range(MOVE_BATCH):
                    if actions >= max_actions: break
                    # 前瞻一步可行性（navmesh）
                    forward_dir = np.array([-math.sin(yaw), 0.0, -math.cos(yaw)], dtype=np.float32)
                    candidate = pos + forward_dir * float(FORWARD_STEP)
                    if not env.sim.pathfinder.is_navigable(candidate):
                        # 前一步無法走，改為微轉
                        act = pick_turn_action(alpha)
                        obs = navigateAndSee(env, act, target_mask_func,video_writer)
                    else:
                        obs = navigateAndSee(env, "move_forward", target_mask_func,video_writer)
                    actions += 1
                step_consumed = True

        if not step_consumed:
            # 保底（理論上不會走到）
            obs = navigateAndSee(env, "move_forward", target_mask_func,video_writer)
            actions += 1

        # === 進度型卡住偵測 → rejoin 到前方可視點 ===
        if len(progress_hist) >= STUCK_WINDOW:
            delta_s = progress_hist[-1] - progress_hist[0]
            if delta_s < STUCK_DELTA_S:
                # 卡住，先嘗試 rejoin 到前方 0.8m 的可視點
                rejoin_s = min(total_len, s + REJOIN_AHEAD)
                RJ = point_at_arclength(world_path, rejoin_s)
                if has_line_of_sight(env, P, RJ):
                    force_target = RJ
                    FORCE_T_STEPS = 10   # 強制朝向該目標幾回合
                    turn_state = 'turning'  # 先讓它朝向，再走
                    progress_hist.clear()   # 重置觀測窗
                    print(f"[REJOIN] LOS ok → set temporary target {RJ}")
                else:
                    # LOS 也不通：長轉到空曠扇區 → 小幅後退 → 短距探走
                    turn_act, ang = find_escape_route(obs["depth"], hfov_deg=90.0, num_sectors=5)
                    if turn_act is None:
                        turn_act = pick_turn_action(+0.2)
                        ang = 35.0
                    need_deg = ang + ESCAPE_TURN_OVERSHOOT_DEG
                    turn_steps = max(3, int(math.ceil(need_deg / float(TURN_ANGLE))))
                    escape_plan = {
                        "stage": "turn",
                        "turn_action": turn_act,
                        "turn_steps_left": turn_steps,
                        "back_steps_left": ESCAPE_BACK_STEPS,
                        "fwd_steps_left": ESCAPE_FWD_PROBE_STEPS,
                    }
                    print(f"[ESCAPE] {turn_act} ~{ang:.1f}° → back {ESCAPE_BACK_STEPS} → fwd {ESCAPE_FWD_PROBE_STEPS}")
                    progress_hist.clear()
                    continue  # 交給上面的 escape_plan 區塊處理

        # Debug
        print(f"[DBG] s={s:.2f}  e_ct={e_ct:.2f}  Ld={Ld:.2f}  ang={angle_deg:.1f}°  "
            f"near={near_ratio:.3f}  turn={'Y' if turn_state else 'N'}  "
            f"forceT={'Y' if force_target is not None else 'N'}")

    print("[TIMEOUT] Max actions reached; stopping.")


import math, random
import numpy as np
from collections import defaultdict

# ====== 你可以放到 rrt_world.py 或直接黏到專案 ======

# ---------- 幫手：NavMesh 取樣 ----------
def random_navigable_point(sim):
    # Habitat 自帶隨機可走點（若沒有此API，就自己在 bounds 內多次亂抽 + is_navigable）
    return sim.pathfinder.get_random_navigable_point()

def snap(sim, x, z):
    p = np.array([x, 0.0, z], dtype=np.float32)
    sp = sim.pathfinder.snap_point(p)
    return float(sp[0]), float(sp[2])

# ---------- 幫手：單點「帶安全距離」檢查 ----------
def _is_locally_navigable(sim, q3, eps=0.03):
    """
    用 snap_point 檢查點是否真在當前 NavMesh 上：
    - 先把 q3 貼到 NavMesh（取得 sp）
    - 若 sp 不是可走 or 與 q3 在水平距離上偏移超過 eps，視為不可走
    """
    sp = sim.pathfinder.snap_point(q3)
    if not sim.pathfinder.is_navigable(sp):
        return False
    dx = float(sp[0] - q3[0]); dz = float(sp[2] - q3[2])
    return (dx*dx + dz*dz) <= (eps*eps)

def point_has_clearance(sim, x, z, clearance_m, dirs=8, eps=0.03):
    """
    修正版：每個取樣點使用「當地地板高度」，並以 snap 檢查是否真在原處。
    """
    # 先把中心點貼到 NavMesh，拿到正確高度 y
    c3 = np.array([x, 0.0, z], dtype=np.float32)
    c3 = sim.pathfinder.snap_point(c3)
    if not sim.pathfinder.is_navigable(c3):
        return False

    # 中心點自己要能站
    if not _is_locally_navigable(sim, c3, eps=eps):
        return False

    # 半徑採樣：每個offset點都用同一層的「當地高度」檢查
    for k in range(max(1, dirs)):
        th = 2.0*math.pi*k/max(1, dirs)
        q3 = np.array([c3[0] + clearance_m*math.cos(th),
                       c3[1],  # 先用中心高度，snap時會自動對齊
                       c3[2] + clearance_m*math.sin(th)], dtype=np.float32)
        if not _is_locally_navigable(sim, q3, eps=eps):
            return False
    return True


def find_nearest_clear_point(sim, start_xy, max_radius=2.0, step=0.05, clearance_m=0.25):
    sx, sz = start_xy
    for r in np.arange(0, max_radius, step):
        for th in np.linspace(0, 2*math.pi, 16, endpoint=False):
            nx = sx + r*math.cos(th)
            nz = sz + r*math.sin(th)
            if point_has_clearance(sim, nx, nz, clearance_m, dirs=4):
                return (nx, nz)
    return None


# ---------- 幫手：線段「帶安全距離」檢查（掃掠圓盤） ----------
def edge_collision_free_with_clearance(sim, A, B, clearance_m, step_m=0.05, side_samples=2, eps=0.03):
    """
    修正版：沿邊中心 + 兩側法向偏移以「局部可走檢查」判斷。
    每個取樣點都 snap 並用 eps 約束，避免被拉到遠處造成誤判。
    """
    ax, az = A; bx, bz = B
    dx, dz = bx-ax, bz-az
    dist = max(1e-9, math.hypot(dx, dz))
    tx, tz = dx/dist, dz/dist
    nx, nz = -tz, tx

    n_steps = max(2, int(math.ceil(dist/step_m)))
    offsets = [0.0]
    if clearance_m > 0:
        offsets += [+1.0, -1.0]
        if side_samples >= 4:
            offsets += [+0.5, -0.5]

    for k in range(n_steps+1):
        r = k / n_steps
        cx = ax + dx*r
        cz = az + dz*r

        # 中心
        c3 = np.array([cx, 0.0, cz], dtype=np.float32)
        c3 = sim.pathfinder.snap_point(c3)
        if not _is_locally_navigable(sim, c3, eps=eps):
            return False

        # 兩側
        for s in offsets[1:]:
            ox = c3[0] + s*clearance_m*nx
            oz = c3[2] + s*clearance_m*nz
            q3 = np.array([ox, c3[1], oz], dtype=np.float32)
            if not _is_locally_navigable(sim, q3, eps=eps):
                return False

    return True

def find_nearest_safe_world_point(sim, x, z, clearance_m, rmax=1.2, dr=0.05, dirs=16, eps=0.03):
    """
    以同心圓由近到遠搜尋最近的安全點；回傳 (x,z) 或 None。
    """
    # 先用貼地中心
    c3 = sim.pathfinder.snap_point(np.array([x, 0.0, z], dtype=np.float32))
    if point_has_clearance(sim, c3[0], c3[2], clearance_m, dirs=max(8, dirs), eps=eps):
        return (float(c3[0]), float(c3[2]))

    R = 0.0
    while R <= rmax + 1e-9:
        for k in range(dirs):
            th = 2.0*math.pi*k/dirs
            q3 = np.array([c3[0] + R*math.cos(th), c3[1], c3[2] + R*math.sin(th)], dtype=np.float32)
            if point_has_clearance(sim, float(q3[0]), float(q3[2]), clearance_m, dirs=max(8, dirs), eps=eps):
                return (float(q3[0]), float(q3[2]))
        R += dr
    return None


# ---------- RRT（世界座標版；可改 RRT* 但先給易用版） ----------
class RRTWorld:
    def __init__(self, sim, step_m=0.25, goal_sample_rate=0.15,
                 max_iter=8000, clearance_m=0.25, node_retry=100):
        self.sim = sim
        self.step_m = step_m
        self.goal_rate = goal_sample_rate
        self.max_iter = max_iter
        self.clearance_m = clearance_m
        self.node_retry = node_retry

        self.nodes = []       # [(x,z)]
        self.parent = {}      # child_idx -> parent_idx

    def _sample_free_with_clearance(self, bounds=None):
        # 簡單策略：從 navmesh 取隨機點，直到通過 clearance 檢查
        for _ in range(self.node_retry):
            p = self.sim.pathfinder.get_random_navigable_point()
            x, z = float(p[0]), float(p[2])
            if point_has_clearance(self.sim, x, z, self.clearance_m, dirs=4):
                return (x, z)
        return None

    @staticmethod
    def _nearest(nodes, q):
        qx, qz = q
        best_i, best_d2 = -1, 1e18
        for i, (x, z) in enumerate(nodes):
            d2 = (x-qx)*(x-qx) + (z-qz)*(z-qz)
            if d2 < best_d2:
                best_i, best_d2 = i, d2
        return best_i

    def _steer(self, from_xy, to_xy):
        fx, fz = from_xy; tx, tz = to_xy
        dx, dz = tx-fx, tz-fz
        dist = math.hypot(dx, dz)
        if dist <= self.step_m:
            return (tx, tz)
        r = self.step_m / dist
        return (fx + dx*r, fz + dz*r)

    def plan(self, start_xy, goal_xy, goal_thresh_m=0.30):
        # 起終點 snap
        s3 = self.sim.pathfinder.snap_point(np.array([start_xy[0], 0.0, start_xy[1]], dtype=np.float32))
        g3 = self.sim.pathfinder.snap_point(np.array([goal_xy[0],  0.0, goal_xy[1]],  dtype=np.float32))

        # 起終點「就近修正到有安全距」
        s_safe = find_nearest_safe_world_point(self.sim, float(s3[0]), float(s3[2]), self.clearance_m)
        g_safe = find_nearest_safe_world_point(self.sim, float(g3[0]), float(g3[2]), self.clearance_m)
        if s_safe is None:
            raise RuntimeError("Start 附近找不到滿足安全距的可走點（可調小 clearance 或換起點）")
        if g_safe is None:
            raise RuntimeError("Goal 附近找不到滿足安全距的可走點（可調小 clearance 或換目標）")
        sx, sz = s_safe; gx, gz = g_safe

        # 連通性快檢：不同 island 直接回報（避免白跑 RRT）
        sp = habitat_sim.ShortestPath()
        sp.requested_start = np.array([sx, 0.0, sz], dtype=np.float32)
        sp.requested_end   = np.array([gx, 0.0, gz], dtype=np.float32)
        found = self.sim.pathfinder.find_path(sp)
        if (not found) or (not np.isfinite(sp.geodesic_distance)) or (sp.geodesic_distance > 1e8):
            raise RuntimeError("Start 與 Goal 不在同一連通區（可能不同樓層或被牆阻隔）。請換一個目標或調整起點。")


        self.nodes = [(sx, sz)]
        self.parent = {}

        for it in range(self.max_iter):
            # 目標偏置取樣
            if random.random() < self.goal_rate:
                qrand = (gx, gz)
            else:
                qrand = self._sample_free_with_clearance()
                if qrand is None:
                    continue

            # 找最近節點 & 向外延伸
            idx = self._nearest(self.nodes, qrand)
            qnear = self.nodes[idx]
            qnew  = self._steer(qnear, qrand)

            # 邊檢查：帶 clearance 的掃掠測試
            if edge_collision_free_with_clearance(self.sim, qnear, qnew, self.clearance_m, step_m=self.step_m*0.5):
                # snap 一下更穩
                qnew = snap(self.sim, *qnew)
                # 新節點也要有 clearance
                if not point_has_clearance(self.sim, qnew[0], qnew[1], self.clearance_m, dirs=4):
                    continue
                self.nodes.append(qnew)
                self.parent[len(self.nodes)-1] = idx

                # 是否到目標
                if math.hypot(qnew[0]-gx, qnew[1]-gz) <= goal_thresh_m:
                    # 追加一段到 goal（如需）
                    if edge_collision_free_with_clearance(self.sim, qnew, (gx, gz), self.clearance_m, step_m=self.step_m*0.5):
                        self.nodes.append((gx, gz))
                        self.parent[len(self.nodes)-1] = len(self.nodes)-2
                    return self._reconstruct_path(len(self.nodes)-1)

        return None  # 失敗

    def _reconstruct_path(self, last_idx):
        path = []
        i = last_idx
        while i in self.parent:
            path.append(self.nodes[i])
            i = self.parent[i]
        path.append(self.nodes[0])
        path.reverse()
        return path

# ----------（可選）用 Geodesic 路徑把 RRT 的折線「貼緊 NavMesh」 ----------
def geodesic_bridge(sim, A, B, step=0.15):
    """使用正確的 ShortestPath API 來獲取兩點間的 geodesic 路徑"""
    sp = habitat_sim.ShortestPath()
    sp.requested_start = np.array([A[0], 0.0, A[1]], dtype=np.float32)
    sp.requested_end = np.array([B[0], 0.0, B[1]], dtype=np.float32)
    found = sim.pathfinder.find_path(sp)
    
    if found and len(sp.points) >= 2:
        pts = [(p[0], p[2]) for p in sp.points]
        # 重採樣成等距點，方便控制器（步長 step）
        out = [pts[0]]
        for i in range(len(pts)-1):
            dx = pts[i+1][0]-pts[i][0]
            dz = pts[i+1][1]-pts[i][1]
            seg = max(1e-6, math.hypot(dx, dz))
            k = max(1, int(seg/step))
            for t in range(1, k+1):
                r = t/k
                out.append((pts[i][0]+dx*r, pts[i][1]+dz*r))
        return out
    return [A, B]  # 如果找不到路徑，直接返回兩點

def geodesic_refine_polyline(sim, coarse_path, step=0.15):
    refined = [coarse_path[0]]
    for i in range(len(coarse_path)-1):
        segment = geodesic_bridge(sim, coarse_path[i], coarse_path[i+1], step=step)
        refined.extend(segment[1:])
    return refined


# ==========================================================
# 主程式
# ==========================================================
if __name__ == "__main__":
    print("=== HW2 Part3 (V3) — World-Coord RRT on NavMesh ===")

    # 基本路徑與輸入
    DPI = 100
    currdir = os.path.dirname(os.path.abspath(__file__))
    MAP_PATH    = os.path.join(currdir, f"map{DPI}.png")
    EXCEL_PATH  = os.path.join(currdir, "color_coding_semantic_segmentation_classes.xlsx")
    BOUNDS_PATH = os.path.join(currdir, "coordinate_bounds.json")
    OUTPUT_PATH = os.path.join(currdir, "part3OUTPUT")
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    
    # 讀圖與 bounds（僅用來做像素<->世界的轉換與可視化）
    map_img = cv2.imread(MAP_PATH)
    if map_img is None:
        raise FileNotFoundError(MAP_PATH)
    h, w = map_img.shape[:2]
    bounds = load_bounds(BOUNDS_PATH)

    # 語意表 & 有哪些類別真實出現在地圖
    color_map = load_semantic_table(EXCEL_PATH)
    id_map    = load_semantic_ID_table(EXCEL_PATH)
    unique_colors = np.unique(map_img.reshape(-1, 3), axis=0)
    unique_set    = {tuple(c.tolist()) for c in unique_colors}
    available_classes = []
    for name, rgb in color_map.items():
        bgr = tuple(reversed(rgb))
        if bgr in unique_set:
            available_classes.append(name)
    if not available_classes:
        raise RuntimeError("❌ 地圖上找不到任何語意類別，請確認 map 與 color map 是否一致。")
    print(f"[INFO] 可用的目標類別：{available_classes}")

    # 使用者選目標 + 顯示所有候選 goal，點選後回傳最近的一個像素座標
    target_class = input(f"請輸入目標類別 {available_classes}: ").strip().lower()
    if target_class not in available_classes:
        raise ValueError(f"⚠️ '{target_class}' 不在清單中。")

    goals_list, goal_mask = find_all_object_instances(MAP_PATH, color_map, target_class)
    if not goals_list or goal_mask is None:
        raise ValueError(f"⚠️ 找不到目標類別 '{target_class}' 的任何區域。")

    # 互動式挑選 goal（像素）
    goal_px = visualize_multiple_goals(MAP_PATH, goals_list, target_class)

    # 互動式挑選 start（像素）
    start_px = select_start(MAP_PATH, goal_px)

    # 像素→世界（(x,z)）
    start_world = pixel_to_world(start_px[0], start_px[1], w, h, bounds)
    goal_world  = pixel_to_world(goal_px[0],  goal_px[1],  w, h, bounds)

    # === 啟動 Habitat 環境 ===
    sim_settings = {
        "scene": os.path.join(currdir, "replica_v1/apartment_0/habitat/mesh_semantic.ply"),
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
    navmesh_path = os.path.join(currdir, "replica_v1/apartment_0/habitat/mesh_semantic.navmesh")
    loaded = env.sim.pathfinder.load_nav_mesh(navmesh_path)
    print("[NAVMESH] loaded:", loaded)
    # === 世界座標 RRT 規劃（帶安全距離） ===
    SAFE_CLEARANCE_M = 0.2   # 你想離牆/家具的距離（m）
    RRT_STEP_M       = 0.05   # RRT 延伸步長（m）
    GOAL_BIAS        = 0.20   # 目標偏置機率
    GOAL_THRESH_M    = 0.50   # 視為到達 goal 的半徑（m）

    rrt = RRTWorld(
        env.sim,
        step_m=RRT_STEP_M,
        goal_sample_rate=GOAL_BIAS,
        max_iter=12000,
        clearance_m=SAFE_CLEARANCE_M,
        node_retry=200
    )

    print(f"[PLAN] start_world={start_world}, goal_world={goal_world}")
    path = habitat_sim.ShortestPath()
    path.requested_start = np.array([start_world[0], 0.0, start_world[1]], np.float32)
    path.requested_end   = np.array([goal_world[0], 0.0, goal_world[1]], np.float32)
    found = env.sim.pathfinder.find_path(path)
    print("[DEBUG] path found:", found, " length:", path.geodesic_distance)

    print("[DEBUG] start navigable:", env.sim.pathfinder.is_navigable(np.array([start_world[0], 0.0,start_world[1]], np.float32)))
    print("[DEBUG] goal navigable:", env.sim.pathfinder.is_navigable(np.array([goal_world[0], 0.0, goal_world[1]], np.float32)))

    path_world = rrt.plan(start_world, goal_world, goal_thresh_m=GOAL_THRESH_M)
    if path_world is None or len(path_world) < 2:
        raise RuntimeError("❌ RRT 規劃失敗（在當前 clearance 與迭代上限下）。請放寬 SAFE_CLEARANCE_M 或調整 goal。")

    # 用 NavMesh geodesic 把折線貼地並等距重採樣，利於控制器
    path_world = geodesic_refine_polyline(env.sim, path_world, step=0.15)
    print(f"[INFO] 規劃成功：世界路徑點數 = {len(path_world)}")

    # === 視覺化：把世界路徑投影回像素圖上存檔 ===
    path_px = [world_to_pixel(x, z, w, h, bounds) for (x, z) in path_world]
    vis = map_img.copy()
    for i in range(len(path_px)-1):
        p1 = (int(path_px[i][0]),   int(path_px[i][1]))
        p2 = (int(path_px[i+1][0]), int(path_px[i+1][1]))
        cv2.line(vis, p1, p2, (0, 0, 255), 2)
    cv2.circle(vis, (int(start_px[0]), int(start_px[1])), 6, (0,255,0), -1)
    cv2.circle(vis, (int(goal_px[0]),  int(goal_px[1])),  6, (255,0,0), -1)
    cv2.imwrite(os.path.join(OUTPUT_PATH, f"world_rrt_path_{target_class}.png"), vis)
    print(f"[INFO] 已輸出像素地圖視覺化：{OUTPUT_PATH}/world_rrt_path_{target_class}.png")

    # === 初始化 agent 位置和朝向：直接設置，面向路徑第一個航點 ===
    # 使用 snap_point 獲取正確的高度，但直接設置位置和朝向
    start_pos_3d = np.array([start_world[0], 0.0, start_world[1]], dtype=np.float32)
    snapped = env.sim.pathfinder.snap_point(start_pos_3d)
    
    # 計算初始朝向：面向路徑的第一個航點（第二個點）
    if len(path_world) >= 2:
        dx = path_world[1][0] - start_world[0]
        dz = path_world[1][1] - start_world[1]
        init_yaw = math.atan2(-dx, -dz)
    else:
        init_yaw = 0.0
    
    # 直接設置位置和朝向（使用 snap 後的高度）
    st = habitat_sim.AgentState()
    st.position = snapped  # 使用 snap 後的位置（包含正確的高度）
    st.rotation = quat_from_angle_axis(init_yaw, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    env.agent.set_state(st)
    
    print(f"[RESET] Agent position set to: {snapped}, yaw={math.degrees(init_yaw):.1f}°")
    
    # 檢查朝向品質
    s = env.agent.get_state()
    yaw = get_yaw_from_quat(s.rotation)
    if len(path_world) >= 2:
        fwd = np.array([-math.sin(yaw), 0.0, -math.cos(yaw)])
        to_next = np.array([dx, 0.0, dz])
        cosang = float(np.dot(fwd, to_next) / (np.linalg.norm(fwd)*np.linalg.norm(to_next)+1e-9))
        print(f"[CHECK] cos(angle to next waypoint) = {cosang:.3f}")

    # === 錄影器設定 ===
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = os.path.join(OUTPUT_PATH, f"{target_class}_navigation.mp4")
    video_writer = cv2.VideoWriter(video_path, fourcc, FPS, (512, 512))
    cv2.namedWindow("Navigation (Overlay)", cv2.WINDOW_AUTOSIZE)

    # === 導航（吃世界座標路徑）- 使用簡單的 Turn-then-Move 策略 ===
    navigate_simple_turn_move(
        env,
        world_path=path_world,
        target_mask_func=target_mask,  # 你的語義遮罩函式
        goal_pixel=goal_px,            # face_goal 仍需像素座標
        bounds=bounds,
        w=w, h=h,
        video_writer=video_writer,
        dist_tol=0.10,                 # 抵達航點的距離閾值（米）
        yaw_tol_deg=4.0,               # 對準的角度容忍度（度）
        max_actions=MAX_ACTIONS
    )

    # 收尾
    env.sim.close()
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"[🎥 SAVED] Navigation video saved at {video_path}")
    print(f"[✅ DONE] World-Coord RRT + V3 Controller finished. Outputs in {OUTPUT_PATH}/")