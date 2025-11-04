# part3_v4_backup.py 保留所有多餘程式的v4版本
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
    ESCAPE_FWD_PROBE_STEPS = max(4, int(0.35 / float(FORWARD_STEP)))  # 探走 ~0.35m
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


# ==========================================================
# 主程式
# ==========================================================
if __name__ == "__main__":
    
    print("=== HW2 Part3 (V3 - 統一控制器版本) ===")
    DPI = 100
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
    h, w = map_img_gray.shape[:2]
    _, binary_map = cv2.threshold(map_img_gray, 240, 255, cv2.THRESH_BINARY)
    bounds = load_bounds(BOUNDS_PATH)

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
    goal = visualize_multiple_goals(MAP_PATH, goals_list, target_class)
    
    # 2) 依『公尺→像素』決定安全距離（建議 0.25~0.35m）
    px_per_m = compute_pixels_per_meter(bounds, w, h)
    CLEARANCE_M = 0.12   # 想離牆/椅子 30 公分
    CLEAR_PX   = int(round(CLEARANCE_M * px_per_m))
    print(f"[SAFE] px_per_m≈{px_per_m:.2f} → CLEAR_PX={CLEAR_PX}")

    # 3) 以「縮小可行區」達成「障礙物膨脹」
    inflated_free = inflate_free_space(binary_map_with_goal, CLEAR_PX)

    # 4) 若起點/終點被吃掉（太貼邊），投影回最近安全像素
    edt = compute_clearance_map(inflated_free)

    # goal = goals_list[goal_idx - 1]
    print(f"[INFO] 選擇 {goal} 作為 RRT* 終點。")

    # 挑選起點 執行 RRT* 
    start = select_start(MAP_PATH, goal)
    safe_start = find_nearest_safe_pixel(edt, start, CLEAR_PX)
    safe_goal  = find_nearest_safe_pixel(edt, goal,  CLEAR_PX)

    if (tuple(map(int, safe_start)) != tuple(map(int, start)) or 
        tuple(map(int, safe_goal))  != tuple(map(int, goal))):
        print(f"[SAFE] start/goal 調整→ start:{start}→{safe_start} | goal:{goal}→{safe_goal}")

    # 5) 用「膨脹後」地圖跑 RRT*
    path, nodes = rrt_star_planning(inflated_free, safe_start, safe_goal,MIN_SAFE_DIST=0, SAFE_WEIGHT=500000)
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

    visualize_rrt(binary_map, nodes, safe_start, goal, path)

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
    try:
        habitat_sim.Simulator.close()
    except Exception:
        pass
    env = HabitatEnvWrapper(sim_settings)
    
    # 建立視窗
    cv2.namedWindow("Navigation (Overlay)", cv2.WINDOW_AUTOSIZE)
    # cv2.namedWindow("Depth (Debug)", cv2.WINDOW_AUTOSIZE)
    # cv2.namedWindow("Semantic (Debug)", cv2.WINDOW_AUTOSIZE) # 語義通常太雜亂，可選


    # 設置初始位置 初始朝向 (面向路徑的第二個點)
    start_x, start_z = pixel_to_world(safe_start[0], safe_start[1], w, h, bounds)   
    # 先 snap，再用 snap 後位置算朝向
    start_world = np.array([start_x, 0.0, start_z], dtype=np.float32)
    snap = env.sim.pathfinder.snap_point(start_world)

    dx = world_path[1][0] - snap[0]
    dz = world_path[1][1] - snap[2]
    init_yaw = math.atan2(-dx, -dz)          # 弧度

    env.reset_to([snap[0], 0.0, snap[2]], init_yaw)

    s = env.agent.get_state()
    yaw = get_yaw_from_quat(s.rotation)
    fwd = np.array([-math.sin(yaw), 0.0, -math.cos(yaw)])
    to_next = np.array([dx, 0.0, dz])
    cosang = float(np.dot(fwd, to_next) / (np.linalg.norm(fwd)*np.linalg.norm(to_next)+1e-9))
    print(f"[CHECK] cos(angle to next waypoint) = {cosang:.3f} (越接近 1 越對)")

    # 影片
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 編碼器
    video_path = os.path.join(OUTPUT_PATH, f"{target_class}_navigation.mp4")
    video_writer = cv2.VideoWriter(video_path, fourcc, FPS, (512, 512))  # (width, height)
    visualize_path_on_map(MAP_PATH, path, goal, start, target_class, OUTPUT_PATH, save_prefix="rrt_result_V3_initial")

    # === 執行控制 ===
    video_path = f"{OUTPUT_PATH}/{target_class}_V3_controller.mp4"
    run_navigation(
    env,
    world_path=world_path,
    bounds=bounds,
    target_mask_func=target_mask,
    goal_pixel=goal,
    video_writer=video_writer,
    w=w, h=h,
    lookahead_m=0.05,      # 可調：0.3~0.8 視路徑密度
    turn_on_deg=8.0,
    turn_off_deg=3.0,
    near_m=0.2,
    near_ratio_thr=0.2,
    arrival_thresh=ARRIVAL_THRESH,
    max_actions=MAX_ACTIONS
)

    env.sim.close()
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"[🎥 SAVED] Navigation video saved at {video_path}")
    print(f"[✅ DONE] V3 導航影片與地圖已輸出至 {OUTPUT_PATH}/")