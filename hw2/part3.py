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
# 座標轉換
# ==========================================================
def load_bounds(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["xmin"], data["xmax"], data["zmin"], data["zmax"]

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
        if abs(d) > math.radians(5):  # 放寬閾值
            if d > 0:
                actions.append("turn_left")
                yaw += step
            else:
                actions.append("turn_right")
                yaw -= step
        # 不要無限迴圈，避免震盪
        # print(f"[turn] desired={math.degrees(desired_yaw):.1f}, curr={math.degrees(yaw):.1f}, diff={math.degrees(d):.1f}")


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
# 導航與影片錄製
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


def run_navigation(env, world_path, target_mask, output_video="result.mp4"):
    actions = generate_actions_from_world_path(world_path)
    print("actions:",actions)
    vw = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (512, 512))
    frame_count = 0

    # for idx, act in enumerate(actions):
    #     print(f"[Step {idx}] action={act}, pos={env.agent.get_state().position}")
    #     # print(f"[Step {idx}] action={act}, pos={env.agent.get_state().position}, world path:{world_path[idx]}")

    #     obs = env.step(act)
    #     # if idx % 10 != 0:
    #     #     continue
    #     rgb = obs["rgb"]
    #     mask = target_mask(obs)
    #     vis = overlay_mask(rgb, mask)
    #     vw.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    #     frame_count += 1
    #     if idx % 200 == 0:
    #         print(f"[INFO] Progress: {idx}/{len(actions)} actions executed")
    for idx, act in enumerate(actions):
        for sub in range(int(1)):  # 每個動作輸出3幀
            obs = env.step(act)
            rgb = obs["rgb"]
            mask = target_mask(obs)
            vis = overlay_mask(rgb, mask)
            vw.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            frame_count += 1


    vw.release()
    print(f"[INFO] Navigation complete, {frame_count} frames saved ✅")


def run_navigation_replan(env, binary_map,safe_binary_map, color_map, bounds, start, goal, target_mask,
                        output_video="result_replan.mp4", replan_thresh=0.3):
    """
    主導航流程：包含 stuck / deviation / goal 三種事件觸發
    """
    vw = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (512, 512))
    stuck_counter = 0
    frame_count = 0

    h, w = binary_map.shape
    current_state = env.agent.get_state()
    current_pos = current_state.position.copy()

    # === 初始路徑規劃 ===
    path_pixel, _ = rrt_star_planning(safe_binary_map, start, goal)
    world_path = [pixel_to_world(u, v, w, h, bounds) for (u, v) in path_pixel]
    # actions = generate_actions_from_world_path(world_path)
    
    # 從 current_state 獲取初始 yaw
    q = current_state.rotation
    init_yaw = 2 * math.atan2(q.imag[1], q.real) # 從四元數提取 Y 軸旋轉
    
    actions = generate_actions_from_world_path(world_path, current_yaw_rad=init_yaw)
    print(f"[INIT] RRT 路徑生成完成，共 {len(world_path)} 點")
    print("[DEBUG] First world_path point:", world_path[0])
    print("[DEBUG] Last world_path point:", world_path[-1])
    print("[DEBUG] Typical segment dist:", np.mean(
        [math.hypot(world_path[i+1][0]-world_path[i][0],
                    world_path[i+1][1]-world_path[i][1])
        for i in range(len(world_path)-1)]))
    count = 0
    try:
        while True:
            count +=1
            print(f"replan count:{count}")
            for act in actions:
                obs = env.step(act)
                print(f"obs:",obs["depth"].shape)
                depth = obs["depth"]  # shape: (512, 512)

                # 範例：定義幾個距離層（單位公尺）
                near = (depth < 0.4)          # 很近的東西
                mid  = (depth >= 0.4) & (depth < 1.5)
                far  = (depth >= 1.5)

                # 這樣你可以看到每一層的布林遮罩
                print(np.sum(near), np.sum(mid), np.sum(far))  # 各層有多少像素
                pos = env.agent.get_state().position.copy()
                dist_to_goal = np.linalg.norm(pos[[0, 2]] - np.array(world_path[-1]))
                dist_to_path = distance_to_path(pos, world_path)
                print(f"[DEBUG] pos=({pos[0]:.2f},{pos[2]:.2f}), "
                    f"goal=({world_path[-1][0]:.2f},{world_path[-1][1]:.2f}), "
                    f"dist_to_goal={dist_to_goal:.3f}, dist_to_path={dist_to_path:.3f}")
                # ======= 三種事件偵測 =======
                if dist_to_goal < ARRIVAL_JUDGE:
                    print(f"[SUCCESS] Reached goal ✅ ({pos[0]:.2f}, {pos[2]:.2f})")
                    rgb = obs["rgb"].copy()
                    vis = end_anime(rgb)
                    for _ in range(FPS * 3):
                        vw.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                        frame_count += 1
                    vw.release()
                    print(f"[END] Navigation finished with ending screen, {frame_count} frames saved.")
                    return
                move_dist = np.linalg.norm(pos - current_pos)

                if dist_to_path > replan_thresh:
                    # print(f"[REPLAN] Deviated from path ({dist_to_path:.2f} m) → 重新規劃")
                    current_pos = pos.copy()
                    break  # 退出內層 loop，重新規劃

                if move_dist < 0.01:
                    stuck_counter += 1
                else:
                    stuck_counter = 0

                if stuck_counter > 20:
                    # print("[REPLAN] Agent stuck → 重新規劃")
                    current_pos = pos.copy()
                    break  # 跳出重新規劃
                current_pos = pos.copy()
                # === 繪製畫面 ===
                rgb = obs["rgb"]
                mask = target_mask(obs)
                vis = overlay_mask(rgb, mask)
                for sub in range(FPS//15):  # 每個動作輸出3幀
                    vw.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                    frame_count += 1
                if frame_count % 10 == 0:
                    print(f"[DEBUG] 已寫入 {frame_count} 幀到影片")

                # current_pos = pos.copy()
            else:
                # 如果沒 break（未重規劃）則繼續
                continue

            # ======= 重新規劃 =======
            # print(current_pos[0], current_pos[2], bounds, w, h)
            # print("[DEBUG] bounds =", bounds)

            start_pixel = world_to_pixel(current_pos[0], current_pos[2], w, h, bounds)
            try:
                path_pixel, _ = rrt_star_planning(safe_binary_map, start_pixel, goal)
            except Exception as e:
                print(f"❌ [ERROR] 重新規劃失敗: {e}")
                raise  # 保證影片會被 finally 釋放
            counter = 0
            while path_pixel is None:
                print(f"❌ [REPLAN FAIL] 無法找到可行路徑， again。{start_pixel}")
                try:
                    path_pixel, _ = rrt_star_planning(safe_binary_map, start_pixel, goal)
                except Exception as e:
                    print(f"❌ [ERROR] 重新規劃失敗: {e}")
                    raise  # 保證影片會被 finally 釋放
                counter +=1
                if path_pixel or counter >=10:
                    break
            if counter >=10  or count>50 : # or count>20
                print(f"[Fail] counter:{counter},frame count:{frame_count},count:{count}")
                print(f"[Fail] not Reached goal but fail to find path ({pos[0]:.2f}, {pos[2]:.2f})")
                vw.release()
                print(f"[END] Navigation finished, {frame_count} frames saved.")
                return
            
            world_path = [pixel_to_world(u, v, w, h, bounds) for (u, v) in path_pixel]
            # actions = generate_actions_from_world_path(world_path)
            q_replan = current_state.rotation
            replan_yaw = 2 * math.atan2(q_replan.imag[1], q_replan.real)
            
            actions = generate_actions_from_world_path(world_path, current_yaw_rad=replan_yaw)
            counter = 0
            # print(f"[INFO] Replan done: {len(world_path)} points, {len(actions)} actions.")
        
    except KeyboardInterrupt:
        print("\n🛑 [INTERRUPT] 使用者手動中斷，保存錄影...")

    except Exception as e:
        print(f"\n❌ [UNCAUGHT ERROR] {type(e).__name__}: {e}")
        print("🚨 自動保存目前錄影並退出。")

    finally:
        # 保證影片與環境釋放
        if vw is not None:
            try:
                vw.release()
                print(f"[SAVE] 影片已安全保存 ({frame_count} 幀)")
            except Exception as e:
                print(f"[WARN] 影片釋放出錯: {e}")

        try:
            env.sim.close()
            print("[CLEANUP] 模擬器已關閉 ✅")
        except Exception as e:
            print(f"[WARN] 模擬器關閉出錯: {e}")

    vw.release()
    print(f"[END] Navigation finished, {frame_count} frames saved.")

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

# def run_navigation_replan(env, binary_map,safe_binary_map, color_map, bounds, start, goal, target_mask,
#                         output_video="result_replan.mp4", replan_thresh=ARRIVAL_JUDGE*0.95):
#     """
#     主導航流程：
#     1. RRT 全域路徑規劃
#     2. Proactive Avoidance (主動避障)
#     3. Reactive Replan (反應式重新規劃)
#     4. Oscillation Escape (震盪脫困)
#     """
#     vw = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (512, 512))
#     stuck_counter = 0
#     frame_count = 0
#     h, w = binary_map.shape

#     # === ✅ NEW: 脫困狀態機 ===
#     # 追蹤 "被動卡住" (stuck_counter) 的脫困嘗試
#     # 0=未嘗試, 1=已嘗試右轉90度, 2=已嘗試左轉90度, 3=已嘗試180度
#     stuck_escape_level = 0
#     # ==========================

#     obs = env.step("turn_right") # dict_keys(['rgb', 'depth', 'semantic'])
#     current_state = env.agent.get_state()
#     current_pos = current_state.position.copy()

#     # === 初始路徑規劃 ===
#     path_pixel, _ = rrt_star_planning(safe_binary_map, start, goal)
#     world_path = [pixel_to_world(u, v, w, h, bounds) for (u, v) in path_pixel]
#     world_path = simplify_path(world_path, min_step=0.4)

#     q = current_state.rotation
#     init_yaw = 2 * math.atan2(q.imag[1], q.real)
#     actions = generate_actions_from_world_path(world_path, current_yaw_rad=init_yaw)
    
#     print(f"[INIT] RRT 路徑生成完成，共 {len(world_path)} 點")

#     # (系統 A) 震盪偵測器
#     avoidance_turn_counter = 0
#     OSCILLATION_LIMIT = 25 
#     count = 0
#     try:
#         while True:
#             count +=1
#             print(f"replan count:{count}")
            
#             for act_planned in actions:
                
#                 # === 系統 A: 主動避障邏輯 ===
#                 depth = obs["depth"]
#                 h_depth, w_depth = depth.shape
                
#                 center_h_min = int(h_depth * 0.3)  # 只看前方 (畫面的下半部)
#                 center_h_max = h_depth
                
#                 # 只在限定區域內計算 near_mask
#                 near_depth_slice = depth
#                 near_mask = (near_depth_slice < 0.65) # 使用調整後的 0.8m
#                 near_pixel_count = np.sum(near_mask)
#                 print(f"obs count:{near_pixel_count}, avoidance_turn_counter:{avoidance_turn_counter}")
                
#                 OBSTACLE_THRESHOLD = 1000 # 調整後的閾值
                
#                 action_to_take = act_planned
#                 pos = env.agent.get_state().position.copy()
#                 dist_to_goal = np.linalg.norm(pos[[0, 2]] - np.array(world_path[-1]))

#                 if act_planned == "move_forward" and near_pixel_count > OBSTACLE_THRESHOLD and dist_to_goal>0.65:
#                     print("主動避障")
#                     left_side_near = np.sum(near_mask[:, :near_depth_slice.shape[1]//2])
#                     right_side_near = np.sum(near_mask[:, near_depth_slice.shape[1]//2:])
                    
#                     if left_side_near > right_side_near:
#                         action_to_take = "turn_right"
#                     else:
#                         action_to_take = "turn_left"
#                     avoidance_turn_counter +=1
                    
#                 elif act_planned == "move_forward":
#                     # 成功前進，重置 [主動避障] 計數器
#                     avoidance_turn_counter -=1
#                 # === END: 主動避障 ===

#                 # === 系統 A: 脫困 (B計畫) ===
#                 if avoidance_turn_counter > OSCILLATION_LIMIT:
#                     print("🆘 [ESCAPE A] 偵測到 [主動避障震盪]，嘗試 [主要脫困 - 轉 180 度]")
                    
#                     turn_steps = 90 // TURN_ANGLE 
#                     for i in range(turn_steps):
#                         obs = env.step("turn_right") 
#                         # ... (錄影) ...
#                         rgb = obs["rgb"]; mask = target_mask(obs); vis = overlay_mask(rgb, mask)
#                         for sub in range(FPS//15): vw.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)); frame_count += 1
                    
#                     current_pos = env.agent.get_state().position.copy()
#                     stuck_escape_level = 0 # 重置 "被動脫困" 狀態
#                     avoidance_turn_counter = 0
#                     break # 觸發重新規劃
#                 # === END: 系統 A 脫困 ===

#                 # 執行動作
#                 obs = env.step(action_to_take) 
#                 pos = env.agent.get_state().position.copy()
#                 dist_to_goal = np.linalg.norm(pos[[0, 2]] - np.array(world_path[-1]))
#                 dist_to_path = distance_to_path(pos, world_path)
#                 print(f"[DEBUG] pos=({pos[0]:.2f},{pos[2]:.2f}), "
#                     f"goal=({world_path[-1][0]:.2f},{world_path[-1][1]:.2f}), "
#                     f"dist_to_goal={dist_to_goal:.3f}, dist_to_path={dist_to_path:.3f}")
#                 # --- 事件偵測 ---
#                 if dist_to_goal < ARRIVAL_JUDGE:
#                     print(f"[SUCCESS] Reached goal ✅ ({pos[0]:.2f}, {pos[2]:.2f})")
#                     vw.release()
#                     return

#                 if dist_to_path > replan_thresh:
#                     print(f"[REPLAN] 偏離路徑 ({dist_to_path:.2f} m) → 重新規劃")
#                     current_pos = pos.copy()
#                     stuck_escape_level = 0 # 重置 "被動脫困" 狀態
#                     break  # 退出內層 loop，重新規劃

#                 # --- ✅ NEW: 系統 B (被動卡住) 脫困邏輯 ---
#                 move_dist = np.linalg.norm(pos - current_pos)
                
#                 if move_dist < 0.01 and action_to_take == "move_forward":
#                     # 嘗試前進但失敗，累加 "被動卡住" 計數器
#                     stuck_counter += 1
#                 else:
#                     # 任何成功移動 (或轉彎) 都重置計數器
#                     stuck_counter = 0
#                     stuck_escape_level = 0 # 只要有移動，就重置脫困等級

#                 if stuck_counter > 20:
#                     # 觸發了「被動卡住」
                    
#                     if stuck_escape_level == 0:
#                         # 第一次卡住：嘗試右轉 90 度
#                         print("🆘 [ESCAPE B] Agent [被動卡住]，嘗試 [次要脫困 - 轉 90 度]")
#                         turn_steps = 90 // TURN_ANGLE
#                         for i in range(turn_steps):
#                             obs = env.step("turn_right")
#                             # ... (錄影) ...
#                             rgb = obs["rgb"]; mask = target_mask(obs); vis = overlay_mask(rgb, mask)
#                             for sub in range(FPS//15): vw.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)); frame_count += 1
#                         stuck_escape_level = 1 # 升級

#                     elif stuck_escape_level == 1:
#                         # 第二次卡住：嘗試左轉 90 度
#                         print("🆘 [ESCAPE B] Agent [仍被動卡住]，嘗試 [次要脫困 - 轉 -90 度]")
#                         turn_steps = 90 // TURN_ANGLE
#                         for i in range(turn_steps):
#                             obs = env.step("turn_left")
#                             # ... (錄影) ...
#                             rgb = obs["rgb"]; mask = target_mask(obs); vis = overlay_mask(rgb, mask)
#                             for sub in range(FPS//15): vw.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)); frame_count += 1
#                         stuck_escape_level = 2 # 升級

#                     else: # stuck_escape_level >= 2
#                         # 最終嘗試：轉 180 度
#                         print("🆘 [ESCAPE B] Agent [仍被動卡住]，嘗試 [主要脫困 - 轉 180 度]")
#                         turn_steps = 180 // TURN_ANGLE
#                         for i in range(turn_steps):
#                             obs = env.step("turn_right")
#                             # ... (錄影) ...
#                             rgb = obs["rgb"]; mask = target_mask(obs); vis = overlay_mask(rgb, mask)
#                             for sub in range(FPS//15): vw.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)); frame_count += 1
#                         stuck_escape_level = 0 # 重置
                    
#                     # 執行完任何 "被動脫困" 後，重置計數器並 break 去重新規劃
#                     current_pos = env.agent.get_state().position.copy()
#                     stuck_counter = 0 # 重置
#                     break # 觸發重新規劃
#                 # --- ✅ END: 系統 B 脫困 ---
                    
#                 current_pos = pos.copy()
                
#                 # === 繪製畫面 ===
#                 rgb = obs["rgb"]
#                 mask = target_mask(obs)
#                 vis = overlay_mask(rgb, mask)
#                 for sub in range(FPS//15):
#                     vw.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
#                     frame_count += 1
#             else:
#                 continue

#             # ======= 重新規劃 (Re-plan) =======
#             # (此區塊不變)
            
#             start_pixel = world_to_pixel(current_pos[0], current_pos[2], w, h, bounds)
#             path_pixel, _ = rrt_star_planning(safe_binary_map, start_pixel, goal)
#             counter = 0
#             while path_pixel is None:
#                 print(f"❌ [REPLAN FAIL] 無法找到可行路徑， again。{start_pixel}")
#                 path_pixel, _ = rrt_star_planning(safe_binary_map, start_pixel, goal)
#                 counter +=1
#                 if path_pixel or counter >=20:
#                     break
                
#             if counter >=20  or count>20 :
#                 print(f"[Fail] counter:{counter},frame count:{frame_count},count:{count}")
#                 print(f"[Fail] not Reached goal but fail to find path ({pos[0]:.2f}, {pos[2]:.2f})")
#                 vw.release()
#                 return
            
#             world_path = [pixel_to_world(u, v, w, h, bounds) for (u, v) in path_pixel]
#             world_path = simplify_path(world_path, min_step=0.4)

#             current_state_replan = env.agent.get_state()
#             q_replan = current_state_replan.rotation
#             replan_yaw = 2 * math.atan2(q_replan.imag[1], q_replan.real)
            
#             actions = generate_actions_from_world_path(world_path, current_yaw_rad=replan_yaw)
#             stuck_counter = 0 
#             print(f"[INFO] Replan done: {len(world_path)} points, {len(actions)} actions.")

        
#     except KeyboardInterrupt:
#         print("\n🛑 [INTERRUPT] 使用者手動中斷，保存錄影...")

#     except Exception as e:
#         print(f"\n❌ [UNCAUGHT ERROR] {type(e).__name__}: {e}")
#         print("🚨 自動保存目前錄影並退出。")

#     finally:
#         # 保證影片與環境釋放
#         if vw is not None:
#             try:
#                 vw.release()
#                 print(f"[SAVE] 影片已安全保存 ({frame_count} 幀)")
#             except Exception as e:
#                 print(f"[WARN] 影片釋放出錯: {e}")

#         try:
#             env.sim.close()
#             print("[CLEANUP] 模擬器已關閉 ✅")
#         except Exception as e:
#             print(f"[WARN] 模擬器關閉出錯: {e}")

#     vw.release()
#     print(f"[END] Navigation finished, {frame_count} frames saved.")


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
    TARGET_CLASS = "sofa"
    BOUNDS_PATH = "/home/clu98753cs13/Desktop/course/phyai/physical-ai25/hw2/coordinate_bounds.json"
    OUTPUT_PATH = "./part3OUTPUT"
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # === Part2 輸出接續 ===
    color_map = load_semantic_table(EXCEL_PATH)
    id_map = load_semantic_ID_table(EXCEL_PATH)

    goal, mask = find_object_region(MAP_PATH, color_map, TARGET_CLASS)
    start = (335, 240) # window test final: 要繞過椅子等 還要左右轉彎
    start = (236, 457) # window test1: 直線行走左右有障礙 可能卡牆 pass
    
    start = (382, 256) # sofa test1: 直線走 pass
    # start = (212, 428) # base-cabinet test1: 直線走

    map_gray = cv2.imread(MAP_PATH, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(map_gray, 240, 255, cv2.THRESH_BINARY)
# === 解決方案：侵蝕可行走區域 (建立安全緩衝區) ===
    print("[INFO] 正在侵蝕可行走區域以建立安全緩衝區...")
    
    # 決定緩衝區的大小 (kernel 越大，緩衝區越寬，路徑越保守)
    # 5x5 或 7x7 通常是個好的開始
    kernel_size = 0
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

    # # ----------------------------------------------------------
    # # test: forward 動作是否生效 + 錄影
    # print("[DEBUG] smoke test: try 10 forward steps and record")

    # vw = cv2.VideoWriter(f"{OUTPUT_PATH}/smoke_forward_test.mp4",
    #                     cv2.VideoWriter_fourcc(*"mp4v"),
    #                     5, (512, 512))
    # last_pos = env.agent.get_state().position.copy()

    # for i in range(10):
    #     obs = env.step("move_forward")
    #     pos = env.agent.get_state().position
    #     moved = np.linalg.norm(pos - last_pos) > 1e-3
    #     print(f"[smoke] i={i}, moved={moved}, pos={pos}")
    #     frame = obs["rgb"]
    #     vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    #     last_pos = pos.copy()

    # vw.release()
    # print("[DEBUG] smoke test video saved as smoke_forward_test.mp4")
    # # ----------------------------------------------------------
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
        
        print(f"[DEBUG] colormap (from excel): {color_map[TARGET_CLASS.lower()]}, used target_rgb: {target_rgb}")
        mask = cv2.inRange(semantic_img, target_rgb, target_rgb)
        print(f"[DEBUG] mask pixels: {np.sum(mask > 0)}")  # 看有沒有非0像素

        cv2.imwrite("debug_semantic_rgb.png", cv2.cvtColor(semantic_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite("debug_mask.png", mask )

        return (mask > 0).astype(np.uint8)


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
    run_navigation_replan(env, binary, safe_binary_map, color_map, bounds, start, goal, target_mask, output_video=video_path)

    # run_navigation(env, world_path, target_mask, output_video=video_path)
    visualize_path_on_map(MAP_PATH, path, goal, start, TARGET_CLASS, OUTPUT_PATH)
    env.sim.close()
    print(f"[✅ DONE] 導航影片與地圖已輸出至 {OUTPUT_PATH}/")
