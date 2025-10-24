import os
import cv2
import math
import json
import numpy as np
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
from PIL import Image
import matplotlib.pyplot as plt

# ==========================================================
# Habitat 環境封裝
# ==========================================================

# ==========================================================
# 動作生成（安全版本）
# ==========================================================
FORWARD_STEP = 0.05
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
        return {
            "rgb": obs["color_sensor"][:, :, [2, 1, 0]],
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

    return int(round(u)), int(round(v))


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
def overlay_mask(rgb, mask, color=(255, 0, 0), alpha=0.35):
    out = rgb.copy()
    color_layer = np.zeros_like(rgb)
    color_layer[..., :] = color
    blended = cv2.addWeighted(rgb, 1.0, color_layer, alpha, 0)
    out[mask.astype(bool)] = blended[mask.astype(bool)]
    return out

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
    path_pixel, _ = rrt_planning(safe_binary_map, start, goal)
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

    while True:
        count +=1
        print(f"replan count:{count}")
        for act in actions:
            obs = env.step(act)
            pos = env.agent.get_state().position.copy()
            dist_to_goal = np.linalg.norm(pos[[0, 2]] - np.array(world_path[-1]))
            dist_to_path = distance_to_path(pos, world_path)
            print(f"[DEBUG] pos=({pos[0]:.2f},{pos[2]:.2f}), "
                f"goal=({world_path[-1][0]:.2f},{world_path[-1][1]:.2f}), "
                f"dist_to_goal={dist_to_goal:.3f}, dist_to_path={dist_to_path:.3f}")
            # ======= 三種事件偵測 =======
            if dist_to_goal < 0.00001:
                print(f"[SUCCESS] Reached goal ✅ ({pos[0]:.2f}, {pos[2]:.2f})")
                vw.release()
                print(f"[END] Navigation finished, {frame_count} frames saved.")

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
        path_pixel, _ = rrt_planning(safe_binary_map, start_pixel, goal)
        counter = 0
        while path_pixel is None:
            print(f"❌ [REPLAN FAIL] 無法找到可行路徑， again。{start_pixel}")
            path_pixel, _ = rrt_planning(safe_binary_map, start_pixel, goal)
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

    vw.release()
    print(f"[END] Navigation finished, {frame_count} frames saved.")

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

    from part2 import rrt_planning, load_semantic_table, find_object_region

    MAP_PATH = "/home/clu98753cs13/Desktop/course/phyai/physical-ai25/hw2/map.png"
    EXCEL_PATH = "/home/clu98753cs13/Desktop/course/phyai/physical-ai25/hw2/color_coding_semantic_segmentation_classes.xlsx"
    TARGET_CLASS = "window"
    BOUNDS_PATH = "/home/clu98753cs13/Desktop/course/phyai/physical-ai25/hw2/coordinate_bounds.json"
    OUTPUT_PATH = "./part3OUTPUT"
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # === Part2 輸出接續 ===
    color_map = load_semantic_table(EXCEL_PATH)
    goal, mask = find_object_region(MAP_PATH, color_map, TARGET_CLASS)
    start = (335, 240)

    map_gray = cv2.imread(MAP_PATH, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(map_gray, 240, 255, cv2.THRESH_BINARY)
# === 解決方案：侵蝕可行走區域 (建立安全緩衝區) ===
    print("[INFO] 正在侵蝕可行走區域以建立安全緩衝區...")
    
    # 決定緩衝區的大小 (kernel 越大，緩衝區越寬，路徑越保守)
    # 5x5 或 7x7 通常是個好的開始
    kernel_size = 15 
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # 'binary' 中可行走區域是 255 (白色)，障礙物是 0 (黑色)
    # 'erode' (侵蝕) 會縮小白**色區域，使其遠離黑色邊界
    # 這等同於讓障礙物 (牆壁) "膨脹" 了 (kernel_size / 2) 個像素
    safe_binary_map = cv2.erode(binary, kernel, iterations=1)
    
    print("[INFO] 安全緩衝區建立完畢。")
    # === 解決方案結束 ===
    path, _ = rrt_planning(safe_binary_map, start, goal)
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
    for i in range(5):
        (x1, z1), (x2, z2) = world_path[i], world_path[i+1]
        print(f"Segment {i}: dist={math.hypot(x2 - x1, z2 - z1):.3f}")

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

    def target_mask(obs):
        semantic_img = obs["semantic"]
        target_rgb = np.array(color_map[TARGET_CLASS.lower()])[::-1]
        mask = cv2.inRange(semantic_img, target_rgb, target_rgb)
        return (mask > 0).astype(np.uint8)

    # === 導航 + 輸出 ===
    env = HabitatEnvWrapper(sim_settings)
    
    # TODO: 1.Which code snippet affects my position and angle?
    
    # TODO: 2.set init position 
    start_x, start_z = pixel_to_world(start[0], start[1], w, h, bounds)
    # start_x, start_z = start_x * 40, start_z * 40   # 若你有 global scaling
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
