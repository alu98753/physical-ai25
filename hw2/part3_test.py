import math, random, numpy as np
import habitat_sim
from habitat_sim.nav import ShortestPath


# This function generates a config for the simulator.
# It contains two parts:
# one for the simulator backend
# one for the agent, where you can attach a bunch of sensors

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

def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.gpu_device_id = 0

    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    
    agent_cfg.action_space["move_backward"] = habitat_sim.agent.ActionSpec(
        "move_forward", habitat_sim.agent.ActuationSpec(amount=-0.25)  # -0.25 表示向後走 0.25m
    )
    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    #depth snesor
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    #semantic snesor
    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec, semantic_sensor_spec]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])



sim_settings = {
    "scene": "replica_v1/apartment_0/habitat/mesh_semantic.ply",  # Scene path
    "default_agent": 0,  # Index of the default agent
    "sensor_height": 1.5,  # Height of sensors in meters, relative to the agent
    "width": 512,  # Spatial resolution of the observations
    "height": 512,
    "sensor_pitch": 0,  # sensor pitch (x rotation in rads)
}

# ========= 你原本的 sim/cfg 建議保留 =========
cfg = make_simple_cfg(sim_settings)
sim = habitat_sim.Simulator(cfg)

# === 1) 載入 navmesh（很重要：用對應場景的 .navmesh）===
navmesh_path = "replica_v1/apartment_0/habitat/mesh_semantic.navmesh"
sim.pathfinder.load_nav_mesh(navmesh_path)  # 成功後 sim.pathfinder.is_loaded() 會是 True
pf = sim.pathfinder

# === 2) 參數（單位：公尺）===
STEP_SIZE_M      = 0.25          # RRT 每次延伸步長
NEIGHBOR_COEFF   = 1.5           # RRT* 鄰域半徑係數（r ~ coeff * sqrt(log(n)/n)）
CLEARANCE_M      = 0.20          # 你想要的最小安全距離（公尺）
SEGMENT_CHECK_DS = 0.05          # 邊段離散檢查間距（越小越嚴）
GOAL_TOL_M       = 0.20          # 抵達判定半徑（公尺）
MAX_ITER         = 20000

# === 3) 工具函式 ===
def same_island(a, b):
    return pf.get_island(a) == pf.get_island(b)

def snap(p):
    # 把點吸到 NavMesh（保持樓層/高度合規）
    return pf.snap_point(p)

def point_ok(p):
    # 點可走且保有安全距離
    if not pf.is_navigable(p): return False
    d = pf.distance_to_closest_obstacle(p, max_search_radius=CLEARANCE_M+0.5)
    return d >= CLEARANCE_M

def segment_ok(a, b):
    # 沿著線段做稠密採樣，要求每個採樣點都可走且距離障礙 >= CLEARANCE_M
    ab = np.array(b) - np.array(a)
    L = np.linalg.norm(ab)
    if L < 1e-6: return point_ok(a)
    n = max(2, int(math.ceil(L / SEGMENT_CHECK_DS)))
    for i in range(n+1):
        t = i / n
        p = a + t * ab
        p = snap(p)  # 緊貼 NavMesh
        if (not pf.is_navigable(p)) or (pf.distance_to_closest_obstacle(p, CLEARANCE_M+0.5) < CLEARANCE_M):
            return False
    return True

def rand_sample_on_island(island_idx):
    # 從全域取樣，直到落在同一個 island 且滿足安全距離
    for _ in range(1000):
        p = pf.get_random_navigable_point()
        if pf.get_island(p) != island_idx: 
            continue
        if point_ok(p): 
            return p
    return None

def steer(a, b, step=STEP_SIZE_M):
    v = np.array(b) - np.array(a)
    d = np.linalg.norm(v)
    if d <= step: 
        return snap(b)
    return snap(a + v * (step / d))

# === 4) RRT* 主流程（世界座標） ===
class Node:
    __slots__ = ("p","parent","cost")
    def __init__(self, p, parent=None, cost=0.0):
        self.p = np.array(p, dtype=np.float32)
        self.parent = parent
        self.cost = cost

def rrt_star_world(start_world, goal_world):
    # 將起終點吸到 NavMesh，並確認在同一樓層/連通區
    s0 = snap(np.array(start_world, dtype=np.float32))
    g0 = snap(np.array(goal_world,  dtype=np.float32))
    assert pf.is_navigable(s0) and pf.is_navigable(g0), "Start/Goal 不在可走區"
    assert same_island(s0, g0), "Start/Goal 不在同一 island（可能不同樓層）"
    assert point_ok(s0) and point_ok(g0), "Start/Goal 未滿足安全距離，請移動或放寬 CLEARANCE_M"

    island_idx = pf.get_island(s0)
    nodes = [Node(s0, parent=None, cost=0.0)]
    goal_node = None

    for it in range(MAX_ITER):
        # Goal bias
        if random.random() < 0.25:
            x_rand = g0
        else:
            x_rand = rand_sample_on_island(island_idx)
            if x_rand is None:
                continue

        # 最近鄰
        dists = [np.linalg.norm(n.p - x_rand) for n in nodes]
        idx   = int(np.argmin(dists))
        x_near = nodes[idx].p

        # 延伸
        x_new = steer(x_near, x_rand, STEP_SIZE_M)
        if not segment_ok(x_near, x_new):
            continue

        # RRT* 鄰域重接
        new_cost = nodes[idx].cost + np.linalg.norm(x_new - x_near)
        node_new = Node(x_new, parent=nodes[idx], cost=new_cost)

        # 鄰域半徑（經典 r ~ c * sqrt(log n / n)）
        r = NEIGHBOR_COEFF * math.sqrt(math.log(len(nodes)+1) / (len(nodes)+1))
        r = max(r, STEP_SIZE_M*2)

        # 找更好的 parent（保證每段邊「segment_ok」）
        for j, nj in enumerate(nodes):
            if np.linalg.norm(nj.p - x_new) <= r:
                c_try = nj.cost + np.linalg.norm(nj.p - x_new)
                if c_try + 1e-6 < node_new.cost and segment_ok(nj.p, x_new):
                    node_new.parent = nj
                    node_new.cost   = c_try

        nodes.append(node_new)

        # Rewire
        for j, nj in enumerate(nodes[:-1]):
            if np.linalg.norm(nj.p - x_new) <= r:
                c_try = node_new.cost + np.linalg.norm(nj.p - x_new)
                if c_try + 1e-6 < nj.cost and segment_ok(x_new, nj.p):
                    nj.parent = node_new
                    nj.cost   = c_try

        # 抵達判定
        if np.linalg.norm(node_new.p - g0) <= GOAL_TOL_M and segment_ok(node_new.p, g0):
            goal_node = Node(g0, parent=node_new, cost=node_new.cost + np.linalg.norm(g0 - node_new.p))
            break

    # 回溯輸出路徑
    if goal_node is None:
        return None  # 規劃失敗，可調 MAX_ITER / CLEARANCE_M
    path = []
    cur = goal_node
    while cur is not None:
        path.append(cur.p.copy())
        cur = cur.parent
    path.reverse()
    return path

# === 5) 使用範例 ===
start = np.array([0.0, 0.0, 0.0])  # 你的世界座標（會 snap 到地面）
goal  = np.array([2.0, 0.0, 5.0])
path_world = rrt_star_world(start, goal)
print(f"path has {len(path_world) if path_world else 0} waypoints")
