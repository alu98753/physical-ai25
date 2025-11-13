import numpy as np
import open3d as o3d
import argparse
import math
import os
import cv2
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R
import glob
import time

def depth_image_to_point_cloud(rgb, depth, fov_deg: float = 90.0, depth_scale: float = 100.0,
                            depth_trunc_m: float = 10.0):
    H, W = depth.shape
    f = (W * 0.5) / math.tan(math.radians(fov_deg) * 0.5)
    intr = o3d.camera.PinholeCameraIntrinsic(W, H, f, f, W * 0.5, H * 0.5)
    color_o3d = o3d.geometry.Image(rgb.astype(np.uint8))
    depth_m = (depth.astype(np.float32) / 255.0) * 10.0

    depth_o3d = o3d.geometry.Image(depth_m)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=1.0,
        depth_trunc=depth_trunc_m,
        convert_rgb_to_intensity=False
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)
    return pcd


def preprocess_point_cloud(pcd: o3d.geometry.PointCloud, voxel_size: float):
    """
    以體素降採樣降低點數，並估計法向量和 FPFH 特徵（後續 ICP/特徵計算更穩定）。

    Args:
        pcd (o3d.geometry.PointCloud): 原始點雲
        voxel_size (float): 體素大小（公尺），例：0.1

    Returns:
        tuple: (pcd_down, pcd_fpfh) - 降採樣點雲和 FPFH 特徵
    """
    assert isinstance(pcd, o3d.geometry.PointCloud)
    assert voxel_size > 0.0

    # 1) 體素降採樣
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)

    # 2) 估計法向量
    # 半徑設為 ~ 2 個體素，鄰居數做上限
    radius_normal = voxel_size * 2.0
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normal,
            max_nn=60
        )
    )
    # 將法向量朝向一致（可提升配準穩定性）
    pcd_down.orient_normals_consistent_tangent_plane(k=30)

    # 3) 計算 FPFH 特徵
    radius_feature = voxel_size * 6.0
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=60)
    )

    return pcd_down, pcd_fpfh



def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """
    全域配準 (Global Registration)：
    使用 FPFH 特徵與 RANSAC 做初步對齊。

    Args:
        source_down (o3d.geometry.PointCloud): 降採樣後來源點雲
        target_down (o3d.geometry.PointCloud): 降採樣後目標點雲
        source_fpfh (o3d.pipelines.registration.Feature): 來源特徵
        target_fpfh (o3d.pipelines.registration.Feature): 目標特徵
        voxel_size (float): 體素大小（會影響搜尋半徑）

    Returns:
        o3d.pipelines.registration.RegistrationResult
    """
    distance_threshold = voxel_size * 1.5

    print(f"[INFO] Running Global Registration (RANSAC) with distance threshold = {distance_threshold:.3f}")

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down,
        source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,  # 對齊優秀版本：每次隨機取4點
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 400)  # 對齊優秀版本：(50000, 400)
        # criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500) # 「最大嘗試次數」,confidence： ANSAC 找到了一個不錯的對齊（例如，有 100 個點是匹配的），然後在接下來的 500 次新嘗試中，都沒能找到一個能讓超過 100 個點匹配的更好方案，它就會停止，

    )
    print("[INFO] RANSAC Done. Inlier RMSE:", result.inlier_rmse)
    return result


def compute_transformation_pt2plane_matrix(source_pts: np.ndarray, target_pts: np.ndarray, target_normals: np.ndarray):
    """
    【加速版 Pt2Plane 求解】使用矩陣化運算代替迴圈。
    
    Args:
        source_pts (np.ndarray): 已變換到當前迭代的來源點 (N, 3)。
        target_pts (np.ndarray): 目標匹配點 (N, 3)。
        target_normals (np.ndarray): 目標點的法向量 (N, 3)。
        
    Returns:
        tuple: (delta_T (4x4), rmse (float), fitness (float))
    """
    N = source_pts.shape[0]
    if N < 6:
        return np.eye(4), 9999.0, 0.0

    errors = np.sum((source_pts - target_pts) * target_normals, axis=1) # (N,)

    # === 2. 計算雅可比矩陣 J (N x 6) ===
    J_rot = np.cross(source_pts, target_normals) # (N, 3)
    
    J_trans = target_normals # (N, 3)
    
    # c. 組合 J (N x 6)
    J = np.hstack([J_rot, J_trans]) 
    
    A = J.T @ J # 矩陣乘法，高速計算 
    
    b_vec = J.T @ errors 

    try:
        x = np.linalg.solve(A, -b_vec) 
    except np.linalg.LinAlgError:
        return np.eye(4), 9999.0, 0.0

    R_delta = R.from_rotvec(x[0:3]).as_matrix()
    t_delta = x[3:6]
    
    delta_T = np.eye(4)
    delta_T[:3, :3] = R_delta
    delta_T[:3, 3] = t_delta

    total_sq_error = np.sum(errors**2)
    rmse = np.sqrt(total_sq_error / N)
    
    fitness = N / source_pts.shape[0] 
    return delta_T, rmse, fitness

def my_local_icp_algorithm_accelerated(source_down, target_down, trans_init, voxel_size_icp, mean_depth=None):
    t_start = time.time()
    source_pts_init = get_points(source_down)
    target_pts = get_points(target_down)
    N_s_total = source_pts_init.shape[0]
    # === 0️⃣ 快取機制 (KDTree & 法向量) ===
    key_tgt = id(target_down)
    
    # KDTree 構建 (O(N_t log N_t) 一次性開銷)
    if key_tgt not in KD_TREE_CACHE:
        print(f"[INFO] Building KDTree for target cloud (N={target_pts.shape[0]})...")
        KD_TREE_CACHE[key_tgt] = KDTree(target_pts)
    target_kdtree = KD_TREE_CACHE[key_tgt]
    
    # 法向量估計 (O(N_t * N_k log N_t) 一次性開銷)
    if key_tgt not in NORMAL_CACHE:
        t_normals = time.time()
        print(f"[INFO] Estimating Normals for target cloud (N={target_pts.shape[0]})...")
        NORMAL_CACHE[key_tgt] = estimate_normals_pca(target_pts, k=30)
        print(f"[INFO] Normals computation time: {time.time() - t_normals:.2f} s")
    target_normals = NORMAL_CACHE[key_tgt]
    
    # === 1️⃣ Threshold（對齊優秀版本）===
    # 優秀版本使用 1.5 * voxel_size，更嚴格更精確
    threshold_icp = voxel_size_icp * 1.5  # 改為 1.5（對齊優秀版本）
    print(f"[INFO] Running Custom Accelerated Pt2Plane ICP. threshold = {threshold_icp:.4f}")

    # === 2️⃣ 單尺度 ICP（對齊優秀版本）===
    max_iter = 30  # 改為 30（對齊優秀版本）
    tolerance_rot = 1e-6
    tolerance_trans = 1e-6

    current_trans = trans_init.copy()  # 使用 copy() 避免修改原始值
    
    for iteration in range(max_iter):
        t_iter = time.time()
        # 1. 變換源點雲 (O(N_s))
        source_pts_curr = (current_trans[:3, :3] @ source_pts_init.T).T + current_trans[:3, 3]
        
        # 2. KDTree 查詢 (O(N_s log N_t)) <--- 每次迭代最耗時部分
        matched_s, matched_t, matched_normals, valid_mask = find_closest_points_cached(
            source_pts_curr, target_pts, target_normals, target_kdtree, threshold_icp
        )
        t_kdtree = time.time() - t_iter
        
        N_matched = matched_s.shape[0]
        if N_matched < 6:
            print(f"   ├─ Iter {iteration:2d}: insufficient matches ({N_matched}). KDTime={t_kdtree:.4f}s")
            break

        # 3. Pt2Plane 矩陣化求解 (O(N_corr) + O(6^3)) <--- 優化加速點
        t_solve_start = time.time()
        delta_T, rmse, _ = compute_transformation_pt2plane_matrix(
            matched_s, matched_t, matched_normals
        )
        t_solve = time.time() - t_solve_start
        
        # 4. 更新整體變換（改為 Right-multiply，對齊優秀版本）
        current_trans = current_trans @ delta_T  # Right-multiply（關鍵修正）
        
        # 5. 檢查收斂 (O(1))
        rot_vec = R.from_matrix(delta_T[:3, :3]).as_rotvec()
        
        # 修正後的 Fitness
        fitness = N_matched / N_s_total # 匹配點數 / 總源點數
        
        print(f"   ├─ Iter {iteration:2d}: RMSE={rmse:.6f}, fit={fitness:.4f}, KDT: {t_kdtree:.4f}s, SOL: {t_solve:.4f}s")
        
        if (np.linalg.norm(delta_T[:3, 3]) < tolerance_trans) and (np.linalg.norm(rot_vec) < tolerance_rot):
            print("   ├─ Converged.")
            break

    # 儲存結果
    final_result = ICPResult(current_trans, rmse, fitness)

    print(f"[INFO] Final Custom Accelerated Pt2Plane ICP → Fitness {final_result.fitness:.4f}, RMSE {final_result.inlier_rmse:.4f}, time={time.time()-t_start:.2f}s")
    return final_result.transformation

def local_icp_algorithm(source_down, target_down, trans_init, voxel_size_reg, mean_depth=None):
    """
    Multi-scale Point-to-Plane ICP (移除 Colored ICP，參數對齊優秀版本)
    --------------------------------
    移除 Colored ICP（因為沒有幫助）
    保留多尺度結構（2 層）
    使用簡單的 Point-to-Plane ICP（像優秀版本）
    使用預設迭代次數（30，像優秀版本）
    參數對齊優秀版本：threshold = voxel_size_reg * 3
    保留時間統計
    """

    t_start = time.time()

    # === 1️⃣ 固定 threshold（對齊優秀版本）===
    threshold_icp = voxel_size_reg * 3.0  # 0.1 * 3 = 0.3（與優秀版本一致）
    print(f"[INFO] Running Multi-scale Point-to-Plane ICP, threshold = {threshold_icp:.4f}")

    # === 2️⃣ 法向量檢查（點雲應該已經有法向量，但檢查一下）===
    if not source_down.has_normals():
        print("[WARN] Source point cloud has no normals, estimating...")
        source_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size_reg * 2, max_nn=60)
        )
    if not target_down.has_normals():
        print("[WARN] Target point cloud has no normals, estimating...")
        target_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size_reg * 2, max_nn=60)
        )

    # === 3️⃣ 多尺度 ICP (2 層版本，使用 Point-to-Plane) ===
    # 參數設計：
    # - Coarse: 較寬鬆的 threshold，幫助初始對齊
    # - Fine: 使用最終 threshold，精確對齊
    # - 迭代次數：使用預設值（30），像優秀版本
    icp_scales = [
        (threshold_icp * 1.5),  # coarse stage: 0.45
        (threshold_icp)         # fine stage: 0.3
    ]

    trans_icp = trans_init
    final_result = None

    for scale_idx, max_dist in enumerate(icp_scales, 1):
        t_scale = time.time()
        stage_name = "Coarse" if scale_idx == 1 else "Fine"
        print(f"[INFO] ICP {stage_name} Stage: max_dist={max_dist:.4f}, using default iterations (30)")

        # 使用簡單的 Point-to-Plane ICP（像優秀版本）
        # 不指定 criteria，使用預設值（max_iteration=30）
        result_icp = o3d.pipelines.registration.registration_icp(
            source_down, target_down,
            max_correspondence_distance=max_dist,
            init=trans_icp,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
            # 使用預設的 ICPConvergenceCriteria（max_iteration=30，像優秀版本）
        )

        print(f"   ├─ fitness={result_icp.fitness:.4f}, RMSE={result_icp.inlier_rmse:.4f}, time={time.time()-t_scale:.2f}s")

        # 提前結束：如果 fitness 已經很好，不需要繼續
        if result_icp.fitness > 0.95:
            print(f"   ├─ Early stop: fitness ({result_icp.fitness:.4f}) already excellent (> 0.95).")
            final_result = result_icp
            break

        trans_icp = result_icp.transformation
        final_result = result_icp

    # === 4️⃣ Fallback (Point-to-Point, 低 fitness 修正) ===
    if final_result.fitness < 0.1:
        print(f"[WARN] Low fitness ({final_result.fitness:.2f})... Retrying with Point-to-Point ICP.")
        final_result = o3d.pipelines.registration.registration_icp(
            source_down,
            target_down,
            max_correspondence_distance=threshold_icp,
            init=trans_init,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            # 使用預設迭代次數
        )
        print(f"[INFO] Fallback ICP → Fitness={final_result.fitness:.4f}, RMSE={final_result.inlier_rmse:.4f}")

    print(f"[INFO] Final ICP → Fitness {final_result.fitness:.4f}, RMSE {final_result.inlier_rmse:.4f}, time={time.time()-t_start:.2f}s")
    return final_result.transformation


# =========================================================================
# 輔助函式 
# =========================================================================

def get_points(pcd_o3d):
    """將 Open3D 點雲物件的點座標轉為 NumPy 陣列 (N, 3)。"""
    # 由於不能呼叫 o3d.utility.Vector3dVector，假設我們能直接存取點陣列
    # 實際程式碼中，您可能需要傳入 pcd.points 的 NumPy 陣列版本
    return np.asarray(pcd_o3d.points)

def skew(v):
    """計算向量 v 的斜對稱矩陣 (Skew-symmetric matrix)"""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def compute_transformation_pt2plane(source_pts: np.ndarray, target_pts: np.ndarray, target_normals: np.ndarray):
    """
    【代替 Open3D 的 TransformationEstimationPointToPlane】
    使用 Gauss-Newton 求解 Point-to-Plane 增量變換。
    
    Args:
        source_pts (np.ndarray): 已變換到當前迭代的來源點 (N, 3)。
        target_pts (np.ndarray): 目標匹配點 (N, 3)。
        target_normals (np.ndarray): 目標點的法向量 (N, 3)。
        
    Returns:
        tuple: (delta_T (4x4), rmse (float), fitness (float))
    """
    N = source_pts.shape[0]
    if N < 6:
        # 至少需要 3 對點，且需要 6 個方程才能解 6 個自由度
        return np.eye(4), 9999.0, 0.0

    # 1. 初始化 A (6x6) = J^T J 和 b (6x1) = - J^T e
    A = np.zeros((6, 6), dtype=np.float64)
    b = np.zeros(6, dtype=np.float64)
    
    total_sq_error = 0.0
    
    for i in range(N):
        p = source_pts[i] # p_i' = R p_i + t (current transformed point)
        q = target_pts[i]
        n = target_normals[i]
        
        # 誤差 e_i (點到平面的距離)
        error = (p - q) @ n
        total_sq_error += error**2
        
        # 雅可比矩陣 J_i (1x6 向量)
        # J_i = [ (p_i' x n_i).T, n_i.T ]
        
        # 旋轉部分的偏導數 J_rot (p_i' x n_i)
        J_rot = skew(p) @ n
        
        J_i = np.hstack([J_rot, n]) # (6,)
        
        # 累積 A 和 b
        # A += J_i^T J_i  --> np.outer(J_i, J_i)
        A += np.outer(J_i, J_i)
        
        # b += J_i^T e_i  --> J_i * error
        b += J_i * error

    # 2. 求解 6x6 線性方程組 A x = -b
    # x = [omega_x, omega_y, omega_z, t_x, t_y, t_z]
    try:
        x = np.linalg.solve(A, -b)
    except np.linalg.LinAlgError:
        # print("[ERROR] Cannot solve Pt2Plane linear system (singular matrix).")
        return np.eye(4), 9999.0, 0.0

    # 3. 將 6x1 增量向量 x 轉為 4x4 增量變換矩陣 delta_T
    R_delta = R.from_rotvec(x[0:3]).as_matrix() # 使用 scipy.spatial.transform
    t_delta = x[3:6]
    
    delta_T = np.eye(4)
    delta_T[:3, :3] = R_delta
    delta_T[:3, 3] = t_delta

    # 4. 計算 RMSE 和 Fitness
    rmse = np.sqrt(total_sq_error / N)
    fitness = N / source_pts.shape[0] # N 是匹配點數，這裡應該是總源點數，但為模擬 Open3D 先用 N

    return delta_T, rmse, fitness

def estimate_normals_pca(pts: np.ndarray, k: int = 30):
    """
    【代替 Open3D 的 estimate_normals】
    使用 PCA (Principal Component Analysis) 估計每個點的法向量。
    
    Args:
        pts (np.ndarray): 點雲座標 (N, 3)。
        k (int): 鄰居數量。
        
    Returns:
        np.ndarray: 法向量 (N, 3)。
    """
    k = 10
    N = pts.shape[0]
    normals = np.zeros((N, 3), dtype=np.float64)
    
    if N < k:
        print("[WARN] Point cloud size is smaller than k. Returning zero normals.")
        return normals
        
    # 建立 KDTree 進行 k-NN 查詢
    kdtree = KDTree(pts)
    
    # 查詢 k 個最近鄰居的索引
    # data: 距離, indices: 索引 (N, k)
    _, indices = kdtree.query(pts, k=k)
    
    # 對每個點計算法向量
    for i in range(N):
        neighbor_indices = indices[i]
        neighbors = pts[neighbor_indices]
        
        centroid = np.mean(neighbors, axis=0)
        
        centered_neighbors = neighbors - centroid
        
        H = centered_neighbors.T @ centered_neighbors
        
        eigen_values, eigen_vectors = np.linalg.eigh(H)
        
        normal = eigen_vectors[:, np.argmin(eigen_values)]
        
        if np.dot(normal, pts[i]) > 0:
             normal = -normal
             
        normals[i] = normal / np.linalg.norm(normal) # 確保是單位向量
        
    print(f"[INFO] PCA Normals estimated for {N} points.")
    return normals

def orient_normals_consistent_tangent_plane(normals: np.ndarray, pts: np.ndarray, k: int = 30):
    """
    【簡化版法線一致化】 
    Open3D 的 orient_normals_consistent_tangent_plane 複雜。
    這裡採用簡單的基於視點/質心的一致化，但實際 Point-to-Plane 依賴準確的方向。
    """
    # 由於我們在 estimate_normals_pca 中已經簡單地讓法向量朝向原點，
    # 這裡可以暫時跳過複雜的 KDTree 尋找和 MST 演算法。
    # 如果需要更高的精度，必須實作一個更精確的法向一致性檢查。
    pass

def find_closest_points(source_pts, target_pts, target_normals, max_dist):
    """
    【核心步驟 1】使用 SciPy KDTree 尋找最近點對，並返回法向量。
    """
    
    # 建立 Target 點雲的 KDTree 
    target_kdtree = KDTree(target_pts)
    
    # 查詢最近鄰，返回距離和索引
    distances, closest_target_indices = target_kdtree.query(
        source_pts, 
        k=1, 
        distance_upper_bound=max_dist
    )
    
    # 篩選出在最大距離 max_dist 內的點對
    valid_mask = distances < max_dist
    
    source_indices = np.where(valid_mask)[0]
    target_indices = closest_target_indices[valid_mask]
    
    # 返回對應的點對和法向量
    return (
        source_pts[source_indices],         # matched_s
        target_pts[target_indices],         # matched_t
        target_normals[target_indices],     # matched_normals
        valid_mask
    )


def compute_transformation_svd(source_pts, target_pts):
    """
    【核心步驟 2】使用 SVD 求解 Point-to-Point 變換 (旋轉 R 和平移 T)。
    這是最小二乘法 ICP 的標準解法。
    """
    N = source_pts.shape[0]
    if N < 3:
        # 點數不足，無法計算有效的變換矩陣
        return np.eye(4), 0.0, 0.0 # R, T, RMSE, fitness

    # 計算質心
    centroid_s = np.mean(source_pts, axis=0)
    centroid_t = np.mean(target_pts, axis=0)

    # 去中心化
    source_centered = source_pts - centroid_s
    target_centered = target_pts - centroid_t

    # 計算協方差矩陣 H
    H = source_centered.T @ target_centered

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # 計算旋轉矩陣 R
    R = Vt.T @ U.T
    
    # 處理反射（若 det(R) = -1）
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # 計算平移向量 t
    t = centroid_t - R @ centroid_s

    # 組合成 4x4 變換矩陣
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    # 計算誤差 (RMSE)
    transformed_s = (R @ source_pts.T).T + t
    sq_diff = np.sum((transformed_s - target_pts)**2)
    rmse = np.sqrt(sq_diff / N)
    
    # 模擬 fitness (這裡使用簡單的匹配率 N/總點數)
    # 實際 O3D fitness 複雜得多
    fitness = N / source_pts.shape[0] # 因為傳入的 source_pts 已經是匹配的，這裡模擬用

    return T, rmse, fitness

# =========================================================================
# 模仿 local_icp_algorithm 的流程
# =========================================================================

class ICPResult:
    """模擬 Open3D RegistrationResult 的結果類別，用於回傳 ICP 狀態。"""
    def __init__(self, T, rmse, fitness):
        self.transformation = T
        self.inlier_rmse = rmse
        self.fitness = fitness

KD_TREE_CACHE = {}
NORMAL_CACHE = {}

def find_closest_points_cached(source_pts, target_pts, target_normals, kdtree, max_dist):
    """使用外部快取的 KDTree 進行查詢"""
    distances, closest_target_indices = kdtree.query(source_pts, k=1, distance_upper_bound=max_dist)
    valid_mask = distances < max_dist
    source_indices = np.where(valid_mask)[0]
    target_indices = closest_target_indices[valid_mask]
    return (
        source_pts[source_indices],
        target_pts[target_indices],
        target_normals[target_indices],
        valid_mask
    )

def my_local_icp_algorithm(source_down, target_down, trans_init, voxel_size_icp, mean_depth=None):
    t_start = time.time()
    source_pts_init = get_points(source_down)
    target_pts = get_points(target_down)
    
    # === 0️⃣ 快取機制 ===
    key_tgt = id(target_down)
    
    if key_tgt not in KD_TREE_CACHE:
        KD_TREE_CACHE[key_tgt] = KDTree(target_pts)
    target_kdtree = KD_TREE_CACHE[key_tgt]
    
    if key_tgt not in NORMAL_CACHE:
        NORMAL_CACHE[key_tgt] = estimate_normals_pca(target_pts, k=30)
    target_normals = NORMAL_CACHE[key_tgt]
    
    # === 1️⃣ Threshold（對齊優秀版本）===
    # 優秀版本使用 1.5 * voxel_size，更嚴格更精確
    threshold_icp = voxel_size_icp * 1.5  # 改為 1.5（對齊優秀版本）
    print(f"[INFO] Running Custom Single-Scale Pt2Plane ICP. threshold = {threshold_icp:.4f}")

    # === 2️⃣ 單尺度 ICP（對齊優秀版本）===
    max_iter = 30  # 改為 30（對齊優秀版本）

    current_trans = trans_init.copy()  # 使用 copy() 避免修改原始值
    
    for iteration in range(max_iter):
        t_iter = time.time()
        source_pts_curr = (current_trans[:3, :3] @ source_pts_init.T).T + current_trans[:3, 3]
        
        # KDTree 查詢 (最慢的部分)
        matched_s, matched_t, matched_normals, valid_mask = find_closest_points_cached(
            source_pts_curr, target_pts, target_normals, target_kdtree, threshold_icp
        )
        t_kdtree = time.time() - t_iter
        
        if matched_s.shape[0] < 6:
            print(f"   ├─ Iter {iteration:2d}: insufficient matches. KDTime={t_kdtree:.4f}s")
            break

        # Pt2Plane 求解
        delta_T, rmse, fitness = compute_transformation_pt2plane(matched_s, matched_t, matched_normals)
        # 更新整體變換（改為 Right-multiply，對齊優秀版本）
        current_trans = current_trans @ delta_T  # Right-multiply（關鍵修正）
        t_solve = time.time() - (t_iter + t_kdtree)
        
        print(f"   ├─ Iter {iteration:2d}: RMSE={rmse:.4f}, fit={fitness:.4f}, KDT: {t_kdtree:.4f}s, SOL: {t_solve:.4f}s")
        
        # 檢查收斂
        rot_vec = R.from_matrix(delta_T[:3, :3]).as_rotvec()
        if (np.linalg.norm(delta_T[:3, 3]) < 1e-4) or (np.linalg.norm(rot_vec) < 1e-4):
            print("   ├─ Converged.")
            break

    # 儲存結果
    final_result = ICPResult(current_trans, rmse, fitness)

    print(f"[INFO] Final Pt2Plane ICP → Fitness {final_result.fitness:.4f}, RMSE {final_result.inlier_rmse:.4f}, time={time.time()-t_start:.2f}s")
    return final_result.transformation



def reconstruct(args):
    """
    Reconstruction pipeline using batch processing architecture.

    Steps:
    1. Load all RGB, depth images
    2. Convert all to point clouds (batch)
    3. Preprocess all point clouds (downsample, normals, FPFH)
    4. Global Registration (RANSAC) - pairwise
    5. Local Registration (ICP) - pairwise
    6. Transform and merge all point clouds to world coordinate
    7. Accumulate estimated camera poses
    """
    start_time = time.time() 

    # === 路徑 ===
    rgb_paths = sorted(
        glob.glob(os.path.join(args.data_root, "rgb", "*.png")),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )
    depth_paths = sorted(
        glob.glob(os.path.join(args.data_root, "depth", "*.png")),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )
    gt_pose_path = os.path.join(args.data_root, "GT_pose.npy")
    if not os.path.exists(gt_pose_path):
        raise FileNotFoundError(f"Cannot find GT_pose.npy in {args.data_root}")

    gt_pose = np.load(gt_pose_path)
    print("gt_pose[0]:",gt_pose[0])
    
    N = len(rgb_paths)
    if N == 0:
        raise RuntimeError("No RGB/Depth images found. Check data_root path!")

    print(f"[INFO] Loaded {N} frames from {args.data_root}")

    # === 超參數 ===
    voxel_size_reg = 0.1  # 統一體素大小，用於 RANSAC 和 ICP

    # === 1️⃣ 批次載入所有點雲 ===
    t_load = time.time()
    print(f"\n[INFO] Step 1: Loading all point clouds...")
    pcds = []
    for i in range(N):
        rgb_cv = cv2.imread(rgb_paths[i])
        rgb_cv = cv2.cvtColor(rgb_cv, cv2.COLOR_BGR2RGB)
        depth_cv = cv2.imread(depth_paths[i], cv2.IMREAD_UNCHANGED)
        pcd = depth_image_to_point_cloud(rgb_cv, depth_cv)
        pcds.append(pcd)
        if (i + 1) % 20 == 0:
            print(f"  Loaded {i + 1}/{N} point clouds...")
    print(f"[INFO] All point clouds loaded. Time: {time.time() - t_load:.2f} s")

    # === 2️⃣ 統一預處理所有點雲 ===
    t_preprocess = time.time()
    print(f"\n[INFO] Step 2: Preprocessing all point clouds (voxel={voxel_size_reg}m)...")
    pcds_down = []
    pcds_fpfh = []
    for i, pcd in enumerate(pcds):
        pcd_down, pcd_fpfh = preprocess_point_cloud(pcd, voxel_size_reg)
        pcds_down.append(pcd_down)
        pcds_fpfh.append(pcd_fpfh)
        if (i + 1) % 20 == 0:
            print(f"  Preprocessed {i + 1}/{N} point clouds...")
    print(f"[INFO] All point clouds preprocessed. Time: {time.time() - t_preprocess:.2f} s")

    # === 3️⃣ 批次配準（逐對處理）===
    print(f"\n[INFO] Step 3: Pairwise registration...")
    pairwise_T = []
    
    for i in range(N - 1):
        t_pair = time.time()
        print(f"\n[Frame {i} → {i+1}]")
        
        source_down = pcds_down[i + 1]
        target_down = pcds_down[i]
        source_fpfh = pcds_fpfh[i + 1]
        target_fpfh = pcds_fpfh[i]

        # 初始變換矩陣
        if i == 0:
            trans = np.eye(4)
        else:
            trans = pairwise_T[i - 1]

        # === 4️⃣ 全域初始對齊（改進的 RANSAC 策略）===
        t_ransac = time.time()
        
        # 根據樓層採用不同的 RANSAC 策略
        if args.floor == 2:
            # Floor 2 複雜場景：每 3 幀才執行一次 RANSAC
            if i % 3 == 0:
                print(f"[INFO] 執行 RANSAC (Frame {i} is divisible by 3)")
                result_ransac = execute_global_registration(
                    source_down, target_down, source_fpfh, target_fpfh, voxel_size_reg
                )
                print(f"[INFO] RANSAC fitness: {result_ransac.fitness:.4f}")
                
                # 只有在 fitness 足夠好時才使用 RANSAC 結果
                if result_ransac.fitness > 0.2:
                    trans = result_ransac.transformation
                    print("[INFO] RANSAC accepted (fitness > 0.2)")
                else:
                    print("[WARN] RANSAC rejected (low fitness), using previous transform")
            else:
                print(f"[INFO] 跳過 RANSAC (Frame {i} not divisible by 3)")
        else:
            # Floor 1 或其他場景：每幀都執行 RANSAC
            result_ransac = execute_global_registration(
                source_down, target_down, source_fpfh, target_fpfh, voxel_size_reg
            )
            trans = result_ransac.transformation
            print(f"[INFO] RANSAC fitness: {result_ransac.fitness:.4f}")
        
        print(f"[INFO] Global Registration time: {time.time() - t_ransac:.2f} s")

        # === 5️⃣ 局部 ICP 精修 ===
        t_icp = time.time()
        if args.version == 'open3d':
            print("[INFO] Calling Open3D local_icp_algorithm (with Colored ICP)")
            result_icp = local_icp_algorithm(source_down, target_down, trans, voxel_size_reg)
        elif args.version == 'my_icp':
            print("[INFO] Calling Custom my_local_icp_algorithm")
            result_icp = my_local_icp_algorithm_accelerated(source_down, target_down, trans, voxel_size_reg)
        else:
            raise ValueError(f"Unknown ICP version: {args.version}. Must be 'open3d' or 'my_icp'.")
        
        print(f"[INFO] Local ICP time: {time.time() - t_icp:.2f} s")
        print(f"[INFO] Pair {i}→{i+1} total time: {time.time() - t_pair:.2f} s")
        
        pairwise_T.append(result_icp)

    print(f"\n[INFO] All pairwise registrations done. Total time: {time.time() - start_time:.2f} s")

    # === 6️⃣ 點雲合併與軌跡累積 ===
    t_merge = time.time()
    print(f"\n[INFO] Step 4: Merging point clouds and accumulating camera trajectory...")
    
    global_pcd = o3d.geometry.PointCloud()
    T_world = [np.eye(4)]
    pred_cam_poses = [np.eye(4)]  # 第一幀的姿態是單位矩陣

    for i in range(N):
        if i > 0:
            # 直接累積變換：T_i = T_{i-1} @ ΔT_{i-1→i}
            T_world.append(T_world[i - 1] @ pairwise_T[i - 1])
            pred_cam_poses.append(T_world[i])
        
        # 複製點雲並變換到世界座標
        pcd_i_world = o3d.geometry.PointCloud(pcds[i])
        pcd_i_world.transform(T_world[i])
        global_pcd += pcd_i_world

    print(f"[INFO] Point clouds merged. Time: {time.time() - t_merge:.2f} s")
    print(f"[INFO] Total points: {len(global_pcd.points)}")
    
    print("\n[INFO] Reconstruction Done.")
    print(f"[INFO] Total time: {time.time() - start_time:.2f} seconds")
    
    return global_pcd, np.array(pred_cam_poses)

def remove_ceiling_points(pcd, y_threshold=0.3):
    """篩掉天花板以上的點（y > threshold）"""
    pts = np.asarray(pcd.points)
    # y倒過來
    mask = pts[:, 1] > - y_threshold
    pcd_filtered = o3d.geometry.PointCloud()
    pcd_filtered.points = o3d.utility.Vector3dVector(pts[mask])
    if pcd.has_colors():
        pcd_filtered.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask])
    return pcd_filtered

def umeyama_alignment(ground_truth, prediction):
    """
    執行 Umeyama 對齊 (Sim(3)) 於真實與預測的相機軌跡之間。
    Perform Umeyama alignment (Sim(3)) between ground truth and predicted camera trajectories.
    
    Args:
        ground_truth (np.ndarray): Ground truth positions (N, 3)
        prediction (np.ndarray): Predicted positions (N, 3)
    
    Returns:
        tuple: (scale, R, t) - 尺度、旋轉矩陣、平移向量
    """
    assert ground_truth.shape == prediction.shape, "Error: Ground truth and prediction have different shapes."
    
    # 計算質心
    mu_gt = np.mean(ground_truth, axis=0)
    mu_pred = np.mean(prediction, axis=0)
    
    # 去中心化
    gt_centered = ground_truth - mu_gt
    pred_centered = prediction - mu_pred
    
    # 計算協方差矩陣
    W = pred_centered.T @ gt_centered
    
    # SVD 分解
    U, S, Vt = np.linalg.svd(W)
    V = Vt.T
    
    # 計算旋轉矩陣
    R = V @ U.T
    
    # 處理反射情況
    if np.linalg.det(R) < 0:
        V[:, -1] *= -1
        R = V @ U.T
    
    # 計算尺度
    if (S.sum() < 1e-8) or (np.sum(pred_centered ** 2) < 1e-8):
        scale = 1.0
    else:
        scale = S.sum() / np.sum(pred_centered ** 2)
    
    print(f"[INFO] Umeyama 尺度係數 (Scale factor): {scale:.6f}")
    
    # 計算平移
    t = mu_gt - scale * R @ mu_pred
    
    return scale, R, t

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--floor', type=int, default=1, help="Floor number to reconstruct (1 or 2)")
    parser.add_argument('-v', '--version', type=str, default='open3d', help='ICP version to use: open3d or my_icp')
    parser.add_argument('--data_root', type=str, help='(Optional) Override data collection path')
    args = parser.parse_args()

    # --- 1. Set data path ---
    if args.data_root is None:
        if args.floor == 100:
            args.data_root = "data_collection/first_floor_short/"
        elif args.floor == 1:
            args.data_root = "data_collection/first_floor/"
        elif args.floor == 2:
            args.data_root = "data_collection/second_floor/"
        else:
            raise ValueError("Floor must be 1 or 2.")
    
    # --- 2. Run reconstruction ---
    result_pcd, pred_cam_poses = reconstruct(args)
    points = np.asarray(result_pcd.points)
    print(points[:,1].min(), points[:,1].max())

    # --- 3. Load Ground Truth data ---
    gt_pose_path = os.path.join(args.data_root, "GT_pose.npy")
    gt_poses_raw = np.load(gt_pose_path)

    # --- 4. Umeyama Alignment (Sim(3)) ---
    # 提取預測的相機位置
    pred_cam_pos = np.array([pose[:3, 3] for pose in pred_cam_poses])
    
    # 提取 GT 相機位置（只取前 3 個元素，即 x, y, z）
    gt_cam_pos = gt_poses_raw[:len(pred_cam_pos), :3]
    
    # 執行 Umeyama 對齊來計算最佳的尺度、旋轉、平移
    print("\n[INFO] 執行 Umeyama 對齊 (Performing Umeyama Alignment)...")
    scale, R_umeyama, t_umeyama = umeyama_alignment(gt_cam_pos, pred_cam_pos)
    
    # 構建 4x4 變換矩陣
    T_umeyama = np.eye(4)
    T_umeyama[:3, :3] = scale * R_umeyama
    T_umeyama[:3, 3] = t_umeyama
    
    # 應用 Umeyama 對齊到點雲和軌跡
    result_pcd.transform(T_umeyama)
    
    # 對齊預測的相機位置
    pred_cam_pos_aligned = scale * (R_umeyama @ pred_cam_pos.T).T + t_umeyama
    
    # --- 5. Remove ceiling points (後處理，使用 quantile 方式) ---
    print(f"\n[INFO] Removing ceiling points using quantile method...")
    points = np.asarray(result_pcd.points)
    if args.floor == 1:
        mask = points[:, 1] < np.quantile(points[:, 1], 0.55)
    else:
        mask = points[:, 1] < np.quantile(points[:, 1], 0.5)
    result_pcd = result_pcd.select_by_index(np.where(mask)[0])
    print(f"[INFO] Ceiling removed. Remaining points: {len(result_pcd.points)}")

    # --- 6. Calculate L2 distance (after Umeyama alignment) ---
    distances = np.linalg.norm(pred_cam_pos_aligned - gt_cam_pos, axis=1)
    mean_l2_distance = np.mean(distances)
    print(f"\n[結果] Mean L2 distance (after Umeyama alignment): {mean_l2_distance:.4f}")

    # --- 7. Prepare visualization objects ---
    print(f"[INFO] 準備可視化物件... GT points: {len(gt_cam_pos)}, Pred points: {len(pred_cam_pos_aligned)}")
    
    geometries_to_draw = [result_pcd]
    
    # a. Ground Truth trajectory (black)
    if len(gt_cam_pos) >= 2:
        gt_points = o3d.utility.Vector3dVector(gt_cam_pos)
        gt_lines = [[i, i + 1] for i in range(len(gt_cam_pos) - 1)]
        gt_lineset = o3d.geometry.LineSet(
            points=gt_points,
            lines=o3d.utility.Vector2iVector(gt_lines)
        )
        gt_lineset.paint_uniform_color([0, 0, 0])
        geometries_to_draw.append(gt_lineset)
        print(f"[INFO] GT trajectory created: {len(gt_lines)} lines")
    else:
        print(f"[WARN] GT trajectory 點數不足 ({len(gt_cam_pos)}), 跳過可視化")

    # b. Aligned estimated trajectory (red)
    if len(pred_cam_pos_aligned) >= 2:
        pred_points = o3d.utility.Vector3dVector(pred_cam_pos_aligned)
        pred_lines = [[i, i + 1] for i in range(len(pred_cam_pos_aligned) - 1)]
        pred_lineset = o3d.geometry.LineSet(
            points=pred_points,
            lines=o3d.utility.Vector2iVector(pred_lines)
        )
        pred_lineset.paint_uniform_color([1, 0, 0])
        geometries_to_draw.append(pred_lineset)
        print(f"[INFO] Predicted trajectory created: {len(pred_lines)} lines")
    else:
        print(f"[WARN] Predicted trajectory 點數不足 ({len(pred_cam_pos_aligned)}), 跳過可視化")

    # --- 8. Visualize ---
    
    print("\n[INFO] Visualizing result... Close the window to exit.")

    # === ：輸出結果 ===
    save_name = f"reconstruction_F{args.floor}_{args.version}.ply"
    o3d.io.write_point_cloud(save_name, result_pcd)
    print(f"[SAVE] Reconstructed point cloud saved to: {save_name}")

    # === 🔴 輸出紅線（預測軌跡） ===
    if len(pred_cam_pos_aligned) >= 2:
        pred_save_name = f"trajectory_pred_F{args.floor}_{args.version}.ply"
        o3d.io.write_line_set(pred_save_name, pred_lineset)
        print(f"[SAVE] Predicted trajectory (red) saved to: {pred_save_name}")

    # === ⚫ 輸出黑線（GT 軌跡） ===
    if len(gt_cam_pos) >= 2:
        gt_save_name = f"trajectory_gt_F{args.floor}_{args.version}.ply"
        o3d.io.write_line_set(gt_save_name, gt_lineset)
        print(f"[SAVE] Ground truth trajectory (black) saved to: {gt_save_name}")

    o3d.visualization.draw_geometries(geometries_to_draw)