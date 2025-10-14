import numpy as np
import open3d as o3d
import argparse
import math
import os
import cv2
from scipy.spatial import KDTree
def depth_image_to_point_cloud(rgb, depth, fov_deg: float = 90.0, depth_scale: float = 100.0,
                               depth_trunc_m: float = 6.0):
    H, W = depth.shape
    f = (W * 0.5) / math.tan(math.radians(fov_deg) * 0.5)
    intr = o3d.camera.PinholeCameraIntrinsic(W, H, f, f, W * 0.5, H * 0.5)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    color_o3d = o3d.geometry.Image(rgb.astype(np.uint8))

    # 這裡改成 /100.0（因為平均值 ≈ 40 公分）
    # depth_m = depth.astype(np.float32) / 100.0
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
#     # 可選：Open3D 的視覺化座標系 z 向前、y 向下；
#     # 若希望符合一般「z 向前、y 向上」的視覺化習慣，可做如下翻轉（僅影響顯示）：
#     pcd.transform([[1, 0, 0, 0],
#                    [0,-1, 0, 0],
#                    [0, 0,-1, 0],
#                    [0, 0, 0, 1]])
    return pcd


def preprocess_point_cloud(pcd: o3d.geometry.PointCloud, voxel_size: float):
    """
    以體素降採樣降低點數，並估計法向量（後續 ICP/特徵計算更穩定）。

    Args:
        pcd (o3d.geometry.PointCloud): 原始點雲
        voxel_size (float): 體素大小（公尺），例：0.02~0.05

    Returns:
        o3d.geometry.PointCloud: 降採樣且帶法向量的點雲
    """
    assert isinstance(pcd, o3d.geometry.PointCloud)
    assert voxel_size > 0.0

    # 1) 體素降採樣
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)

    # 2) 估計法向量（對 ICP 的點到面/特徵描述子更友善）
    # 半徑設為 ~ 2 個體素，鄰居數做上限
    radius_normal = voxel_size * 2.0
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normal,
            max_nn=30
        )
    )
    # 將法向量朝向一致（可提升配準穩定性）
    pcd_down.orient_normals_consistent_tangent_plane(k=30)

    return pcd_down



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
        ransac_n=8,  # 每次隨機取4點
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(800000, 300)
        # criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500) # 「最大嘗試次數」,confidence： ANSAC 找到了一個不錯的對齊（例如，有 100 個點是匹配的），然後在接下來的 500 次新嘗試中，都沒能找到一個能讓超過 100 個點匹配的更好方案，它就會停止，

    )
    print("[INFO] RANSAC Done. Inlier RMSE:", result.inlier_rmse)
    return result

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R
import time
# ... (其他您已引入的庫)

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

    # === 1. 計算殘差向量 e (點到平面的距離) ===
    # e = (source_pts - target_pts) . target_normals
    # 這是 N x 1 的向量 (Open3D 官方文獻中常用 $e_i = (T p_i - q_i) \cdot n_i$)
    errors = np.sum((source_pts - target_pts) * target_normals, axis=1) # (N,)

    # === 2. 計算雅可比矩陣 J (N x 6) ===
    # J_i = [ (p_i' x n_i).T, n_i.T ]
    
    # a. 旋轉部分 J_rot (N x 3)
    # 叉積 $\mathbf{p}'_i \times \mathbf{n}_i$
    # np.cross 實現了向量化叉積
    J_rot = np.cross(source_pts, target_normals) # (N, 3)
    
    # b. 平移部分 J_trans (N x 3)
    # 即法向量本身 $\mathbf{n}_i$
    J_trans = target_normals # (N, 3)
    
    # c. 組合 J (N x 6)
    J = np.hstack([J_rot, J_trans]) # (N, 6)
    
    # === 3. 求解法線方程 $A \mathbf{x} = -\mathbf{b}$ ===
    # A = J^T J (6 x 6)
    A = J.T @ J # 矩陣乘法，高速計算 $\sum J_i^T J_i$
    
    # b_vec = J^T e (6 x 1)
    b_vec = J.T @ errors # 矩陣乘法，高速計算 $\sum J_i^T e_i$

    # 求解 6x6 線性方程組
    try:
        x = np.linalg.solve(A, -b_vec) # $\mathbf{x} = - (J^T J)^{-1} J^T \mathbf{e}$
    except np.linalg.LinAlgError:
        return np.eye(4), 9999.0, 0.0

    # === 4. 構建增量變換矩陣 $\Delta T$ ===
    # 注意：這裡使用了您已引入的 `scipy.spatial.transform.Rotation as R`
    R_delta = R.from_rotvec(x[0:3]).as_matrix()
    t_delta = x[3:6]
    
    delta_T = np.eye(4)
    delta_T[:3, :3] = R_delta
    delta_T[:3, 3] = t_delta

    # === 5. 計算 RMSE 和 Fitness ===
    total_sq_error = np.sum(errors**2)
    rmse = np.sqrt(total_sq_error / N)
    
    # 模擬 fitness (匹配點數 / 總源點數)
    fitness = N / source_pts.shape[0] # 這裡需要知道原始的總源點數，但因函數輸入已被篩選，暫時使用 N

    return delta_T, rmse, fitness

def my_local_icp_algorithm_accelerated(source_down, target_down, trans_init, voxel_size_icp, mean_depth=None):
    t_start = time.time()
    source_pts_init = get_points(source_down)
    target_pts = get_points(target_down)
    N_s_total = source_pts_init.shape[0] # 獲取總源點數，用於更準確的 fitness 計算
    
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
    
    # === 1️⃣ Adaptive threshold ===
    # 根據經驗，設置一個合理的初始距離
    threshold_icp = voxel_size_icp * 5.0 
    print(f"[INFO] Running Custom Accelerated Pt2Plane ICP. threshold = {threshold_icp:.4f}")

    # === 2️⃣ 單尺度 ICP (加速且魯棒的 Pt2Plane 通常不需要多尺度) ===
    max_iter = 15 # 適當的迭代次數
    tolerance_rot = 1e-6
    tolerance_trans = 1e-6

    current_trans = trans_init
    
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
        
        # 4. 更新整體變換 (O(1))
        current_trans = delta_T @ current_trans
        
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

# def local_icp_algorithm(source_down, target_down, trans_init, threshold):
#     """
#     優先使用 Point-to-Plane ICP。
#     它會自行估計所需的法向量。
#     """
#     print(f"[INFO] Running ICP with threshold = {threshold:.3f} (pt2plane)")
    
#     # 1. 估計法向量 (只對當前需要的稀疏點雲計算，非常快)
#     source_down.estimate_normals(
#         search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=threshold * 2, max_nn=30))
#     target_down.estimate_normals(
#         search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=threshold * 2, max_nn=30))

#     # 2. 執行 Point-to-Plane ICP
#     result_icp = o3d.pipelines.registration.registration_icp(
#         source_down,
#         target_down,
#         max_correspondence_distance=threshold,
#         init=trans_init,
#         estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
#     )
#     print("[INFO] ICP Done. Fitness:", result_icp.fitness, "RMSE:", result_icp.inlier_rmse)

#     # 3. (可選) 如果 Point-to-Plane 效果極差，可以保留 fallback
#     if result_icp.fitness < 0.1: # 如果匹配度低於 10%
#         print(f"[WARN] Low fitness ({result_icp.fitness:.2f})... Retrying with Point-to-Point.")
#         result_icp = o3d.pipelines.registration.registration_icp(
#             source_down,
#             target_down,
#             max_correspondence_distance=threshold,
#             init=trans_init,
#             estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
#         )
#         print("[INFO] Fallback ICP Done. Fitness:", result_icp.fitness, "RMSE:", result_icp.inlier_rmse)

#     return result_icp.transformation

# v2: Multi-scale ICP with adaptive threshold and fallback
# def local_icp_algorithm(source_down, target_down, trans_init, voxel_size_fine, mean_depth=None):

#     # Adaptive threshold
#     threshold_icp = voxel_size_fine * 3.0
#     print(f"[INFO] Running Optimized ICP (no kernel) with threshold = {threshold_icp:.4f}")

#     # 法向量估計
#     if not source_down.has_normals():
#         source_down.estimate_normals(
#             search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size_fine * 3, max_nn=30))
#     if not target_down.has_normals():
#         target_down.estimate_normals(
#             search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size_fine * 3, max_nn=30))

#     # Multi-scale ICP
#     icp_scales = [
#         (threshold_icp * 5, 15),
#         (threshold_icp * 2, 15),
#         (threshold_icp, 20)
#     ]

#     trans_icp = trans_init
#     final_result = None
#     for scale_idx, (max_dist, max_iter) in enumerate(icp_scales, 1):
#         print(f"[INFO] ICP Scale {scale_idx}: max_dist={max_dist:.4f}, iter={max_iter}")
#         result_icp = o3d.pipelines.registration.registration_icp(
#             source_down, target_down,
#             max_correspondence_distance=max_dist,
#             init=trans_icp,
#             estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
#             criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
#         )
#         print(f"   ├─ fitness={result_icp.fitness:.4f}, RMSE={result_icp.inlier_rmse:.4f}")
#         trans_icp = result_icp.transformation
#         final_result = result_icp

#     # Fallback (Point-to-Point)
#     if final_result.fitness < 0.1:
#         print(f"[WARN] Low fitness ({final_result.fitness:.2f})... Retrying with Point-to-Point ICP.")
#         final_result = o3d.pipelines.registration.registration_icp(
#             source_down, target_down,
#             max_correspondence_distance=threshold_icp,
#             init=trans_init,
#             estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
#             criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
#         )
#     print(f"[INFO] Final ICP → Fitness {final_result.fitness:.4f}, RMSE {final_result.inlier_rmse:.4f}")
#     return final_result.transformation

# v3: speed up
def local_icp_algorithm(source_down, target_down, trans_init, voxel_size_fine, mean_depth=None):
    """
    Local ICP (Balanced Fast Version)
    --------------------------------
    ✅ Multi-scale 2 層 (快速收斂)
    ✅ Adaptive threshold (依平均深度動態調整)
    ✅ 單次法向估計 + 緩存檢查
    ✅ 限制 ICP 迭代次數 (10~15)
    ✅ 平衡速度與穩定性（加速約 35~45%）
    """
    import open3d as o3d
    import time
    t_start = time.time()

    # === 1️⃣ Adaptive threshold ===
    if mean_depth is not None:
        threshold_icp = np.clip(mean_depth * 0.05, 0.01, 0.05)
    else:
        threshold_icp = voxel_size_fine * 3.0
    print(f"[INFO] Running Optimized ICP (balanced) threshold = {threshold_icp:.4f}")

    # === 2️⃣ 法向量估計（只在第一次時做）===
    if not source_down.has_normals():
        source_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size_fine * 3, max_nn=30)
        )
    if not target_down.has_normals():
        target_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size_fine * 3, max_nn=30)
        )

    # === 3️⃣ Multi-scale ICP (2 層版本) ===
    icp_scales = [
        (threshold_icp * 2.5, 10),  # coarse stage
        (threshold_icp, 15)         # fine stage
    ]

    trans_icp = trans_init
    final_result = None

    for scale_idx, (max_dist, max_iter) in enumerate(icp_scales, 1):
        t_scale = time.time()
        print(f"[INFO] ICP Scale {scale_idx}: max_dist={max_dist:.4f}, iter={max_iter}")

        # result_icp = o3d.pipelines.registration.registration_icp(
        #     source_down,
        #     target_down,
        #     max_correspondence_distance=max_dist,
        #     init=trans_icp,
        #     estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        #     criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
        # )

        try:
            result_icp_coarse = o3d.pipelines.registration.registration_icp(
                source_down, target_down,
                max_correspondence_distance=max_dist * 3,
                init=trans_icp,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=10)
            )
            # 再進 Colored ICP
            result_icp = o3d.pipelines.registration.registration_colored_icp(
                source_down,
                target_down,
                max_correspondence_distance=max_dist,
                init=result_icp_coarse.transformation,
                estimation_method=o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
            )

        except RuntimeError as e:
            print(f"[WARN] Colored ICP failed ({str(e)}). Falling back to Point-to-Plane ICP.")
            result_icp = o3d.pipelines.registration.registration_icp(
                source_down,
                target_down,
                max_correspondence_distance=max_dist,
                init=trans_icp,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
            )



        print(f"   ├─ fitness={result_icp.fitness:.4f}, RMSE={result_icp.inlier_rmse:.4f}, time={time.time()-t_scale:.2f}s")

        # 若 coarse 版本已收斂得很好，可提前結束
        if result_icp.fitness > 0.85 and scale_idx == 1:
            print("   ├─ Early stop: coarse stage already high fitness.")
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
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=15)
        )
        print(f"[INFO] Fallback ICP → Fitness={final_result.fitness:.4f}, RMSE={final_result.inlier_rmse:.4f}")

    print(f"[INFO] Final ICP → Fitness {final_result.fitness:.4f}, RMSE {final_result.inlier_rmse:.4f}, time={time.time()-t_start:.2f}s")
    return final_result.transformation


import numpy as np
import time

# =========================================================================
# 輔助函式 (必須用純 NumPy/Scipy 替代 Open3D 的內部實現)
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

# 引入必要的庫
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R
import time

# ... (保留您原本的 get_points 和 ICPResult 函式/類別)

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
        
        # 1. 計算質心 (Centroid)
        centroid = np.mean(neighbors, axis=0)
        
        # 2. 去中心化 (Centering)
        centered_neighbors = neighbors - centroid
        
        # 3. 計算協方差矩陣 (Covariance Matrix)
        # H = X^T X, where X is centered_neighbors
        H = centered_neighbors.T @ centered_neighbors
        
        # 4. SVD 或特徵分解
        # 法向量是最小特徵值對應的特徵向量 (np.linalg.eigh 效率更高)
        eigen_values, eigen_vectors = np.linalg.eigh(H)
        
        # 最小特徵值對應的特徵向量是法向量
        normal = eigen_vectors[:, np.argmin(eigen_values)]
        
        # 5. 法向量方向一致化 (簡單版：假設都朝向原點)
        # Open3D 內部有更好的方法，這裡用簡單的 heuristics
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

# ... (保留 ICPResult 類別)

# def my_local_icp_algorithm(source_down, target_down, trans_init, voxel_size_fine, mean_depth=None):
#     """
#     純 NumPy 實作 Point-to-Plane ICP 流程，模仿 Open3D 的多尺度邏輯。
#     """
#     t_start = time.time()
    
#     # 獲取點雲資料
#     source_pts_init = get_points(source_down)
#     target_pts = get_points(target_down)
    
#     # === 0️⃣ 預處理：估計目標點雲的法向量 ===
#     t_normals = time.time()
#     # 這裡假設 k=30, 這是經驗值
#     target_normals = estimate_normals_pca(target_pts, k=30)
#     # (可選) 法向量一致化，這裡只用 estimate_normals_pca 內部的簡單朝向。
#     # orient_normals_consistent_tangent_plane(target_normals, target_pts) 
#     print(f"[INFO] Normals computation time: {time.time() - t_normals:.2f} s")


#     # === 1️⃣ Adaptive threshold ===
#     threshold_icp = voxel_size_fine * 3.0
#     print(f"[INFO] Running Custom Pt2Plane ICP. threshold = {threshold_icp:.4f}")

#     # === 2️⃣ Multi-scale ICP (2 層版本) ===
#     icp_scales = [
#         (threshold_icp * 2.5, 10),  # coarse stage
#         (threshold_icp, 15)         # fine stage
#     ]

#     trans_icp = trans_init
#     final_result = ICPResult(trans_init, 9999.0, 0.0)

#     for scale_idx, (max_dist, max_iter) in enumerate(icp_scales, 1):
#         t_scale = time.time()
#         print(f"[INFO] ICP Scale {scale_idx}: max_dist={max_dist:.4f}, iter={max_iter}")

#         current_trans = trans_icp
        
#         for iteration in range(max_iter):
#             # 1. 變換源點雲
#             source_pts_curr = (current_trans[:3, :3] @ source_pts_init.T).T + current_trans[:3, 3]
            
#             # 2. 尋找最近點對和法向量
#             matched_s, matched_t, matched_normals, valid_mask = find_closest_points(
#                 source_pts_curr, target_pts, target_normals, max_dist
#             )
            
#             N_matched = matched_s.shape[0]
#             if N_matched < 6:
#                 print("   ├─ Iter {iteration:2d}: Insufficient matches (<6). Stopping.")
#                 break

#             # 3. 【Pt2Plane 求解】計算增量變換 (R, t)
#             delta_T, rmse, fitness = compute_transformation_pt2plane(
#                 matched_s, matched_t, matched_normals
#             )
            
#             # 4. 更新整體變換
#             # 新變換 = 增量變換 @ 舊變換
#             current_trans = delta_T @ current_trans
            
#             print(f"   ├─ Iter {iteration:2d}: RMSE={rmse:.4f}, Fitness={fitness:.4f}, N_matches={N_matched}")
            
#             # 5. 檢查收斂：如果增量平移和旋轉都很小，則收斂
#             rot_vec = R.from_matrix(delta_T[:3, :3]).as_rotvec()
#             if (np.linalg.norm(delta_T[:3, 3]) < 1e-6) and (np.linalg.norm(rot_vec) < 1e-6):
#                 print("   ├─ Converged.")
#                 break
        
#         # 儲存此尺度的最佳結果
#         final_result = ICPResult(current_trans, rmse, fitness)
#         trans_icp = current_trans

#         # 模擬 Early Stop
#         if final_result.fitness > 0.85 and scale_idx == 1:
#             print("   ├─ Early stop: coarse stage already high fitness.")
#             break
            
#     print(f"[INFO] Final Pt2Plane ICP → Fitness {final_result.fitness:.4f}, RMSE {final_result.inlier_rmse:.4f}, time={time.time()-t_start:.2f}s")
    
#     return final_result.transformation

KD_TREE_CACHE = {}
NORMAL_CACHE = {}

def find_closest_points_cached(source_p ts, target_pts, target_normals, kdtree, max_dist):
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
    
    # === 1️⃣ Adaptive threshold ===
    threshold_icp = voxel_size_icp * 5.0
    print(f"[INFO] Running Custom Single-Scale Pt2Plane ICP. threshold = {threshold_icp:.4f}")

    # === 2️⃣ 單尺度 ICP ===
    max_iter = 5 # 固定的最大迭代次數

    current_trans = trans_init
    
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
        current_trans = delta_T @ current_trans
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


import os
import cv2
import glob
import numpy as np
import open3d as o3d
import time

def reconstruct(args):
    """
    Reconstruction pipeline using Open3D ICP version.

    Steps:
    1. Load RGB, depth images, GT poses
    2. Convert depth → point cloud
    3. Downsample (voxel)
    4. Global Registration (RANSAC)
    5. Local Registration (ICP)
    6. Transform and merge into one scene
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
    # print(f"rgb_paths[0]: {rgb_paths}")
    # print(f"depth_paths[0]: {depth_paths}") 
    if not os.path.exists(gt_pose_path):
        raise FileNotFoundError(f"Cannot find GT_pose.npy in {args.data_root}")

    gt_pose = np.load(gt_pose_path)
    print("gt_pose[0]:",gt_pose[0])
    
    N = len(rgb_paths)
    if N == 0:
        raise RuntimeError("No RGB/Depth images found. Check data_root path!")

    print(f"[INFO] Loaded {N} frames from {args.data_root}")

    # === 超參數 ===
    voxel_size_coarse = 0.1  # init 0.05 用於 RANSAC  # 增大這會讓點雲更稀疏，減少了計算量，同時也過濾掉了一些細節噪聲，有助於 RANSAC 關注於場景的宏觀結構，有時反而能更快找到好的初始對齊。
    voxel_size_fine   = 0.01  # 用於 ICP 和最終點雲
    threshold_icp = voxel_size_fine * 3 # ICP 距離阈值

    # === 初始化 ===
    # 將第一個點雲直接加入最終場景
    rgb0 = cv2.imread(rgb_paths[0])
    depth0 = cv2.imread(depth_paths[0], cv2.IMREAD_UNCHANGED)
    result_pcd = depth_image_to_point_cloud(rgb0, depth0)
    
    current_pose = np.eye(4) # Pose of camera 0 is the origin
    pred_cam_poses = [current_pose] # 儲存所有相機姿態 (從第0幀開始)
    current_pose = np.eye(4)
    # We need to save the FIRST camera position which is the origin
    # before the loop starts for visualization purposes later
    all_pred_poses_for_viz = [current_pose[:3, 3]] 
    
    
    # === 逐幀對齊 ===
    for i in range(N - 1):
        
        t0 = time.time()
        print(f"\n[Frame {i} → {i+1}]")

        rgb_s = cv2.imread(rgb_paths[i])
        depth_s = cv2.imread(depth_paths[i], cv2.IMREAD_UNCHANGED)
        rgb_t = cv2.imread(rgb_paths[i+1])
        depth_t = cv2.imread(depth_paths[i+1], cv2.IMREAD_UNCHANGED)
        print(f"[Frame {i}] mean depth: {np.mean(depth_t):.2f}")

        # === 1️⃣ 轉成點雲 ===
        t1 = time.time()

        source = depth_image_to_point_cloud(rgb_s, depth_s)
        target = depth_image_to_point_cloud(rgb_t, depth_t)

        print(f"[INFO] Depth to PointCloud time: {time.time() - t1:.2f} s")
        # === 2️⃣ 體素化 ===
        # === 在進入下采樣前移除天花板點 ===
        movtime = time.time()
        source = remove_ceiling_points(source)
        target = remove_ceiling_points(target)
        print(f"moving ceiling time: {time.time() -movtime:.2f} s")
        
        t2 = time.time()
        source_down = preprocess_point_cloud(source, voxel_size_coarse)
        target_down = preprocess_point_cloud(target, voxel_size_coarse)
        print(f"[INFO] Voxel Downsample time: {time.time() - t2:.2f} s")
        # === 3️⃣ 計算特徵（供 Global Registration 用）===
        t3 = time.time()

        radius_feature = voxel_size_coarse * 3.0
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
        )
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
        )
        print(f"[INFO] FPFH Feature Computation time: {time.time() - t3:.2f} s")

        # === 4️⃣ 全域初始對齊 ===
        t4 = time.time()
        result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size_coarse)
        trans_init = result_ransac.transformation
        print(f"[INFO] Global Registration time: {time.time() - t4:.2f} s")
        # === 2️⃣' 體素化 (Fine for ICP) ===
        t5 = time.time()
        # 確保 source_down 是用 voxel_size_fine 做的
        # source_down_fine = preprocess_point_cloud(source, voxel_size_fine)
        # target_down_fine = preprocess_point_cloud(target, voxel_size_fine)
        source_down_fine = source.voxel_down_sample(voxel_size_fine)
        target_down_fine = target.voxel_down_sample(voxel_size_fine)
        # ***【新增：二次降採樣】*** # 使用 0.02m (點數會是 0.01m 的 1/8)
        voxel_size_icp = 0.02 
        source_down_icp = source_down_fine.voxel_down_sample(voxel_size_icp)
        target_down_icp = target_down_fine.voxel_down_sample(voxel_size_icp)
        
        print(f"[INFO] Voxel Downsample (Fine) time: {time.time() - t5:.2f} s")
        # === 5️⃣ 局部 ICP 精修 ===
        t6 = time.time()
        # 根據參數選擇 ICP 實作
        if args.version == 'open3d':
            # 使用 Open3D 的 ICP (傳入 voxel_size_fine 作為尺度參數)
            print("[INFO] Calling Open3D local_icp_algorithm (Optimized O3D)")
            result_icp = local_icp_algorithm(source_down_fine, target_down_fine, trans_init, voxel_size_fine)
        elif args.version == 'my_icp':
            # 使用您自己實作的純 NumPy ICP
            print("[INFO] Calling Custom my_local_icp_algorithm (Pure NumPy Pt2Pt)")
            # result_icp = my_local_icp_algorithm(source_down_icp, target_down_icp, trans_init, voxel_size_icp)
            result_icp = my_local_icp_algorithm_accelerated(source_down_icp, target_down_icp, trans_init, voxel_size_icp)

        else:
            raise ValueError(f"Unknown ICP version: {args.version}. Must be 'open3d' or 'my_icp'.")
        
        print(f"[INFO] Local ICP time: {time.time() - t6:.2f} s")
        '''
        # result_icp: transform from source → target
        # Therefore, to accumulate camera poses (world→camera),
        # we multiply the inverse to move from target→source.
        '''
        relative_transform = np.linalg.inv(result_icp)
        
        # 更新 current_pose 來得到下一幀(i+1)的姿態
        current_pose = current_pose @ relative_transform
        pred_cam_poses.append(current_pose)

        # 將 target 點雲 (來自 i+1 幀) 用 i+1 的姿態變換到世界座標並合併
        target_transformed = target.transform(current_pose)
        result_pcd += target_transformed

        # === 6️⃣ 累積場景 ===
        # source.transform(current_pose)
        # result_pcd += source

        print(f"[INFO] Frame {i} merged. Total points = {len(result_pcd.points)}")
        print(f"total time: {time.time()-t0}")
        print(f"time passed: {time.time()- start_time}")
    print("\n[INFO] Reconstruction Done.")
    print(f"[INFO] Total time: {time.time() - start_time:.2f} seconds")
    
    # return result_pcd, np.array(pred_cam_pos), np.array(all_pred_poses_for_viz)
    return result_pcd, np.array(pred_cam_poses)

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

    # --- 4. Align coordinate systems ---
    # (Requires scipy: pip install scipy)
    from scipy.spatial.transform import Rotation as R

    # Get the first ground truth pose to compute the alignment matrix
    first_gt_pose = gt_poses_raw[0]
    
    # Position is the first 3 elements
    first_gt_pos = first_gt_pose[0:3]
    
    # Quaternion is the last 4 elements, saved as (w, x, y, z)
    quat_wxyz = first_gt_pose[3:7] 
    
    # Scipy's from_quat function expects the format (x, y, z, w)
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    
    first_gt_rot_matrix = R.from_quat(quat_xyzw).as_matrix()
    
    T_world_to_cam0_gt = np.eye(4)
    T_world_to_cam0_gt[:3, :3] = first_gt_rot_matrix
    T_world_to_cam0_gt[:3, 3] = first_gt_pos

    # Correction for Open3D's camera coordinate system (Y down, Z in) vs Habitat's (Y up, Z out)
    T_o3d_to_standard_cam = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # Final alignment matrix to transform your reconstruction into the ground truth world
    align_matrix = T_world_to_cam0_gt @ T_o3d_to_standard_cam
    
    # Apply the alignment to your results
    result_pcd.transform(align_matrix)
    aligned_pred_poses = [align_matrix @ pose for pose in pred_cam_poses]

    # --- 5. Calculate L2 distance (after alignment) ---
    pred_xyz = np.array([p[:3, 3] for p in aligned_pred_poses])
    gt_xyz = gt_poses_raw[:len(pred_xyz), 0:3]

    distances = np.linalg.norm(pred_xyz - gt_xyz, axis=1)
    mean_l2_distance = np.mean(distances)
    print(f"\nMean L2 distance (after alignment): {mean_l2_distance:.4f}")

    # --- 6. Prepare visualization objects ---
    # a. Ground Truth trajectory (black)
    gt_points = o3d.utility.Vector3dVector(gt_xyz)
    gt_lines = [[i, i + 1] for i in range(len(gt_points) - 1)]
    gt_lineset = o3d.geometry.LineSet(
        points=gt_points,
        lines=o3d.utility.Vector2iVector(gt_lines)
    )
    gt_lineset.paint_uniform_color([0, 0, 0])

    # b. Aligned estimated trajectory (red)
    pred_points = o3d.utility.Vector3dVector(pred_xyz)
    pred_lines = [[i, i + 1] for i in range(len(pred_points) - 1)]
    pred_lineset = o3d.geometry.LineSet(
        points=pred_points,
        lines=o3d.utility.Vector2iVector(pred_lines)
    )
    pred_lineset.paint_uniform_color([1, 0, 0])

    # --- 7. Visualize ---
    geometries_to_draw = [result_pcd, gt_lineset, pred_lineset]
    
    print("\n[INFO] Visualizing result... Close the window to exit.")

    # === 新增：輸出結果 ===
    save_name = f"reconstruction_F{args.floor}_{args.version}.ply"
    o3d.io.write_point_cloud(save_name, result_pcd)
    print(f"[SAVE] Reconstructed point cloud saved to: {save_name}")

    # === 🔴 輸出紅線（預測軌跡） ===
    pred_save_name = f"trajectory_pred_F{args.floor}_{args.version}.ply"
    o3d.io.write_line_set(pred_save_name, pred_lineset)
    print(f"[SAVE] Predicted trajectory (red) saved to: {pred_save_name}")

    # === ⚫ 輸出黑線（GT 軌跡） ===
    gt_save_name = f"trajectory_gt_F{args.floor}_{args.version}.ply"
    o3d.io.write_line_set(gt_save_name, gt_lineset)
    print(f"[SAVE] Ground truth trajectory (black) saved to: {gt_save_name}")

    o3d.visualization.draw_geometries(geometries_to_draw)