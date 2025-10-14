import numpy as np
import open3d as o3d
import argparse
import math
import os
import cv2

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
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 200)
        # criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500) # 「最大嘗試次數」,confidence： ANSAC 找到了一個不錯的對齊（例如，有 100 個點是匹配的），然後在接下來的 500 次新嘗試中，都沒能找到一個能讓超過 100 個點匹配的更好方案，它就會停止，

    )
    print("[INFO] RANSAC Done. Inlier RMSE:", result.inlier_rmse)
    return result


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
# def local_icp_algorithm(source_down, target_down, trans_init, voxel_size_fine, mean_depth=None):
#     """
#     Local ICP (Balanced Fast Version)
#     --------------------------------
#     ✅ Multi-scale 2 層 (快速收斂)
#     ✅ Adaptive threshold (依平均深度動態調整)
#     ✅ 單次法向估計 + 緩存檢查
#     ✅ 限制 ICP 迭代次數 (10~15)
#     ✅ 平衡速度與穩定性（加速約 35~45%）
#     """
#     import open3d as o3d
#     import time
#     t_start = time.time()

#     # === 1️⃣ Adaptive threshold ===
#     if mean_depth is not None:
#         threshold_icp = np.clip(mean_depth * 0.05, 0.01, 0.05)
#     else:
#         threshold_icp = voxel_size_fine * 3.0
#     print(f"[INFO] Running Optimized ICP (balanced) threshold = {threshold_icp:.4f}")

#     # === 2️⃣ 法向量估計（只在第一次時做）===
#     if not source_down.has_normals():
#         source_down.estimate_normals(
#             search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size_fine * 3, max_nn=30)
#         )
#     if not target_down.has_normals():
#         target_down.estimate_normals(
#             search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size_fine * 3, max_nn=30)
#         )

#     # === 3️⃣ Multi-scale ICP (2 層版本) ===
#     icp_scales = [
#         (threshold_icp * 2.5, 10),  # coarse stage
#         (threshold_icp, 15)         # fine stage
#     ]

#     trans_icp = trans_init
#     final_result = None

#     for scale_idx, (max_dist, max_iter) in enumerate(icp_scales, 1):
#         t_scale = time.time()
#         print(f"[INFO] ICP Scale {scale_idx}: max_dist={max_dist:.4f}, iter={max_iter}")

#         # result_icp = o3d.pipelines.registration.registration_icp(
#         #     source_down,
#         #     target_down,
#         #     max_correspondence_distance=max_dist,
#         #     init=trans_icp,
#         #     estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
#         #     criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
#         # )

#         try:
#             result_icp = o3d.pipelines.registration.registration_colored_icp(
#                 source_down,
#                 target_down,
#                 max_correspondence_distance=max_dist,
#                 init=trans_icp,
#                 estimation_method=o3d.pipelines.registration.TransformationEstimationForColoredICP(),
#                 criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
#             )
#         except RuntimeError as e:
#             print(f"[WARN] Colored ICP failed ({str(e)}). Falling back to Point-to-Plane ICP.")
#             result_icp = o3d.pipelines.registration.registration_icp(
#                 source_down,
#                 target_down,
#                 max_correspondence_distance=max_dist,
#                 init=trans_icp,
#                 estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
#                 criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
#             )



#         print(f"   ├─ fitness={result_icp.fitness:.4f}, RMSE={result_icp.inlier_rmse:.4f}, time={time.time()-t_scale:.2f}s")

#         # 若 coarse 版本已收斂得很好，可提前結束
#         if result_icp.fitness > 0.85 and scale_idx == 1:
#             print("   ├─ Early stop: coarse stage already high fitness.")
#             final_result = result_icp
#             break

#         trans_icp = result_icp.transformation
#         final_result = result_icp

#     # === 4️⃣ Fallback (Point-to-Point, 低 fitness 修正) ===
#     if final_result.fitness < 0.1:
#         print(f"[WARN] Low fitness ({final_result.fitness:.2f})... Retrying with Point-to-Point ICP.")
#         final_result = o3d.pipelines.registration.registration_icp(
#             source_down,
#             target_down,
#             max_correspondence_distance=threshold_icp,
#             init=trans_init,
#             estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
#             criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=15)
#         )
#         print(f"[INFO] Fallback ICP → Fitness={final_result.fitness:.4f}, RMSE={final_result.inlier_rmse:.4f}")

#     print(f"[INFO] Final ICP → Fitness {final_result.fitness:.4f}, RMSE {final_result.inlier_rmse:.4f}, time={time.time()-t_start:.2f}s")
#     return final_result.transformation

# v4
def local_icp_algorithm(source_down, target_down, trans_init, voxel_size_fine, mean_depth=None):
    """
    Local ICP (Upgraded with a robust try-except block)
    ----------------------------------------------------------
    ✅ [Fix] Catches the "No correspondences found" error to prevent crashes.
    ✅ Falls back to a more stable Point-to-Plane ICP if Colored ICP fails.
    """
    import open3d as o3d
    import numpy as np

    threshold_icp = voxel_size_fine * 3.0
    print(f"[INFO] Running Colored ICP with threshold = {threshold_icp:.4f}")

    # --- 1. 法向量估計 ---
    if not source_down.has_normals():
        source_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size_fine * 2, max_nn=30)
        )
    if not target_down.has_normals():
        target_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size_fine * 2, max_nn=30)
        )
    
    result_icp = None # Initialize result_icp to None

    # --- 2. 執行 Colored ICP with Error Handling ---
    try:
        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source_down,
            target_down,
            max_correspondence_distance=threshold_icp,
            init=trans_init,
            estimation_method=o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=20)
        )
        print(f"[INFO] Colored ICP Done. Fitness: {result_icp.fitness:.4f}, RMSE: {result_icp.inlier_rmse:.4f}")
    except RuntimeError as e:
        print(f"[ERROR] Colored ICP failed with error: {e}")
        # If it fails, result_icp will remain None, forcing the fallback.

    # --- 3. Fallback to Point-to-Plane ICP ---
    # If Colored ICP failed (result_icp is None) or had low fitness
    if result_icp is None or result_icp.fitness < 0.1:
        if result_icp: # This check is for the low fitness case
             print(f"[WARN] Low fitness ({result_icp.fitness:.2f})... Retrying with Point-to-Plane ICP.")
        else: # This is for the RuntimeError case
             print(f"[WARN] No correspondences found. Retrying with Point-to-Plane ICP.")
        
        result_icp = o3d.pipelines.registration.registration_icp(
            source_down,
            target_down,
            max_correspondence_distance=threshold_icp,
            init=trans_init,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20)
        )
        print(f"[INFO] Fallback ICP Done. Fitness: {result_icp.fitness:.4f}, RMSE: {result_icp.inlier_rmse:.4f}")

    return result_icp.transformation

def my_local_icp_algorithm(source_down, target_down, trans_init, voxel_size):
    # TODO: Write your own ICP function
    raise NotImplementedError
    return result

import os
import cv2
import glob
import numpy as np
import open3d as o3d
import time
def reconstruct(args):
    """
    Reconstruction pipeline (Upgraded with Keyframing and Pose Graph Optimization)
    -----------------------------------------------------------------------------
    ✅ [Change #1] Keyframe機制，跳過非關鍵幀的RANSAC，大幅加速
    ✅ [Change #2] 建立Pose Graph，記錄所有相對位姿
    ✅ 在流程最後執行全局優化，從根本上修正累積漂移 (Drift)
    """
    import open3d as o3d
    import numpy as np
    import time
    import cv2
    import glob
    import os

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
    print("gt_pose[0]:", gt_pose[0])

    N = len(rgb_paths)
    if N == 0:
        raise RuntimeError("No RGB/Depth images found. Check data_root path!")

    print(f"[INFO] Loaded {N} frames from {args.data_root}")

    # === 超參數 ===
    voxel_size_coarse = 0.1  # 用於 RANSAC 的體素大小
    voxel_size_fine = 0.01  # 用於 ICP 的體素大小
    key_interval = 2        # 每 5 幀設置一個關鍵幀

    # === 初始化 Pose Graph 和其他變數 ===
    pose_graph = o3d.pipelines.registration.PoseGraph()
    current_pose = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(current_pose))
    
    pcd_fragments = [] # 用於儲存每一幀的原始點雲
    last_transform = np.identity(4) # 用於非關鍵幀的初始猜測

    # === 逐幀對齊與建立姿態圖 ===
    for i in range(N - 1):
        print(f"\n[Frame {i} → {i+1}]")

        # --- 1. 讀取影像並轉為點雲 ---
        rgb_s = cv2.imread(rgb_paths[i])
        depth_s = cv2.imread(depth_paths[i], cv2.IMREAD_UNCHANGED)
        source = depth_image_to_point_cloud(rgb_s, depth_s)
        source = remove_ceiling_points(source)

        rgb_t = cv2.imread(rgb_paths[i+1])
        depth_t = cv2.imread(depth_paths[i+1], cv2.IMREAD_UNCHANGED)
        target = depth_image_to_point_cloud(rgb_t, depth_t)
        target = remove_ceiling_points(target)
        
        # 在迴圈開始時就儲存原始點雲，以供最後合併
        if i == 0:
            pcd_fragments.append(source)
        pcd_fragments.append(target)

        # --- 2. Keyframe 邏輯：決定 trans_init ---
        if (i + 1) % key_interval == 0:
            # --- 這是關鍵幀：執行完整的 RANSAC 來獲得穩定的初始對齊 ---
            print("[INFO] Keyframe detected. Running Global Registration...")
            source_down = preprocess_point_cloud(source, voxel_size_coarse)
            target_down = preprocess_point_cloud(target, voxel_size_coarse)
            
            radius_feature = voxel_size_coarse * 5.0
            source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
            target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
            
            result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size_coarse)
            trans_init = result_ransac.transformation
        else:
            # --- 這是普通幀：使用上一幀的運動結果作為初始猜測，跳過 RANSAC ---
            print("[INFO] Odometry frame. Using previous transform as init.")
            trans_init = last_transform

        # --- 3. 局部 ICP 精修 (使用新的 Colored ICP 函式) ---
        source_down_fine = source.voxel_down_sample(voxel_size_fine)
        target_down_fine = target.voxel_down_sample(voxel_size_fine)
        
        # 注意：我們傳遞 voxel_size_fine 給函式，而不是 threshold_icp
        result_icp = local_icp_algorithm(source_down_fine, target_down_fine, trans_init, voxel_size_fine)
        last_transform = result_icp # 儲存這次的結果，供下一普通幀使用

        # --- 4. 更新姿態圖 ---
        # `result_icp` 是 T_{target <- source}，即 T_{i+1 <- i}
        # 更新當前姿態：P_{i+1} = P_i @ inv(T_{i+1 <- i})
        # 注意: Open3D 的 pose 是 T_{world <- cam}
        # p_world = P_i @ p_i_local
        # p_i_local = inv(T_{i+1 <- i}) @ p_{i+1}_local
        # p_world = P_i @ inv(result_icp) @ p_{i+1}_local
        # 所以 P_{i+1} = P_i @ inv(result_icp)
        current_pose = current_pose @ np.linalg.inv(result_icp)
        
        # 添加新的節點和邊到姿態圖
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(current_pose))
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(i, i + 1, result_icp, uncertain=False)
        )

    print("\n[INFO] All frames processed. Starting Global Optimization...")

    # === 全局優化姿態圖 ===
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=voxel_size_coarse, # 使用粗體素尺寸作為距離閾值
        edge_prune_threshold=0.25,
        reference_node=0)
    
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option)
    
    print("[INFO] Global Optimization finished.")

    # === 使用優化後的姿態合併所有點雲 ===
    print("[INFO] Merging point clouds with optimized poses...")
    result_pcd = o3d.geometry.PointCloud()
    pred_cam_poses = []


    for i in range(len(pcd_fragments)):
        optimized_pose = pose_graph.nodes[i].pose
        pred_cam_poses.append(optimized_pose)

        fragment_transformed = pcd_fragments[i].transform(optimized_pose)
        result_pcd += fragment_transformed

        # 只在第一次非空時計算 auto_voxel
        if i == 0:
            bbox_diag = np.linalg.norm(result_pcd.get_max_bound() - result_pcd.get_min_bound())
            auto_voxel = max(bbox_diag * 0.005, 0.001)  # 加最小值防止0
            print(f"[INFO] auto voxel_size = {auto_voxel:.4f}")

        # 🔹 對目前累積點雲進行體素濾波
        result_pcd = result_pcd.voxel_down_sample(voxel_size=auto_voxel)
    print("\n[INFO] Reconstruction Done.")
    print(f"[INFO] Total time: {time.time() - start_time:.2f} seconds")

    return result_pcd, np.array(pred_cam_poses)

# def process_frame(rgb_path, depth_path, voxel_size_coarse):
#     """
#     處理單一畫面：讀取影像、轉成點雲、去天花板、降採樣並計算 FPFH 特徵。
    
#     返回:
#         pcd (o3d.geometry.PointCloud): 原始點雲 (已去天花板)
#         pcd_down (o3d.geometry.PointCloud): 粗降採樣後的點雲
#         pcd_fpfh (o3d.pipelines.registration.Feature): FPFH 特徵
#     """
#     # 讀取影像
#     rgb = cv2.imread(rgb_path)
#     depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    
#     # 轉點雲並去天花板
#     pcd = depth_image_to_point_cloud(rgb, depth)
#     pcd = remove_ceiling_points(pcd) # 假設你有 remove_ceiling_points 函式
    
#     # 粗降採樣 (for RANSAC)
#     pcd_down = preprocess_point_cloud(pcd, voxel_size_coarse)
    
#     # 計算 FPFH 特徵
#     radius_feature = voxel_size_coarse * 5.0
#     pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
#         pcd_down,
#         o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
#     )
    
#     return pcd, pcd_down, pcd_fpfh

# def reconstruct(args):
#     """
#     優化後的重建流程，重複利用上一幀的計算結果。
#     """
#     start_time = time.time() 
#     rgb_paths = sorted(
#         glob.glob(os.path.join(args.data_root, "rgb", "*.png")),
#         key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
#     )
#     depth_paths = sorted(
#         glob.glob(os.path.join(args.data_root, "depth", "*.png")),
#         key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
#     )
#     # === 路徑和參數設定 (與之前相同) ===

#     N = len(rgb_paths)
#     print(f"[INFO] Loaded {N} frames from {args.data_root}")
#     voxel_size_coarse = 0.1
#     voxel_size_fine = 0.01
#     threshold_icp = voxel_size_fine * 3

#     # === 初始化 ===
#     # 預處理第 0 幀，作為第一輪的 source
#     print("[INFO] Pre-processing frame 0...")
#     source_pcd, source_down, source_fpfh = process_frame(rgb_paths[0], depth_paths[0], voxel_size_coarse)
    
#     # 初始化最終點雲和相機姿態
#     result_pcd = source_pcd
#     current_pose = np.eye(4)
#     pred_cam_poses = [current_pose]
#     start_time = time.time()
#     # === 優化後的逐幀對齊迴圈 ===
#     for i in range(1, N):
#         t0 = time.time()
#         print(f"\n[Frame {i-1} → {i}]")

#         # 1️⃣ 處理新的 target 幀 (第 i 幀)
#         # 這裡節省了重新處理 source (第 i-1 幀) 的時間
#         t_proc = time.time()
#         target_pcd, target_down, target_fpfh = process_frame(rgb_paths[i], depth_paths[i], voxel_size_coarse)
#         print(f"[INFO] Process new frame time: {time.time() - t_proc:.2f} s")

#         # 2️⃣ 全域初始對齊 (邏輯不變)
#         t_ransac = time.time()
#         result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size_coarse)
#         trans_init = result_ransac.transformation
#         print(f"[INFO] Global Registration time: {time.time() - t_ransac:.2f} s")

#         # 3️⃣ 局部 ICP 精修 (邏輯不變)
#         t_icp = time.time()
#         source_down_fine = source_pcd.voxel_down_sample(voxel_size_fine)
#         target_down_fine = target_pcd.voxel_down_sample(voxel_size_fine)
#         result_icp = local_icp_algorithm(source_down_fine, target_down_fine, trans_init, threshold_icp)
#         print(f"[INFO] Local ICP time: {time.time() - t_icp:.2f} s")

#         # 4️⃣ 更新姿態並合併點雲 (邏輯不變)
#         relative_transform = np.linalg.inv(result_icp)
#         current_pose = current_pose @ relative_transform
#         pred_cam_poses.append(current_pose)
        
#         target_transformed = target_pcd.transform(current_pose) # 使用未降採樣的點雲來合併
#         result_pcd += target_transformed
        
#         print(f"[INFO] Frame {i} merged. Total points = {len(result_pcd.points)}")
        
#         # 5️⃣ 關鍵：為下一輪準備 source 資料
#         # 將當前的 target 變為下一輪的 source
#         source_pcd = target_pcd
#         source_down = target_down
#         source_fpfh = target_fpfh

#         print(f"Total time for frame: {time.time()-t0:.2f} s , time passed:{time.time()-start_time:.2f} s")

#     print("\n[INFO] Reconstruction Done.")
#     print(f"[INFO] Total time: {time.time() - start_time:.2f} seconds")
    
#     return result_pcd, np.array(pred_cam_poses)



def remove_ceiling_points(pcd, y_threshold=0.5):
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

    # # === Remove ceiling (based on Y height) ===
    # points = np.asarray(result_pcd.points)

    # # Step 1️⃣: 觀察範圍
    # print(f"[INFO] Point height range: minY={points[:,1].min():.2f}, maxY={points[:,1].max():.2f}")

    # # Step 2️⃣: 過濾上方天花板 (例如移除 Y > 1.8 m 的點)
    # ceiling_threshold = 0.85 # 依實際場景調整
    # filtered_idx = points[:,1] > - ceiling_threshold

    # # Step 3️⃣: 建立新點雲
    # filtered_pcd = o3d.geometry.PointCloud()
    # filtered_pcd.points = o3d.utility.Vector3dVector(points[filtered_idx])
    # filtered_pcd.colors = o3d.utility.Vector3dVector(np.asarray(result_pcd.colors)[filtered_idx])

    # print(f"[INFO] Removed {len(points) - np.sum(filtered_idx)} ceiling points.")
    # result_pcd = filtered_pcd

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