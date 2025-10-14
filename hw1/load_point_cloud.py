import open3d as o3d
import os
import argparse

def load_geometry_if_exists(path, geom_type="point_cloud"):
    """安全載入 PLY 檔，若不存在則略過"""
    if not os.path.exists(path):
        print(f"[WARN] File not found: {path}")
        return None
    
    if geom_type == "point_cloud":
        geom = o3d.io.read_point_cloud(path)
    elif geom_type == "line_set":
        geom = o3d.io.read_line_set(path)
    else:
        raise ValueError(f"Unknown geometry type: {geom_type}")
    
    print(f"[LOAD] Loaded {geom_type}: {path} ({len(geom.points)} points)")
    return geom


def load_reconstruction(floor: int, version: str = "open3d"):
    """
    根據樓層載入對應的重建結果 (支援 F1, F2, F100)
    會同時載入:
      - reconstruction_F{floor}_{version}.ply
      - trajectory_pred_F{floor}_{version}.ply
      - trajectory_gt_F{floor}_{version}.ply
    """
    # 組合檔名
    base_name = f"F{floor}_{version}"
    recon_path = f"reconstruction_{base_name}.ply"
    pred_path  = f"trajectory_pred_{base_name}.ply"
    gt_path    = f"trajectory_gt_{base_name}.ply"

    # 載入各物件
    geometries = []
    pcd = load_geometry_if_exists(recon_path, "point_cloud")
    if pcd: geometries.append(pcd)

    pred_lines = load_geometry_if_exists(pred_path, "line_set")
    if pred_lines: geometries.append(pred_lines)

    gt_lines = load_geometry_if_exists(gt_path, "line_set")
    if gt_lines: geometries.append(gt_lines)

    if not geometries:
        raise FileNotFoundError(f"[ERROR] No reconstruction files found for floor {floor}.")
    
    print(f"\n[INFO] Visualization for Floor {floor} ({version}) — {len(geometries)} geometries loaded.")
    o3d.visualization.draw_geometries(geometries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--floor', type=int, default=100, help="Floor number to load (1, 2, or 100)")
    parser.add_argument('-v', '--version', type=str, default='open3d', help='Version used during reconstruction (e.g., open3d or my_icp)')
    args = parser.parse_args()

    load_reconstruction(args.floor, args.version)
