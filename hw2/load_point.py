import numpy as np
import open3d as o3d

# 讀取資料
current_path = "hw2/semantic_3d_pointcloud/"   

points = np.load(f"{current_path}point.npy")        # (N, 3)
colors = np.load(f"{current_path}color01.npy")    # (N, 3)  或 color01.npy

print(points.shape, colors.shape)
print("unique semantic IDs:", np.unique(colors)[:10])
# 若是顏色範圍為 0~255，轉為 0~1
# if colors.max() > 1:
#     colors = colors / 255.0

# print(points.shape, colors.shape)
y_min, y_max = points[:, 1].min(), points[:, 1].max()
print("y_min =", y_min, "y_max =", y_max)

floor_threshold = y_min + 0
ceiling_threshold = y_max - 0.07

mask = (points[:, 1] > floor_threshold) & (points[:, 1] < ceiling_threshold)
filtered_points = points[mask]
filtered_colors = colors[mask]

print(f"Filtered {len(points) - len(filtered_points)} points removed")

pcd_filtered = o3d.geometry.PointCloud()
pcd_filtered.points = o3d.utility.Vector3dVector(filtered_points)
pcd_filtered.colors = o3d.utility.Vector3dVector(filtered_colors)

o3d.visualization.draw_geometries([pcd_filtered])


o3d.visualization.draw_geometries([pcd])
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
plt.scatter(filtered_points[:, 0], filtered_points[:, 2],
            c=filtered_colors, s=1)
plt.axis("equal")
plt.title("2D Semantic Map (Floor Removed)")
plt.savefig("map.png", dpi=300)
plt.show()
