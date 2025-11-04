import numpy as np
import open3d as o3d
import os
import matplotlib.pyplot as plt

# 讀取資料
current_path = os.path.dirname(os.path.abspath(__file__)) 

points = np.load(f"{current_path}/semantic_3d_pointcloud/point.npy")        # (N, 3)
colors = np.load(f"{current_path}/semantic_3d_pointcloud/color01.npy")    # (N, 3)  或 color01.npy

print(points.shape, colors.shape)
print("unique semantic IDs:", np.unique(colors)[:10])

# print(points.shape, colors.shape)
y_min, y_max = points[:, 1].min(), points[:, 1].max()
print("y_min =", y_min, "y_max =", y_max)

floor_threshold = y_min + 0.012
ceiling_threshold = y_max - 0.07

mask = (points[:, 1] > floor_threshold) & (points[:, 1] < ceiling_threshold)
filtered_points = points[mask]
filtered_colors = colors[mask]

print(f"Filtered {len(points) - len(filtered_points)} points removed")

pcd_filtered = o3d.geometry.PointCloud()
pcd_filtered.points = o3d.utility.Vector3dVector(filtered_points)
pcd_filtered.colors = o3d.utility.Vector3dVector(filtered_colors)

o3d.visualization.draw_geometries([pcd_filtered])

x_coords = filtered_points[:, 0]
z_coords = filtered_points[:, 2]


import json
data = {
    "xmin": x_coords.min(),
    "xmax": x_coords.max(),
    "zmin": z_coords.min(),
    "zmax": z_coords.max()
}
OUTPUT_FILENAME = "coordinate_bounds.json"
with open(OUTPUT_FILENAME,'w',encoding='utf-8') as f:
    json.dump(data, f, indent=4)

plt.figure(figsize=(10, 10))

plt.scatter(
    x_coords, 
    z_coords, # 投影到 XZ 平面
    s=1, 
    c=filtered_colors, 
    marker='.'
)

plt.axis('equal') 
plt.axis('off') 
plt.title("2D Semantic Map of apartment_0 First Floor (X-Z Projection)")

# 儲存地圖

DPI = 100
plt.savefig(f"map{DPI}.png", bbox_inches='tight', pad_inches=0,dpi=DPI)

plt.show()

print("2D semantic map saved as map.png")

plt.figure(figsize=(10, 10))
plt.scatter(
    x_coords,
    z_coords,
    s=1,
    c=filtered_colors,
    marker='.'
)
plt.axis('equal')
plt.axis('off')
plt.title("2D Semantic Map (with min/max boundary)")

# 取四個角點 四個角的座標 (x, z)
xmin, xmax = data["xmin"], data["xmax"]
zmin, zmax = data["zmin"], data["zmax"]
corner_points = [
    (xmin, zmin),  # 左下
    (xmin, zmax),  # 左上
    (xmax, zmin),  # 右下
    (xmax, zmax),  # 右上
]
plt.plot(
    [xmin, xmin, xmax, xmax, xmin],
    [zmin, zmax, zmax, zmin, zmin],
    color="red",
    linewidth=1.2,
    label="Min-Max Bounds"
)

for (x, z) in corner_points:
    plt.scatter(x, z, c='red', s=10)
    plt.text(x, z, f"({x:.2f},{z:.2f})", color='red', fontsize=6)

plt.legend()
plt.savefig("map_minmax.png", bbox_inches='tight', pad_inches=0)
plt.close()
print("map_minmax.png saved (with boundary box and coordinates)")

plt.show()