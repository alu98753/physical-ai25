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
# 若是顏色範圍為 0~255，轉為 0~1
# if colors.max() > 1:
#     colors = colors / 255.0

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


# 步驟 1 & 2: 準備 XZ 座標和顏色
# 選擇 X 座標 (第一列) 和 Z 座標 (第三列)
# 注意：在許多 3D 座標系中，Z 軸通常代表深度或高度。但在 2D 俯視圖中，我們通常使用 X 和 Y 軸。
# 依照作業要求，我們使用 (X, Z) 繪圖 。
x_coords = filtered_points[:, 0]
z_coords = filtered_points[:, 2]

# 步驟 3: 繪製散點圖
# 為了讓地圖看起來更像俯視圖，你可以將 Z 座標視為 Y 座標。

plt.figure(figsize=(10, 10))

# 繪製散點圖: (X, Z) with 顏色
# s=1 讓點非常小，模擬像素。marker='.' 確保點是圓點。
plt.scatter(
    x_coords, 
    z_coords, # 投影到 XZ 平面
    s=1, 
    c=filtered_colors, 
    marker='.'
)

# 設置座標軸：
# 確保 X 和 Z 的比例尺一致，防止地圖變形
plt.axis('equal') 
# 移除座標軸刻度，讓它看起來更像地圖
plt.axis('off') 
# 設置標題（可選）
plt.title("2D Semantic Map of apartment_0 First Floor (X-Z Projection)")

# 步驟 4: 儲存地圖
# 儲存地圖為 "map.png" [cite: 24]
# bbox_inches='tight' 和 pad_inches=0 確保只儲存點雲的部分，沒有多餘白邊
plt.savefig("map.png", bbox_inches='tight', pad_inches=0)

plt.show()

print("2D semantic map saved as map.png")