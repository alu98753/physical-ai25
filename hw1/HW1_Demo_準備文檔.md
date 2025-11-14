# HW1 Demo 準備文檔

## 目錄

1. [Task 1: BEV Projection](#task-1-bev-projection)
2. [Task 2: 3D Reconstruction](#task-2-3d-reconstruction)
3. [核心知識點](#核心知識點)
4. [實現細節](#實現細節)
5. [可能的問題與回答](#可能的問題與回答)

---

## Task 1: BEV Projection

### 1.1 任務要求

- **數據收集**：同時保存一對前視圖（front view）和 BEV（Bird's Eye View）圖像
- **BEV 投影**：在 BEV 圖像上選擇地面區域的點，標記這些點圍成的區域，並將該區域從 BEV 圖像投影到透視圖（前視圖）圖像

### 1.2 相機參數設置

根據 spec：

- **Camera1 (front)**：
  - Position: (0, 1, 0)
  - Orientation: (0, 0, 0) - 水平向前
- **Camera2 (BEV)**：
  - Position: (0, 2.5, 0)
  - Orientation: (-π/2, 0, 0) - 向下看（pitch = -90°）

### 1.3 內參數設置

- **解析度**：512 × 512
- **FOV（視野角）**：90 度（水平和垂直）
- **內參矩陣計算**：
  ```python
  f = 0.5 * width / tan(FOV/2)
  K = [[f, 0, width/2],
       [0, f, height/2],
       [0, 0, 1]]
  ```

### 1.4 投影流程

1. **在 BEV 圖像上點擊選擇點**：使用滑鼠左鍵點擊地面區域
2. **座標轉換流程**：
   - BEV 像素座標 → BEV 相機座標系（假設 Z=0 在地面）
   - BEV 相機座標系 → 世界座標系
   - 世界座標系 → Front 相機座標系
   - Front 相機座標系 → Front 像素座標（投影）
3. **顯示結果**：在 front view 圖像上填充選中的區域

### 1.5 關鍵實現細節

- **外參矩陣**：
  - BEV 相機：旋轉矩陣 `R_bev`（pitch = -90°），平移向量 `t_bev = [0, 2.5, 0]`
  - Front 相機：旋轉矩陣 `R_front = I`（單位矩陣），平移向量 `t_front = [0, 1, 0]`
- **座標轉換**：

  ```python
  # BEV 像素 → BEV 相機座標（假設地面 Z=0）
  P_bev_cam = [x, 0, y]

  # BEV 相機座標 → 世界座標
  P_world = R_bev.T @ (P_bev_cam - t_bev)

  # 世界座標 → Front 相機座標
  P_front_cam = R_front @ P_world + t_front

  # Front 相機座標 → Front 像素座標
  uvw = K @ P_front_cam
  u, v = uvw[0]/uvw[2], uvw[1]/uvw[2]
  ```

---

## Task 2: 3D Reconstruction

### 2.1 任務要求

1. **數據收集**：控制 agent 在場景中行走，同時保存 RGB、深度圖像和相機姿態（GT_pose.npy）
2. **點雲對齊和重建**：使用 ICP 算法對齊不同時刻的點雲
3. **相機姿態估計和可視化**：可視化軌跡並比較估計軌跡與真實軌跡的差異

### 2.2 重建流程（Pipeline）

#### Step 1: 深度圖反投影到點雲

- **函數**：`depth_image_to_point_cloud()`
- **過程**：
  1. 將深度圖從 0-255 範圍轉換為實際深度值（單位：米）
  2. 使用針孔相機模型將深度圖和 RGB 圖像轉換為點雲
  3. 內參數：FOV = 90°，解析度 = 512×512
  4. Depth scale = 100.0（但實際使用時 depth_scale = 1.0）
  5. Depth truncation = 10.0 米

#### Step 2: Voxelization（體素降採樣）

- **目的**：
  - 減少點數，降低記憶體使用
  - 加速後續處理（ICP、特徵計算）
- **參數**：`voxel_size = 0.1` 米
- **方法**：`pcd.voxel_down_sample(voxel_size)`

#### Step 3: 預處理（Preprocessing）

- **法向量估計**：
  - 半徑：`radius_normal = voxel_size * 2 = 0.2` 米
  - 最大鄰居數：`max_nn = 60`
  - 法向量一致化：`orient_normals_consistent_tangent_plane(k=30)`
- **FPFH 特徵計算**：
  - 半徑：`radius_feature = voxel_size * 6 = 0.6` 米
  - 最大鄰居數：`max_nn = 60`
  - 用途：用於 Global Registration（RANSAC）

#### Step 4: Global Registration（全域配準）

- **方法**：RANSAC based on Feature Matching
- **函數**：`execute_global_registration()`
- **參數**：
  - Distance threshold: `voxel_size * 1.5 = 0.15` 米
  - RANSAC n: 4（每次隨機取 4 個點）
  - Max iterations: 50000
  - Confidence: 400
- **作用**：提供 ICP 的初始對齊（initialization）
- **為什麼需要**：ICP 是局部優化算法，需要好的初始值才能收斂到正確解

#### Step 5: Local Registration（局部配準）

- **方法**：ICP（Iterative Closest Point）
- **兩個版本**：
  1. **Open3D 版本**：`local_icp_algorithm()`
     - 使用 Open3D 庫的 `registration_icp()`
     - 多尺度 Point-to-Plane ICP
     - Threshold: `voxel_size * 1.5 = 0.15` 米
     - 迭代次數：預設 30 次
  2. **自定義版本**：`my_local_icp_algorithm_accelerated()`
     - 自己實現的 Point-to-Plane ICP
     - Threshold: `voxel_size * 1.5 = 0.15` 米
     - 迭代次數：30 次
     - 使用 KDTree 加速最近鄰搜索
     - 使用緩存機制（KDTree 和法向量）

#### Step 6: 點雲合併與軌跡累積

- **變換矩陣累積**：
  ```python
  T_world[i] = T_world[i-1] @ pairwise_T[i-1]
  ```
  - `pairwise_T[i-1]` 是從 frame i-1 到 frame i 的變換矩陣
  - 使用矩陣乘法累積所有變換
- **點雲合併**：將每個 frame 的點雲變換到世界座標系後合併

#### Step 7: Umeyama Alignment（Sim(3) 對齊）

- **目的**：對齊估計軌跡和真實軌跡（考慮尺度、旋轉、平移）
- **方法**：Umeyama 算法
- **作用**：計算最佳的尺度係數、旋轉矩陣和平移向量

#### Step 8: 後處理

- **移除天花板**：使用 quantile 方法
  - Floor 1: 保留 y < quantile(0.55)
  - Floor 2: 保留 y < quantile(0.5)

#### Step 9: L2 距離計算

- **計算方法**：
  ```python
  distances = ||pred_cam_pos_aligned - gt_cam_pos||_2
  mean_l2_distance = mean(distances)
  ```
- **意義**：評估相機姿態估計的準確性

#### Step 10: 可視化

- **紅色線**：估計的相機軌跡
- **黑色線**：真實的相機軌跡（GT）
- **點雲**：重建的 3D 場景

---

## 核心知識點

### 3.1 內參矩陣 (Intrinsic Matrix) 的意義

- **定義**：描述相機內部參數的矩陣，將 3D 相機座標系中的點投影到 2D 圖像平面
- **形式**：
  ```
  K = [[fx,  0, cx],
       [ 0, fy, cy],
       [ 0,  0,  1]]
  ```
- **參數說明**：
  - `fx, fy`：焦距（以像素為單位）
  - `cx, cy`：主點（圖像中心，通常是 width/2, height/2）
- **作用**：
  - 將 3D 點從相機座標系投影到圖像座標系
  - 公式：`[u, v, 1]^T = K @ [X, Y, Z]^T`
- **在本作業中**：
  - FOV = 90°，解析度 = 512×512
  - `f = 0.5 * 512 / tan(45°) = 256`
  - `cx = cy = 256`

### 3.2 外參矩陣 (Extrinsic Matrix) 的意義

- **定義**：描述相機在世界座標系中的位置和朝向
- **組成**：
  - 旋轉矩陣 `R`（3×3）：描述相機朝向
  - 平移向量 `t`（3×1）：描述相機位置
- **完整形式**（4×4 齊次變換矩陣）：
  ```
  T = [[R, t],
       [0, 1]]
  ```
- **作用**：
  - 將世界座標系中的點轉換到相機座標系
  - 公式：`P_cam = R @ P_world + t`
- **在本作業中**：
  - Front 相機：`R = I`, `t = [0, 1, 0]`
  - BEV 相機：`R = R_y(-90°)`, `t = [0, 2.5, 0]`

### 3.3 ICP 算法的原理和限制

#### ICP 原理

1. **對應點搜索**：在目標點雲中找到源點雲中每個點的最近鄰
2. **變換估計**：根據對應點對計算最優變換矩陣（旋轉 + 平移）
3. **應用變換**：將變換應用到源點雲
4. **迭代**：重複步驟 1-3 直到收斂

#### ICP 的類型

- **Point-to-Point ICP**：最小化點到點的距離
- **Point-to-Plane ICP**：最小化點到平面的距離（更準確，收斂更快）

#### ICP 的限制

1. **需要好的初始值**：如果初始對齊太差，ICP 可能收斂到局部最優解
2. **對應點搜索的準確性**：如果最近鄰搜索錯誤，ICP 會失敗
3. **點雲密度**：點雲密度不均勻會影響結果
4. **局部最優**：只能找到局部最優解，不能保證全局最優

### 3.4 RANSAC 的作用和必要性

#### RANSAC 的作用

- **Global Registration**：在點雲之間找到粗略的對齊
- **提供 ICP 的初始值**：如果沒有 RANSAC，ICP 可能無法收斂到正確解
- **處理外點（Outliers）**：RANSAC 可以處理點雲中的噪聲和錯誤對應

#### 為什麼需要 RANSAC

- **ICP 是局部優化**：只能從初始值開始優化，如果初始值太差，會收斂到錯誤解
- **RANSAC 是全局搜索**：通過隨機採樣和驗證，找到一個合理的初始對齊
- **結合使用**：RANSAC（粗略對齊）→ ICP（精確對齊）

#### 如果不用 RANSAC 會怎樣？

- ICP 可能收斂到局部最優解
- 如果兩個點雲初始位置相差太大，ICP 無法找到正確對齊
- 重建結果會很差，軌跡誤差很大

### 3.5 改進 ICP 的技巧

#### 1. 多尺度 ICP

- **原理**：先用較大的 threshold 進行粗略對齊，再用較小的 threshold 進行精確對齊
- **實現**：
  - Coarse stage: threshold = 0.45 米
  - Fine stage: threshold = 0.3 米
- **優點**：可以處理較大的初始誤差

#### 2. Point-to-Plane ICP

- **原理**：最小化點到平面的距離，而不是點到點的距離
- **優點**：
  - 更準確
  - 收斂更快
  - 對噪聲更魯棒

#### 3. 法向量估計和一致化

- **法向量估計**：使用 PCA 或鄰居點計算法向量
- **法向量一致化**：確保法向量方向一致（朝向一致）
- **作用**：提高 Point-to-Plane ICP 的準確性

#### 4. Voxel Downsampling

- **目的**：減少點數，加速處理
- **參數選擇**：`voxel_size = 0.1` 米（平衡精度和速度）

#### 5. KDTree 加速

- **目的**：加速最近鄰搜索
- **實現**：使用 KDTree 數據結構，將 O(N²) 降低到 O(N log N)

#### 6. 緩存機制

- **KDTree 緩存**：對同一個 target 點雲，只構建一次 KDTree
- **法向量緩存**：對同一個 target 點雲，只計算一次法向量
- **優點**：大幅減少重複計算

#### 7. 提前結束（Early Stopping）

- **條件**：如果 fitness > 0.95，提前結束迭代
- **優點**：節省計算時間

#### 8. Fallback 機制

- **條件**：如果 Point-to-Plane ICP 的 fitness < 0.1，改用 Point-to-Point ICP
- **原因**：Point-to-Point ICP 在某些情況下更穩定

---

## 實現細節

### 4.1 相機參數設置

- **解析度**：512 × 512
- **FOV**：90 度（水平和垂直）
- **內參計算**：
  ```python
  f = 0.5 * width / tan(FOV/2) = 0.5 * 512 / tan(45°) = 256
  cx = width / 2 = 256
  cy = height / 2 = 256
  ```

### 4.2 Depth Scale 處理

- **Depth scale = 100.0**（在 spec 中提到）
- **實際使用**：`depth_scale = 1.0`（在 `create_from_color_and_depth` 中）
- **深度值轉換**：
  ```python
  depth_m = (depth.astype(np.float32) / 255.0) * 10.0
  ```
  - 將 0-255 的深度圖轉換為 0-10 米的實際深度值

### 4.3 座標系統轉換

- **BEV Projection**：
  1. BEV 像素 → BEV 相機座標
  2. BEV 相機座標 → 世界座標
  3. 世界座標 → Front 相機座標
  4. Front 相機座標 → Front 像素
- **3D Reconstruction**：
  1. 每個 frame 的點雲在各自的相機座標系中
  2. 通過變換矩陣轉換到世界座標系
  3. 合併所有點雲

### 4.4 矩陣乘法累積變換

- **原理**：如果從 frame 0 到 frame 1 的變換是 T₁，從 frame 1 到 frame 2 的變換是 T₂，那麼從 frame 0 到 frame 2 的變換是 T₁ @ T₂
- **實現**：
  ```python
  T_world[i] = T_world[i-1] @ pairwise_T[i-1]
  ```
- **注意**：矩陣乘法的順序很重要（右乘）

---

## 可能的問題與回答

### Q1: 什麼是內參矩陣？它的作用是什麼？

**A**: 內參矩陣描述相機的內部參數（焦距、主點），用於將 3D 相機座標系中的點投影到 2D 圖像平面。在本作業中，FOV=90°，解析度=512×512，焦距 f=256，主點在圖像中心 (256, 256)。

### Q2: 什麼是外參矩陣？它的作用是什麼？

**A**: 外參矩陣描述相機在世界座標系中的位置和朝向，包括旋轉矩陣 R 和平移向量 t。用於將世界座標系中的點轉換到相機座標系。在 BEV projection 中，BEV 相機的位置是 (0, 2.5, 0)，朝向是向下看（pitch = -90°）。

### Q3: 為什麼需要 RANSAC？如果不用 RANSAC 會怎樣？

**A**: RANSAC 用於 Global Registration，提供 ICP 的初始對齊。ICP 是局部優化算法，如果初始值太差，會收斂到局部最優解或錯誤解。如果不用 RANSAC，ICP 可能無法找到正確的對齊，導致重建結果很差。

### Q4: ICP 算法的原理是什麼？

**A**: ICP 通過迭代的方式對齊兩個點雲：

1. 在目標點雲中找到源點雲中每個點的最近鄰
2. 根據對應點對計算最優變換矩陣
3. 應用變換到源點雲
4. 重複直到收斂

### Q5: Point-to-Point 和 Point-to-Plane ICP 的區別是什麼？

**A**:

- **Point-to-Point**：最小化點到點的距離
- **Point-to-Plane**：最小化點到平面的距離，更準確、收斂更快、對噪聲更魯棒

### Q6: 為什麼需要 Voxel Downsampling？

**A**:

- 減少點數，降低記憶體使用
- 加速後續處理（ICP、特徵計算）
- 平衡精度和速度（voxel_size = 0.1 米）

### Q7: 多尺度 ICP 的原理是什麼？

**A**: 先用較大的 threshold 進行粗略對齊（Coarse stage），再用較小的 threshold 進行精確對齊（Fine stage）。這樣可以處理較大的初始誤差，提高對齊的準確性。

### Q8: 如何計算相機軌跡？

**A**: 通過累積每對相鄰 frame 之間的變換矩陣：

```python
T_world[i] = T_world[i-1] @ pairwise_T[i-1]
```

其中 `pairwise_T[i-1]` 是從 frame i-1 到 frame i 的變換矩陣。

### Q9: Umeyama Alignment 的作用是什麼？

**A**: Umeyama Alignment 用於對齊估計軌跡和真實軌跡，考慮尺度、旋轉、平移。它可以計算最佳的尺度係數、旋轉矩陣和平移向量，用於修正重建結果中的尺度誤差和座標系統差異。

### Q10: L2 距離的意義是什麼？

**A**: L2 距離用於評估相機姿態估計的準確性。計算估計的相機位置和真實相機位置之間的歐氏距離，取平均值。距離越小，表示估計越準確。

### Q11: 為什麼要移除天花板？

**A**: 天花板通常包含很多噪聲點，而且對重建結果沒有太大幫助。移除天花板可以讓重建結果更清晰，更容易觀察和比較。

### Q12: BEV projection 的座標轉換流程是什麼？

**A**:

1. BEV 像素座標 → BEV 相機座標（假設地面 Z=0）
2. BEV 相機座標 → 世界座標（使用 BEV 相機的外參）
3. 世界座標 → Front 相機座標（使用 Front 相機的外參）
4. Front 相機座標 → Front 像素座標（使用內參矩陣投影）

### Q13: 如何選擇 voxel_size？

**A**: voxel_size 需要平衡精度和速度：

- 太小：點數多，計算慢，但精度高
- 太大：點數少，計算快，但精度低
- 本作業使用 0.1 米，是一個合理的選擇

### Q14: FPFH 特徵的作用是什麼？

**A**: FPFH（Fast Point Feature Histograms）是點雲的局部特徵描述子，用於 Global Registration（RANSAC）。它可以找到點雲之間的對應關係，即使點雲之間有較大的位移。

### Q15: 為什麼需要法向量？

**A**:

- Point-to-Plane ICP 需要法向量來計算點到平面的距離
- 法向量可以幫助提高 ICP 的準確性和收斂速度
- 法向量需要一致化（orient_normals_consistent_tangent_plane）以確保方向一致

### Q16: 如何判斷 ICP 是否收斂？

**A**:

- Fitness：匹配點數 / 總點數，越高越好（通常 > 0.8 表示收斂良好）
- RMSE：均方根誤差，越小越好
- 迭代次數：達到最大迭代次數或誤差變化很小時停止

### Q18: 緩存機制的作用是什麼？

**A**:

- KDTree 緩存：對同一個 target 點雲，只構建一次 KDTree，避免重複計算
- 法向量緩存：對同一個 target 點雲，只計算一次法向量
- 可以大幅減少計算時間，特別是在處理多個 frame 時

### Q19: 如何處理深度圖？

**A**:

1. 將深度圖從 0-255 範圍轉換為實際深度值（0-10 米）
2. 使用 Open3D 的 `create_from_rgbd_image` 將 RGB 和深度圖轉換為點雲
3. 使用針孔相機模型進行反投影

### Q20: 數據收集時需要注意什麼？

**A**:

- 確保 agent 在場景中均勻行走，覆蓋整個場景
- 保持適當的移動速度，避免相鄰 frame 之間位移太大
- 確保深度圖質量良好，避免噪聲
- 保存足夠的 frame 數（通常需要 100+ frame）

---

## 總結

### 關鍵要點

1. **內參外參的理解**：內參描述相機內部，外參描述相機在世界中的位置
2. **RANSAC + ICP 的組合**：RANSAC 提供初始值，ICP 進行精確對齊
3. **ICP 的限制**：需要好的初始值，是局部優化
4. **改進技巧**：多尺度、Point-to-Plane、法向量、緩存等
5. **座標轉換**：理解不同座標系之間的轉換關係

### Demo 時要注意

1. 清楚說明每個步驟的作用
2. 解釋為什麼需要 RANSAC
3. 說明 Open3D 版本和自定義版本的區別
4. 展示重建結果和軌跡可視化
5. 說明 L2 距離的意義

---

**祝 Demo 順利！**
