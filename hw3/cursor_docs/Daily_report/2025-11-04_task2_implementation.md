# Task 2 開發日誌 - 2025-11-04

## 任務目標

實作 UR5 機械臂的逆向運動學 (IK) 迭代求解器

## 實施計劃

### 階段一：準備工作與迭代初始化

- [進行中] 獲取 D-H 參數
- [進行中] 設定當前關節角度初始值
- [進行中] 建立迭代迴圈結構

### 階段二：核心迭代計算

- [ ] 呼叫 FK 獲得當前狀態
- [ ] 計算位置誤差 Δp
- [ ] 計算姿態誤差 Δo（四元數處理）
- [ ] 組合 6D 誤差向量
- [ ] 雅可比偽逆法求解 Δq
- [ ] 更新關節角度

### 階段三：收斂與限制處理

- [ ] 實作收斂條件檢查
- [ ] 應用關節限制
- [ ] 返回結果

## 開始時間

2025-11-04 開始實作

## 實作細節

### 核心演算法

使用迭代雅可比偽逆法：

- 從當前關節角度開始
- 每次迭代計算誤差並更新
- 直到收斂或達到最大迭代次數

### 關鍵公式

- 誤差向量：Δx = [Δp; Δo]
- 關節增量：Δq = J⁺ · Δx
- 更新規則：q_new = q_current + α · Δq

## 進度更新

- [✅ 完成] 創建開發日誌
- [✅ 完成] 實作階段一：迭代初始化
- [✅ 完成] 實作階段二：核心迭代計算
- [✅ 完成] 實作階段三：收斂與限制處理
- [✅ 完成] 執行測試並驗證

## 測試結果

### 完美通過！總分：40.000 / 40.000 🎉

#### 測試詳情：

1. **ik_test_case_easy.json** (300 個測試案例)

   - Mean Error: 0.001733
   - Error Count: 0 / 300 ✅
   - Score: 13.333 / 13.333 ✅

2. **ik_test_case_medium.json** (100 個測試案例)

   - Mean Error: 0.001371
   - Error Count: 0 / 100 ✅
   - Score: 13.333 / 13.333 ✅

3. **ik_test_case_hard.json** (100 個測試案例)
   - Mean Error: 0.001133
   - Error Count: 0 / 100 ✅
   - Score: 13.333 / 13.333 ✅

## 實作總結

### 成功要點：

1. **正確的迭代結構**

   - 使用 `your_fk` 獲得當前姿態和雅可比矩陣
   - 每次迭代都更新當前關節角度

2. **準確的誤差計算**

   - 位置誤差：簡單的向量相減
   - 姿態誤差：四元數 → 旋轉矩陣 → 誤差旋轉 → 旋轉向量
   - 正確處理 scipy 的四元數格式 [x, y, z, w]

3. **雅可比偽逆法**

   - 使用 `pinv(J)` 計算偽逆
   - 求解關節增量：Δq = J⁺ · Δx
   - 步長控制：q_new = q_current + 0.1 · Δq

4. **收斂與限制處理**
   - 檢查誤差範數是否小於閾值
   - 使用 `np.clip` 確保關節角度在限制範圍內

### 超參數設定：

- **step_rate = 0.1**：平衡收斂速度和穩定性
- **max_iters = 1000**：使用預設值
- **stop_thresh = 0.001**：使用預設值

所有測試案例的平均誤差都遠小於閾值 0.02，表示實作非常準確！

## 環境配置問題與解決方案

### 問題描述

在執行 Task 2 測試時，PyBullet GUI 模式無法啟動，出現以下錯誤：

```
Failed to create an OpenGL context
libGL error: MESA-LOADER: failed to open radeonsi
libGL error: failed to load driver: radeonsi
libGL error: MESA-LOADER: failed to open swrast
libGL error: failed to load driver: swrast
```

**根本原因：** Conda 環境中的 `libstdc++.so.6` 版本過舊（僅支援到 GLIBCXX_3.4.26），而系統 OpenGL 驅動需要 GLIBCXX_3.4.30。

### 診斷過程

1. **檢查硬體環境**

   - GPU：NVIDIA RTX 3060 + AMD 集顯 ✅
   - 顯示器：正常連接 ✅

2. **檢查 OpenGL 配置**

   ```bash
   glxinfo | head -20
   ```

   - 結果：`direct rendering: Yes` ✅
   - 系統 OpenGL 正常運作

3. **檢查驅動文件**

   ```bash
   ls -la /usr/lib/x86_64-linux-gnu/dri/
   ```

   - `radeonsi_dri.so` 存在 ✅
   - `swrast_dri.so` 存在 ✅

4. **發現關鍵問題**
   ```bash
   strings ~/miniconda3/envs/pdm-hw3/lib/libstdc++.so.6 | grep GLIBCXX
   strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX
   ```
   - Conda 環境：最高支援 GLIBCXX_3.4.26 ❌
   - 系統環境：支援 GLIBCXX_3.4.30 ✅
   - **結論：版本衝突導致驅動無法載入**

### 解決方案（選項 B）：使用環境變數

執行 Task 2 時使用以下命令：

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri \
python ik.py
```

**環境變數說明：**

- `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`

  - 強制程式優先載入系統的 libstdc++.so.6
  - 繞過 Conda 環境中的舊版本庫
  - 提供所需的 GLIBCXX_3.4.30 符號

- `LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri`
  - 明確指定 OpenGL 驅動檔案位置
  - 確保 PyBullet 能找到正確的 DRI 驅動

### 驗證結果

使用環境變數後，成功啟動 GUI：

```
Creating context
Created GL 3.3 context
Direct GLX rendering context obtained
Making context current
GL_VENDOR=AMD
GL_RENDERER=RAPHAEL_MENDOCINO (raphael_mendocino, LLVM 15.0.7, DRM 3.57, 6.8.0-85-generic)
GL_VERSION=4.6 (Core Profile) Mesa 23.2.1-1ubuntu3.1~22.04.3
```

**結果確認：**

- ✅ OpenGL 成功初始化（GL 3.3 context）
- ✅ 使用 AMD 渲染器（RAPHAEL_MENDOCINO）
- ✅ GUI 視窗正常顯示
- ✅ 所有測試正常執行

### 替代方案說明

**方案 A：DIRECT 模式（無 GUI）**

如果不需要視覺化，可以修改 `ik.py` 第 231 行：

```python
# 原始（需要 GUI）
physics_client_id = p.connect(p.GUI)

# 修改為（無 GUI）
physics_client_id = p.connect(p.DIRECT)
```

優點：

- 不需要處理 OpenGL 問題
- 執行速度更快
- 完全符合測試要求

缺點：

- 無法視覺化機械臂運動過程

**最終選擇：方案 B（使用環境變數）**

- 保留 GUI 視覺化功能
- 便於除錯和驗證
- 適用於需要觀察機械臂運動的場景

## 完成時間

2025-11-04 - Task 2 完全實作並通過所有測試

## 下一步

準備開始 Task 3: 整合至 Transporter Network

### Task 3 執行注意事項

Task 3 同樣需要使用環境變數來啟用 GUI，執行命令如下：

```bash
cd ravens
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri \
CUDA_VISIBLE_DEVICES=-1 \
python ravens/test.py \
  --assets_root=./ravens/environments/assets/ \
  --disp=True \
  --task=block-insertion-easy \
  --agent=transporter \
  --n_demos=1000 \
  --n_steps=20000
```

**前置要求：**

1. 下載測試資料集：

   - 連結：https://drive.google.com/file/d/1Jh8hAvraT1Zt1YfSNRT_lMJXbsK4Wcse/view?usp=sharing
   - 放置 `block-insertion-easy-test/` 資料夾到 `hw3/ravens/`

2. 下載模型檢查點：
   - 連結：https://drive.google.com/file/d/1cmFbqTzuu6IUJPlx1eOq2djRSubfM94H/view?usp=sharing
   - 放置 `checkpoints/` 資料夾到 `hw3/ravens/`

**備註：**

- 如果不需要視覺化，可將 `--disp=True` 改為 `--disp=False`
- 使用 `CUDA_VISIBLE_DEVICES=-1` 強制使用 CPU（避免 CUDA 相容性問題）
- Task 3 使用與 Task 2 相同的 `your_ik` 函數（已在 `ravens/ravens/environments/environment.py` 中整合）
