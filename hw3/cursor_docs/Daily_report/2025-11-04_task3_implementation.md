# Task 3 開發日誌 - 2025-11-04

## 任務目標

在 Transporter Networks 框架中測試 IK 實作，完成 Block Insertion Task 的 10 個測試案例

## 前置要求

### 資料集下載

1. **測試資料集**

   - 連結：https://drive.google.com/file/d/1Jh8hAvraT1Zt1YfSNRT_lMJXbsK4Wcse/view?usp=sharing
   - 位置：`hw3/ravens/block-insertion-easy-test/`

2. **模型檢查點**
   - 連結：https://drive.google.com/file/d/1cmFbqTzuu6IUJPlx1eOq2djRSubfM94H/view?usp=sharing
   - 位置：`hw3/ravens/checkpoints/`

## 問題診斷與解決過程

### 問題一：OpenGL Context 創建失敗

#### 錯誤訊息

```
libGL error: MESA-LOADER: failed to open radeonsi
libGL error: version `GLIBCXX_3.4.30' not found
Failed to create an OpenGL context
```

#### 根本原因

與 Task 2 相同的問題：Conda 環境中的 `libstdc++.so.6` 版本過舊，無法滿足 OpenGL 驅動需求。

#### 解決方案

使用環境變數強制載入系統的共享庫：

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri
```

**驗證結果：**

- ✅ OpenGL 成功初始化（GL 3.3 context）
- ✅ 使用 AMD 渲染器
- ✅ GUI 視窗正常顯示

---

### 問題二：模型檢查點找不到

#### 錯誤訊息

```
OSError: Unable to open file (unable to open file: name = './checkpoints/block-insertion-easy-transporter-1000-0/attention-ckpt-20000.h5', errno = 2, error message = 'No such file or directory')
```

#### 診斷過程

1. 檢查執行命令，發現缺少 `--n_demos=1000` 參數
2. 程式使用此參數構建檢查點路徑：`{task}-{agent}-{n_demos}-{run_id}`

#### 解決方案

添加缺失的命令參數：

```bash
--n_demos=1000
```

完整路徑變成：`./checkpoints/block-insertion-easy-transporter-1000-0/`

**驗證結果：**

- ✅ 模型檢查點成功載入
- ✅ "Loading pre-trained model at 20000 iterations" 顯示成功

---

### 問題三：測試案例未執行（分數 0/10）

#### 現象

- 程式執行完成但分數為 0.000 / 10.000
- 沒有看到 "Test: 1/10" 等測試輸出
- 模型已成功載入但沒有執行測試

#### 可能原因

測試資料集路徑問題或資料集結構不正確。

#### 解決方案

確認資料集正確放置在 `ravens/block-insertion-easy-test/` 目錄下，使用完整命令執行。

---

## 最終成功方案

### 完整執行命令

```bash
cd /home/clu98753cs13/Desktop/course/phyai/physical-ai25/hw3/ravens

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

### 參數說明

| 參數                      | 說明                                            |
| ------------------------- | ----------------------------------------------- |
| `LD_PRELOAD`              | 強制使用系統 libstdc++（解決 GLIBCXX 版本問題） |
| `LIBGL_DRIVERS_PATH`      | 指定 OpenGL 驅動位置                            |
| `CUDA_VISIBLE_DEVICES=-1` | 強制使用 CPU（避免 CUDA 相容性問題）            |
| `--assets_root`           | 環境資源路徑                                    |
| `--disp=True`             | 啟用 GUI 視覺化                                 |
| `--task`                  | 測試任務類型                                    |
| `--agent`                 | 使用的代理模型                                  |
| `--n_demos=1000`          | 訓練示範數量（用於構建檢查點路徑）              |
| `--n_steps=20000`         | 載入的訓練步數                                  |

---

## 測試結果

### 完美通過！總分：10.000 / 10.000 🎉

#### 詳細結果：

```
============================ Task 3 : Transporter Network ============================

Test: 1/10
Total Reward: 1.0 Done: True

Test: 2/10
Total Reward: 1.0 Done: True

Test: 3/10
Total Reward: 1.0 Done: True

Test: 4/10
Total Reward: 1.0 Done: True

Test: 5/10
Total Reward: 1.0 Done: True

Test: 6/10
Total Reward: 1.0 Done: True

Test: 7/10
Total Reward: 1.0 Done: True

Test: 8/10
Total Reward: 1.0 Done: True

Test: 9/10
Total Reward: 1.0 Done: True

Test: 10/10
Total Reward: 1.0 Done: True

====================================================================================
- Your Total Score : 10.000 / 10.000
====================================================================================
```

**所有 10 個測試案例全部成功完成，每個案例獲得滿分獎勵 1.0！**

---

## IK 整合驗證

### 整合點

Task 3 使用的 IK 函數位於 `ravens/ravens/environments/environment.py` 的 `solve_ik` 方法（第 401-414 行）：

```python
def solve_ik(self, pose):
    """Calculate joint configuration with inverse kinematics."""

    new_pose = pose

    your_joints = your_ik(self.ur5, new_pose, max_iters=100, base_pos=[0, 0, 0])

    your_joints = np.float32(your_joints)
    your_joints[2:] = (your_joints[2:] + np.pi) % (2 * np.pi) - np.pi

    return your_joints
```

### 驗證結果

- ✅ `your_ik` 函數成功整合到 Transporter Network 框架
- ✅ 機械臂成功完成所有抓取和插入動作
- ✅ IK 求解準確，沒有出現運動學錯誤
- ✅ 所有任務都在規定步數內完成

---

## 技術總結

### 關鍵成功要素

1. **環境配置正確**

   - 解決 Conda 環境與系統庫的版本衝突
   - 正確設定 OpenGL 驅動路徑
   - 使用 CPU 避免 CUDA 相容性問題

2. **完整的命令參數**

   - 所有必要參數都正確提供
   - 檢查點路徑構建正確
   - 資料集路徑正確

3. **IK 實作準確**
   - Task 2 實作的 IK 演算法準確可靠
   - 迭代收斂性良好
   - 能處理各種姿態要求

### 注意事項

**TensorFlow 棄用警告**

執行時會看到以下警告（這是正常的）：

```
WARNING:tensorflow: ... set_learning_phase ... is deprecated
```

- 這是因為 Ravens 使用較舊的 TensorFlow API
- **不影響程式功能和分數**
- 可以安全忽略

---

## 完成時間

2025-11-04 - Task 3 完全實作並通過所有測試

---

## HW3 總結

### 所有任務完成狀態

| Task     | 內容                | 分數                | 狀態        |
| -------- | ------------------- | ------------------- | ----------- |
| Task 1   | Forward Kinematics  | 20.000 / 20.000     | ✅ 完成     |
| Task 2   | Inverse Kinematics  | 40.000 / 40.000     | ✅ 完成     |
| Task 3   | Transporter Network | 10.000 / 10.000     | ✅ 完成     |
| **總計** |                     | **70.000 / 70.000** | **✅ 滿分** |

### 核心技術實作

1. **D-H 參數建模** - 正確建立 UR5 機械臂的運動學模型
2. **正向運動學** - 從關節角度計算末端效應器位姿
3. **雅可比矩陣** - 計算速度關係和奇異點分析
4. **逆向運動學** - 迭代雅可比偽逆法求解目標位姿
5. **深度學習整合** - 將 IK 整合到 Transporter Networks 框架

### 關鍵學習

- **運動學理論實踐**：從理論公式到實際程式碼實作
- **數值方法應用**：迭代優化、偽逆矩陣、收斂控制
- **環境配置除錯**：解決系統相依性和版本衝突問題
- **框架整合能力**：將自己的演算法整合到現有深度學習框架

---

## 附錄：快速執行指令

### Task 2 執行

```bash
cd /home/clu98753cs13/Desktop/course/phyai/physical-ai25/hw3
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri \
python ik.py
```

### Task 3 執行

```bash
cd /home/clu98753cs13/Desktop/course/phyai/physical-ai25/hw3/ravens
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
