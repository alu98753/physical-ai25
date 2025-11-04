# 正向運動學 (Forward Kinematics) 實作教學指南

## 📚 教學目標

本教程將帶你一步步理解並實作 UR5 機械臂的正向運動學 (Forward Kinematics) 和雅可比矩陣 (Jacobian Matrix) 計算。

---

## 🎯 整體思路

### 什麼是正向運動學？

**正向運動學 (FK)** 解決的問題是：

> 給定機器人的**關節角度** q = [q₁, q₂, q₃, q₄, q₅, q₆]，求末端執行器在空間中的**位置和姿態**

### 什麼是雅可比矩陣？

**雅可比矩陣 (Jacobian)** 描述的是：

> 關節速度 q̇ 如何影響末端執行器的線速度 v 和角速度 ω
>
> 關係式：[v; ω] = J · q̇

### 實作路線圖

```
輸入：6 個關節角度 q = [q₁, q₂, q₃, q₄, q₅, q₆]
  ↓
階段一：構建 D-H 變換矩陣
  ↓
階段二：計算正向運動學（累積變換）
  ↓
階段三：計算雅可比矩陣
  ↓
輸出：7D 姿態 (3D 位置 + 4D 四元數) + 6×6 雅可比矩陣
```

---

## 📐 階段一：理解並構建 D-H 變換矩陣

### 1.1 什麼是 D-H 參數？

**Denavit-Hartenberg (D-H) 參數**是一種標準化的方法，用 4 個參數描述相鄰兩個關節之間的空間關係：

| 參數          | 含義     | 說明                                     |
| ------------- | -------- | ---------------------------------------- |
| **θ (theta)** | 關節角度 | 繞 Z 軸旋轉，**這是我們的輸入變數 q[i]** |
| **d**         | 連桿偏移 | 沿 Z 軸平移                              |
| **a**         | 連桿長度 | 沿 X 軸平移                              |
| **α (alpha)** | 連桿扭轉 | 繞 X 軸旋轉                              |

### 1.2 UR5 的 D-H 參數

在 `fk.py` 中，`get_ur5_DH_params()` 提供了這些參數：

```python
# Joint 1: a=0,      d=0.0892,  alpha=π/2
# Joint 2: a=-0.425, d=0,       alpha=0
# Joint 3: a=-0.392, d=0,       alpha=0
# Joint 4: a=0,      d=0.1093,  alpha=π/2
# Joint 5: a=0,      d=0.09475, alpha=-π/2
# Joint 6: a=0,      d=0.2023,  alpha=0
```

⚠️ **重要提醒**：必須使用這些參數，不要從網上查找其他值！

### 1.3 經典 D-H 慣例的變換公式

從關節 i-1 到關節 i 的齊次變換矩陣 **A_i** 由以下 4 個基本變換按順序相乘：

```
A_i = Rot_z(θ) · Trans_z(d) · Trans_x(a) · Rot_x(α)
```

**關鍵點**：

- ✅ 順序不能錯：先 Z 軸旋轉 → Z 軸平移 → X 軸平移 → X 軸旋轉
- ✅ 這是**矩陣右乘**的順序

### 1.4 程式碼實作

讓我們逐步構建這 4 個基本變換矩陣：

#### 步驟 1：繞 Z 軸旋轉 θ

```python
ct = np.cos(theta)
st = np.sin(theta)
Rot_z = np.array([
    [ct, -st, 0, 0],
    [st,  ct, 0, 0],
    [ 0,   0, 1, 0],
    [ 0,   0, 0, 1]
])
```

**物理意義**：將座標系繞當前 Z 軸旋轉 θ 角度

#### 步驟 2：沿 Z 軸平移 d

```python
Trans_z = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, d],  # 注意 d 在這裡
    [0, 0, 0, 1]
])
```

**物理意義**：將座標系沿 Z 軸移動 d 距離

#### 步驟 3：沿 X 軸平移 a

```python
Trans_x = np.array([
    [1, 0, 0, a],  # 注意 a 在這裡
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])
```

**物理意義**：將座標系沿 X 軸移動 a 距離

#### 步驟 4：繞 X 軸旋轉 α

```python
ca = np.cos(alpha)
sa = np.sin(alpha)
Rot_x = np.array([
    [1,  0,   0,  0],
    [0, ca, -sa, 0],
    [0, sa,  ca, 0],
    [0,  0,   0, 1]
])
```

**物理意義**：將座標系繞當前 X 軸旋轉 α 角度

#### 步驟 5：組合成 A_i

```python
A_i = Rot_z @ Trans_z @ Trans_x @ Rot_x
```

💡 **理解要點**：

- 使用 `@` 運算符進行矩陣乘法（Python 3.5+）
- 這個順序產生的是從座標系 i-1 到座標系 i 的變換

---

## 🔄 階段二：計算正向運動學

### 2.1 核心思想

正向運動學的目標是計算**從世界座標系到末端執行器的總變換矩陣**。

這通過**累積**所有關節的變換來實現：

```
T_6^W = T_base^W · A_1 · A_2 · A_3 · A_4 · A_5 · A_6
```

其中：

- `T_base^W`: 基座在世界座標系中的位置（已給定）
- `A_i`: 第 i 個關節的 D-H 變換矩陣
- `T_6^W`: 末端執行器在世界座標系中的位姿

### 2.2 為什麼要保存中間結果？

我們不僅需要最終的 `T_6^W`，還需要所有中間的變換矩陣：

- `T_0^W` = 基座變換
- `T_1^W` = T_base^W · A_1
- `T_2^W` = T_base^W · A_1 · A_2
- ...
- `T_6^W` = T_base^W · A_1 · A_2 · A_3 · A_4 · A_5 · A_6

**原因**：階段三計算雅可比矩陣時需要這些中間變換！

### 2.3 程式碼實作

```python
# 初始化：A 是當前的累積變換矩陣
A = get_matrix_from_pose(base_pose)  # T_0 = 基座變換

# 創建列表保存所有中間變換
T_matrices = [A.copy()]  # T_0

# 迴圈處理 6 個關節
for i in range(6):
    # 獲取第 i 個關節的 D-H 參數
    a = DH_params[i]['a']
    d = DH_params[i]['d']
    alpha = DH_params[i]['alpha']
    theta = q[i]  # 關節角度來自輸入

    # 構建 A_i（使用階段一的方法）
    A_i = Rot_z @ Trans_z @ Trans_x @ Rot_x

    # 累積變換：A ← A · A_i
    A = A @ A_i

    # 保存中間結果
    T_matrices.append(A.copy())  # 重要：使用 .copy()
```

### 2.4 關鍵注意點

⚠️ **必須使用 `.copy()`**

```python
T_matrices.append(A.copy())  # ✅ 正確
T_matrices.append(A)         # ❌ 錯誤！只是引用，所有元素會指向同一個物件
```

💡 **迴圈後的狀態**：

- `A` 包含 `T_6^W`（末端執行器的總變換）
- `T_matrices` 包含 `[T_0, T_1, T_2, T_3, T_4, T_5, T_6]`（7 個矩陣）

---

## 🧮 階段三：計算雅可比矩陣

### 3.1 雅可比矩陣的結構

雅可比矩陣 J 是一個 **6×6** 矩陣，每一列對應一個關節：

```
J = [J_1, J_2, J_3, J_4, J_5, J_6]

每個 J_i 是一個 6×1 向量：
J_i = [J_v_i  ]  (3×1) - 線速度部分
      [J_ω_i  ]  (3×1) - 角速度部分
```

### 3.2 旋轉關節的雅可比公式

由於 UR5 的所有關節都是**旋轉關節**，使用幾何雅可比的公式：

**對於第 i 個關節**：

```
J_ω_i = z_{i-1}                    (角速度部分)
J_v_i = z_{i-1} × (p_E - p_{i-1})  (線速度部分，叉積)
```

其中：

- `z_{i-1}`: 第 i-1 個座標系的 Z 軸方向（在世界座標系中）
- `p_{i-1}`: 第 i-1 個座標系原點的位置（在世界座標系中）
- `p_E`: 末端執行器的位置（在世界座標系中）
- `×`: 向量叉積

### 3.3 如何從變換矩陣提取資訊？

一個 4×4 齊次變換矩陣的結構：

```
T = [R_{3×3}  |  p_{3×1}]
    [0  0  0  |    1    ]

其中：
- R: 旋轉矩陣（前 3×3）
- p: 位置向量（前 3 行的第 4 列）
```

**提取方法**：

```python
# 從變換矩陣 T 提取 Z 軸（旋轉矩陣的第 3 列）
z = T[0:3, 2]

# 從變換矩陣 T 提取位置向量
p = T[0:3, 3]
```

💡 **為什麼是第 2 列？**

- 旋轉矩陣的列分別是 X、Y、Z 軸
- 索引從 0 開始：第 0 列是 X，第 1 列是 Y，**第 2 列是 Z**

### 3.4 程式碼實作

```python
# 獲取末端執行器位置（使用未調整的 A）
p_E = A[0:3, 3]

# 初始化雅可比矩陣
jacobian = np.zeros((6, 6))

# 對每個關節計算雅可比列
for i in range(6):
    # 獲取第 i-1 個座標系的變換矩陣
    # 注意：i=0 對應 T_0（基座），i=1 對應 T_1，以此類推
    T_i_minus_1 = T_matrices[i]

    # 提取 Z 軸和位置
    z_i_minus_1 = T_i_minus_1[0:3, 2]  # Z 軸方向
    p_i_minus_1 = T_i_minus_1[0:3, 3]  # 位置

    # 計算線速度部分（叉積）
    J_v_i = cross(z_i_minus_1, p_E - p_i_minus_1)

    # 計算角速度部分
    J_omega_i = z_i_minus_1

    # 填入雅可比矩陣的第 i 列
    jacobian[0:3, i] = J_v_i      # 前 3 行：線速度
    jacobian[3:6, i] = J_omega_i  # 後 3 行：角速度
```

### 3.5 索引對應關係

理解這個索引關係很重要：

| 關節編號 | 迴圈索引 i | 使用的變換矩陣      | 物理意義    |
| -------- | ---------- | ------------------- | ----------- |
| Joint 1  | i=0        | T_matrices[0] = T_0 | 基座座標系  |
| Joint 2  | i=1        | T_matrices[1] = T_1 | 第 1 關節後 |
| Joint 3  | i=2        | T_matrices[2] = T_2 | 第 2 關節後 |
| Joint 4  | i=3        | T_matrices[3] = T_3 | 第 3 關節後 |
| Joint 5  | i=4        | T_matrices[4] = T_4 | 第 4 關節後 |
| Joint 6  | i=5        | T_matrices[5] = T_5 | 第 5 關節後 |

---

## ⚠️ 關鍵注意點與常見錯誤

### 注意點 1：末端位置的選擇

```python
# ✅ 正確：使用未經 adjustment 調整的 A
p_E = A[0:3, 3]

# 計算雅可比...

# adjustment 在函式末尾自動應用
A[:3, :3] = A[:3,:3] @ adjustment
```

**為什麼？**

- 雅可比計算基於標準 D-H 座標系
- `adjustment` 是為了對齊 URDF 定義的末端座標系
- 在計算雅可比時不應該使用調整後的座標系

### 注意點 2：矩陣乘法順序

```python
# ✅ 正確
A_i = Rot_z @ Trans_z @ Trans_x @ Rot_x

# ❌ 錯誤
A_i = Rot_x @ Trans_x @ Trans_z @ Rot_z
```

**記憶技巧**：按照 D-H 慣例的定義順序，從左到右

### 注意點 3：陣列切片

```python
# ✅ 正確：提取 Z 軸（第 3 列，索引 2）
z = T[0:3, 2]

# ❌ 錯誤：這會得到 X 軸
z = T[0:3, 0]

# ✅ 正確：提取位置（第 4 列，索引 3）
p = T[0:3, 3]
```

### 注意點 4：叉積函式

```python
# fk.py 中提供了 cross 函式
def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.cross(a, b)

# 使用時：
J_v_i = cross(z_i_minus_1, p_E - p_i_minus_1)
```

### 注意點 5：深拷貝 vs 引用

```python
# ✅ 正確：創建獨立副本
T_matrices.append(A.copy())

# ❌ 錯誤：只是添加引用
T_matrices.append(A)
# 結果：列表中所有元素都指向最後的 A 值！
```

---

## 🧪 測試與除錯技巧

### 除錯技巧 1：印出中間結果

```python
# 在迴圈中添加除錯輸出
for i in range(6):
    # ... 計算 A_i ...
    A = A @ A_i
    print(f"Joint {i+1}, Position: {A[0:3, 3]}")
    T_matrices.append(A.copy())
```

### 除錯技巧 2：檢查矩陣維度

```python
print(f"A shape: {A.shape}")           # 應該是 (4, 4)
print(f"Jacobian shape: {jacobian.shape}")  # 應該是 (6, 6)
print(f"T_matrices length: {len(T_matrices)}")  # 應該是 7
```

### 除錯技巧 3：驗證旋轉矩陣

旋轉矩陣應該是**正交矩陣**：R^T · R = I

```python
R = A[0:3, 0:3]
should_be_identity = R.T @ R
print(np.allclose(should_be_identity, np.eye(3)))  # 應該是 True
```

---

## 📝 完整程式碼框架

將所有內容整合在一起：

```python
def your_fk(DH_params, q, base_pos):
    # 初始化
    base_pose = list(base_pos) + [0, 0, 0]
    A = get_matrix_from_pose(base_pose)
    jacobian = np.zeros((6, 6))

    # ===== 階段一 & 階段二：正向運動學 =====
    T_matrices = [A.copy()]

    for i in range(6):
        # 獲取 D-H 參數
        a, d, alpha, theta = (DH_params[i]['a'],
                              DH_params[i]['d'],
                              DH_params[i]['alpha'],
                              q[i])

        # 構建 4 個基本變換（階段一）
        # ... (Rot_z, Trans_z, Trans_x, Rot_x) ...

        # 組合並累積
        A_i = Rot_z @ Trans_z @ Trans_x @ Rot_x
        A = A @ A_i
        T_matrices.append(A.copy())

    # ===== 階段三：雅可比矩陣 =====
    p_E = A[0:3, 3]

    for i in range(6):
        T_i_minus_1 = T_matrices[i]
        z_i_minus_1 = T_i_minus_1[0:3, 2]
        p_i_minus_1 = T_i_minus_1[0:3, 3]

        J_v_i = cross(z_i_minus_1, p_E - p_i_minus_1)
        J_omega_i = z_i_minus_1

        jacobian[0:3, i] = J_v_i
        jacobian[3:6, i] = J_omega_i

    # 應用調整並返回
    adjustment = np.asarray([[ 0, -1,  0],
                             [ 0,  0,  0],
                             [ 0,  0, -1]])
    A[:3, :3] = A[:3,:3] @ adjustment
    pose_7d = np.asarray(get_pose_from_matrix(A, 7))

    return pose_7d, jacobian
```

---

## 🎓 理解性問題自測

1. **為什麼 D-H 變換的順序是 Rot_z → Trans_z → Trans_x → Rot_x？**

   - 答：這是經典 D-H 慣例的定義，確保不同機器人使用統一的參數化方法

2. **為什麼需要保存中間變換矩陣？**

   - 答：計算雅可比矩陣時需要每個關節處的座標系資訊

3. **T_matrices[0] 對應什麼？**

   - 答：基座座標系的變換 T_0

4. **為什麼雅可比的角速度部分只是 z 軸？**

   - 答：旋轉關節的轉軸就是其 Z 軸，角速度方向沿轉軸

5. **如果不使用 .copy() 會發生什麼？**

   - 答：列表中所有元素都會指向同一個矩陣物件，最終都變成最後的值

---

## 🚀 下一步

完成 FK 實作後：

1. 執行 `python fk.py` 測試
2. 確保通過所有測試案例（easy, medium, hard）
3. 準備進入 Task 2：逆向運動學 (IK)

---

## 📚 參考資源

- **D-H 參數詳解**：經典 D-H vs Modified D-H
- **雅可比矩陣**：幾何雅可比 vs 分析雅可比
- **齊次變換矩陣**：旋轉 + 平移的統一表示

祝學習順利！🎉
