

### 總體流程與環境設定

`README.md` 首先說明了環境安裝。您需要使用 `conda` 建立一個 `python=3.7` 的環境，並安裝 `requirements.txt` 中的套件。Task 3 會用到 `ravens` 框架，因此 `README.md` 也提供了 TensorFlow 相關的 CUDA/cuDNN 安裝指令。

您的實作將集中在 `fk.py` 和 `ik.py` 兩個檔案的 `TODO` 區域。

---

### 🎯 任務 1: 實作正向運動學 (`fk.py`)

[cite_start]**您的目標：** 在 `your_fk` 函數中，根據給定的6個關節角度 `q`，計算出機器人末端執行器 (end-effector) 的 **7D 姿態 (pose)** 和 **6x6 雅可比矩陣 (Jacobian)** [cite: 42, 44]。

#### 實作過程與注意事項：

1.  **理解 D-H 參數：**
    * `fk.py` 中的 `get_ur5_DH_params()` 函數提供了您**必須**使用的 D-H 參數。
    * [cite_start]作業規格明確指出，這些參數遵循**經典 D-H 慣例 (classic convention)** [cite: 39][cite_start]，並且可能與官方規格不同 [cite: 51]，所以請勿自行查找並替換。
    * D-H 參數表中的每一行 `(a, d, alpha)` 對應一個關節。第 $i$ 個關節的 $\theta_i$ 來自輸入的 `q[i]`。

2.  **計算總變換矩陣 $A$ (Forward Kinematics)：**
    * 您的目標是計算從基座 (base) 到末端執行器 (joint 6) 的總 4x4 齊次變換矩陣 (homogeneous transformation matrix)，我們稱之為 $T_6^0$ 或 $A$。
    * 您需要從基座開始。程式已經幫您初始化了 `A = get_matrix_from_pose(base_pose)`，這是基座在世界座標系下的變換 $T_{base}^W$。
    * 您需要寫一個迴圈，依序處理 6 個關節 (i=1 到 6)。
    * 在迴圈中，為每個關節 $i$ 建立其 D-H 變換矩陣 $A_i$ (也就是 $T_i^{i-1}$)。根據經典 D-H 慣例，這個矩陣是四個基本變換的乘積：
        $A_i = Rot_{z}(\theta_i) \cdot Trans_{z}(d_i) \cdot Trans_{x}(a_i) \cdot Rot_{x}(\alpha_i)$
    * **注意：** 矩陣乘法的順序至關重要。
    * 在迴圈中，不斷將這個 $A_i$ 乘上當前的總變換矩陣：`A = A @ A_i`。
    * 迴圈結束後，`A` 就會是 $T_6^W$ (從世界座標系到第6關節座標系的變換)。
    * **注意：** 在 `your_fk` 函數的末尾，有一個 `adjustment` 矩陣會被乘上。這可能是為了統一 URDF 的末端座標系和 D-H 定義的末端座標系。您的計算**不需要**考慮這個，您只需計算標準的 $T_6^W$ 並將其賦值給 `A` 即可。

3.  **計算雅可比矩陣 `jacobian` (Geometric Jacobian)：**
    * 雅可比矩陣 $J$ 是一個 $6 \times 6$ 矩陣，它關聯了關節速度 $\dot{q}$ 與末端執行器的線速度 $v$ 和角速度 $\omega$。
    * $J = [J_1, J_2, J_3, J_4, J_5, J_6]$，其中 $J_i$ 是第 $i$ 個關節的 $6 \times 1$ 向量。
    * [cite_start]由於 UR5 都是旋轉關節 (revolute joints) [cite: 10]，第 $i$ 個關節的 $J_i$ 計算方式為：
        $J_i = \begin{bmatrix} J_{v_i} \\ J_{\omega_i} \end{bmatrix}$
        * $J_{\omega_i} = z_{i-1}$ (第 $i-1$ 個座標系的 z 軸向量)
        * $J_{v_i} = z_{i-1} \times (p_E - p_{i-1})$ (叉積)
    * **如何計算 $z_{i-1}$ 和 $p_{i-1}$？**
        * 在您計算 FK 的迴圈中，您不僅需要最終的 $A$ (即 $T_6^W$)，還需要**所有**中間的變換矩陣 ($T_0^W, T_1^W, \dots, T_5^W$)。
        * 建議在迴圈中儲存每一個 $T_i^W$ (從 `A` @ $A_1$ @ ... @ $A_i$ 得到的矩陣)。
        * $p_E$ 是末端執行器的位置，即 $T_6^W$ 的位置向量 ($A[0:3, 3]$)。
        * $p_{i-1}$ 是第 $i-1$ 個座標系原點的位置，即 $T_{i-1}^W$ 的位置向量。
        * $z_{i-1}$ 是第 $i-1$ 個座標系的 Z 軸方向，即 $T_{i-1}^W$ 的第三列向量 ($T_{i-1}^W[0:3, 2]$)。
    * 您需要再一個迴圈 (i=0 到 5)，計算出 6 個 $J_i$ 向量，並把它們填入 `jacobian` 矩陣的對應列中。
    * **注意：** $p_E$ 應該使用您計算出的、**未經** `adjustment` 調整的 $A$ 矩陣的位置。

**測試：** 完成後，直接執行 `python fk.py` 來測試您的實作。

---

### 🎯 任務 2: 實作逆向運動學 (`ik.py`)

[cite_start]**您的目標：** 在 `your_ik` 函數中，實現一個**迭代逆向運動學 (IIK) 求解器** [cite: 63]。給定一個目標 7D 姿態 `new_pose`，您需要反覆運算，找出能達到該姿態的 6D 關節角度 `tmp_q`。

[cite_start]**方法：** 規格書要求使用**雅可比偽逆法 (pseudo-inverse method)** [cite: 64]。

#### 實作過程與注意事項：

1.  **取得 D-H 參數：** 在 `your_ik` 函數中，您首先需要呼叫 `get_ur5_DH_params()`，因為您需要用 `your_fk` 來輔助計算。

2.  **迭代迴圈：**
    * 程式已經幫您取得了當前的關節角度 `tmp_q`。這就是您迭代的起點，稱之為 $q_{current}$。
    * [cite_start]您需要一個 `for` 迴圈，最多迭代 `max_iters` 次 [cite: 57]。

3.  **迴圈內的核心步驟：**
    * **A. 計算當前狀態：**
        * 呼叫您在 Task 1 中實作的函數：
          `current_pose_7d, J = your_fk(dh_params, q_current, base_pos)`
        * 這會給您基於 $q_{current}$ 的**當前姿態**和**雅可比矩陣 $J$**。
    * **B. 計算誤差 $\Delta x$：**
        * 這是最關鍵的步驟之一。您需要計算**目標姿態 (new\_pose)** 和**當前姿態 (current\_pose\_7d)** 之間的 6D 誤差 $\Delta x = \begin{bmatrix} \Delta p \\ \Delta o \end{bmatrix}$。
        * **位置誤差 $\Delta p$ (3x1)：** 這很簡單，就是向量相減：
          $\Delta p = p_{target} - p_{current}$
          (其中 $p_{target} = \text{new\_pose}[:3]$，$p_{current} = \text{current\_pose\_7d}[:3]$)
        * **姿態 (旋轉) 誤差 $\Delta o$ (3x1)：** 這比較複雜。您不能直接減去四元數。
            1.  將目標四元數 $Q_{target}$ (來自 `new_pose[3:]`) 和當前四元數 $Q_{current}$ (來自 `current_pose_7d[3:]`) 轉換為 3x3 旋轉矩陣 $R_{target}$ 和 $R_{current}$。 (您可以使用 `scipy.spatial.transform.Rotation`)
            2.  計算誤差旋轉矩陣 $R_{error} = R_{target} \cdot R_{current}^T$ (即 $R_{target} @ R_{current}.T$)。
            3.  將 $R_{error}$ 轉換回軸角 (axis-angle) 表示法，這就是您的 3D 姿態誤差 $\Delta o$。 (可使用 `.as_rotvec()` 方法)
    * **C. 檢查是否收斂：**
        * 計算 $\Delta x$ 的範數 (norm)，例如 `np.linalg.norm(delta_x)`。
        * [cite_start]如果這個誤差範數小於 `stop_thresh` [cite: 57]，代表已經足夠接近目標，您可以 `break` 迴圈。
    * **D. 計算關節增量 $\Delta q$：**
        * 這是 IIK 的核心公式：$\Delta x = J \cdot \Delta q$。
        * 我們要求解 $\Delta q$：$\Delta q = J^{-1} \cdot \Delta x$。
        * 由於 $J$ 可能是奇異 (singular) 的，我們使用**偽逆 (pseudo-inverse)** $J^+$：
          $\Delta q = J^+ \cdot \Delta x$
        * 您可以使用 `scipy.linalg.pinv(J)` 或 `np.linalg.pinv(J)` 來計算 $J^+$。
        * 所以 `delta_q = pinv(J) @ delta_x`。
    * **E. 更新關節角度：**
        * $q_{current} = q_{current} + \Delta q$
        * **強烈建議：** 引入一個小的**步長 (step rate)** $\alpha$ (例如 0.1 或 0.05)，使迭代更穩定：
          $q_{current} = q_{current} + \alpha \cdot \Delta q$
        * **注意：** `ik.py` 中提供了一個 `joint_limits` 變數。在更新 $q_{current}$ 後，您應該使用 `np.clip` 函數將 $q_{current}$ 的值限制在這些範圍內，以防止求解器跑到無效的關節角度。

4.  **返回結果：** 迴圈結束後 (無論是收斂還是達到 `max_iters`)，返回最終的 `q_current` (記得轉成 `list` 格式)。

**測試：** 完成後，執行 `python ik.py` 來測試您的實作。

---

### 🎯 任務 3: 整合至 Transporter Network

[cite_start]這個任務是驗證您的 IK 求解器在實際應用中的表現 [cite: 74]。您**不需要**編寫新程式，而是需要設定環境並執行測試腳本。

#### 執行過程與注意事項：

1.  **下載資料：** 根據 `README.md` 的指引，從 Google Drive 下載 `block-insertion-easy-test/` (資料集) 和 `checkpoints/` (模型權重)。
2.  **放置檔案：** **嚴格**按照 `README.md` 的指示，將這兩個資料夾分別放在 `hw3/ravens/` 目錄下。路徑錯誤會導致測試失敗。
3.  **檢查 `TODO`：**
    * [cite_start]作業規格 (spec) 中有一個**非常重要**的提示 [cite: 34]：`Please check all the "TODO" comments in fk.py, ik.py, ravens/ravens/environments/environments.py`
    * 您**必須**打開 `ravens/ravens/environments/environments.py` 這個檔案，找到 `TODO` 註解。
    * 這個 `TODO` 很可能是用來**切換** PyBullet 內建 IK 和您實作的 `your_ik` 的地方。您需要按照 `TODO` 的指示，註解掉原有的 `pybullet_ik` 並取消註解或加入呼叫 `your_ik` 的程式碼。
4.  **執行測試：**
    * `cd` 進入 `ravens` 目錄。
    * 執行 `README.md` 中提供的 `test.py` 指令。
    * `CUDA_VISIBLE_DEVICES=-1` 是指**不使用 GPU**，這很正常，因為 IK 計算主要在 CPU 上。
5.  **驗證結果：**
    * 您需要觀察 10 次測試 (Test: 1/10 ... 10/10)。
    * [cite_start]如果您的 `your_ik` (以及其依賴的 `your_fk`) 實作正確且穩定，您應該會看到 10 次試驗都成功 (Total Reward: 1.0 Done: True) [cite: 85]。

---

### 📝 最後的報告

[cite_start]別忘了您還需要撰寫一份報告，回答規格書 (spec) 中關於 Task 1, 2, 3 的問題 [cite: 97]，包括：
* [cite_start]**Task 1：** 解釋 FK 實作、D-H vs Craig's 慣例、並填寫完整的 D-H 表 [cite: 99, 100, 101]。
* [cite_start]**Task 2：** 解釋 IK 實作、遇到的問題，以及 (Bonus) 是否嘗試了其他 IK 方法 [cite: 162, 164, 165]。
* [cite_start]**Task 3：** 比較 `your_ik` 和 `pybullet_ik` 的結果 [cite: 169]。