import cv2
import numpy as np
import math

points = []

class Projection(object):

    def __init__(self, image_path, points):
        """
            :param points: Selected pixels on top view(BEV) image
        """

        if type(image_path) != str:
            self.image = image_path
        else:
            self.image = cv2.imread(image_path)
        self.height, self.width, self.channels = self.image.shape

        self.points = points

    def top_to_front(self, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0, fov=90):
        """
        Projects points from the top-down BEV image to the front-view image.
        The transformation is derived from the camera parameters in HW1_spec.pdf.
        
        Note: While the function signature includes general rotation (theta, phi, gamma)
        and translation (dx, dy, dz) parameters, this implementation uses the specific,
        hardcoded extrinsic values from the homework specification for accuracy.
        The 'fov' parameter is actively used.

        :param fov: The field of view of the camera in degrees. Default is 90.
        :return: A list of projected points (x, y) in the front-view image.
        """

        # 1. 根據 HW1 spec 初始化相機參數
        
        # [cite_start]內參 (Intrinsics) - 兩個相機共用 [cite: 36, 37, 38]
        W, H = self.width, self.height
        cx, cy = W / 2, H / 2
        # 使用傳入的 fov 參數計算焦距
        f = (W / 2) / np.tan(np.deg2rad(fov / 2))
        # f = (math.sqrt(W**2 + H**2) / 2) / np.tan(np.deg2rad(fov / 2))
        K = np.array([[f, 0, cx],
                      [0, f, cy],
                      [0, 0,  1]], dtype=np.float32)
        K_inv = np.linalg.inv(K)

        # 外參 (Extrinsics) - 根據 spec 硬編碼
        # [cite_start]前視相機 Camera1 [cite: 27]
        C_front = np.array([0, 1, 0], dtype=np.float32)
        # [cite_start]R_front 是單位矩陣, 因為姿態是 (0,0,0) [cite: 27]

        # [cite_start]BEV 相機 Camera2 [cite: 28]
        C_bev = np.array([0, 2.5, 0], dtype=np.float32)
        # [cite_start]姿態是 (-pi/2, 0, 0), 也就是繞X軸旋轉-90度 [cite: 28]
        # 這裡我們不使用傳入的 theta, 而是用 spec 中指定的 pitch
        pitch = -np.pi / 2
        R_bev = np.array([[1, 0, 0],
                        [0, np.cos(pitch), -np.sin(pitch)],
                        [0, np.sin(pitch),  np.cos(pitch)]], dtype=np.float32)
        R_bev_inv = R_bev.T # 旋轉矩陣的逆等於轉置

        projected_points = []
        
        # 2. for 所有在BEV圖像上點擊的點變換座標
        for p_bev in self.points:
            u, v = p_bev
            p_bev_homogeneous = np.array([u, v, 1], dtype=np.float32)

            # --- 步驟 1: BEV 像素 -> 3D 世界座標 ---
            
            # 將2D像素點逆投影到BEV相機座標系下, 得到一個方向向量
            p_cam_norm = K_inv @ p_bev_homogeneous # @ 是矩陣乘法
            
            # 將方向向量從BEV相機座標系轉到世界座標系
            direction_world = R_bev_inv @ p_cam_norm
            
            # 計算射線與地面 (Y=0) 的交點
            # 射線方程: P_w = C_bev + t * direction_world
            # Y分量為0: C_bev[1] + t * direction_world[1] = 0
            t = -C_bev[1] / direction_world[1]
            P_w = C_bev + t * direction_world # 得到3D世界座標

            # --- 步驟 2: 3D 世界座標 -> 3D 前視相機座標 ---

            # 將世界座標點轉換到前視相機座標系下 (相對於相機位置)
            P_relative_to_front = P_w - C_front
            
            # 轉換到標準相機座標系 (+Z向前, +Y向下)
            P_c_front_proj = np.array([
                P_relative_to_front[0],
                -P_relative_to_front[1],
                -P_relative_to_front[2]
            ], dtype=np.float32)
            
            # --- 步驟 3: 3D 前視相機座標 -> 2D 圖像座標 ---
            
            # 檢查點是否在相機前方
            if P_c_front_proj[2] <= 0:
                continue # 深度為負或零，點在相機後面或剛好在相機上, 忽略

            # 使用內參矩陣K將3D點投影到2D圖像平面
            p_front_homogeneous = K @ P_c_front_proj
            
            # 齊次除法得到最終像素座標
            u_front = p_front_homogeneous[0] / p_front_homogeneous[2]
            v_front = p_front_homogeneous[1] / p_front_homogeneous[2]

            # 檢查投影點是否在圖像範圍內
            # if 0 <= u_front < W and 0 <= v_front < H:
            projected_points.append([int(u_front), int(v_front)])

        return projected_points

    def show_image(self, new_pixels, img_name='projection.png', color=(0, 0, 255), alpha=0.4):
        """
            Show the projection result and fill the selected area on perspective(front) view image.
        """

        new_image = cv2.fillPoly(
            self.image.copy(), [np.array(new_pixels)], color)
        new_image = cv2.addWeighted(
            new_image, alpha, self.image, (1 - alpha), 0)

        cv2.imshow(
            f'Top to front view projection {img_name}', new_image)
        cv2.imwrite(img_name, new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return new_image


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        print(x, ' ', y)
        points.append([x, y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, str(x) + ',' + str(y), (x+5, y+5), font, 0.5, (0, 0, 255), 1)
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('image', img)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:

        print(x, ' ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        # cv2.putText(img, str(b) + ',' + str(g) + ',' + str(r), (x, y), font, 1, (255, 255, 0), 2)
        cv2.imshow('image', img)


if __name__ == "__main__":

    pitch_ang = -90

    front_rgb = "bev_data/front2.png"
    top_rgb = "bev_data/bev2.png"

    # click the pixels on window
    img = cv2.imread(top_rgb, 1)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    projection = Projection(front_rgb, points)
    new_pixels = projection.top_to_front(theta=pitch_ang)
    projection.show_image(new_pixels)
