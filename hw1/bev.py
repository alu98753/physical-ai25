import cv2
import numpy as np

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
        self.points = points # 將傳入的點儲存為實例屬性

    def top_to_front(self, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0, fov=90):
        f = 0.5 * self.width / np.tan(np.radians(fov / 2))
        K = np.array([
            [f, 0, self.width / 2],
            [0, f, self.height / 2],
            [0, 0, 1]
        ])

        # --- Extrinsic for BEV ---
        R_bev = cv2.Rodrigues(np.array([[-np.pi/2, 0, 0]]))[0]  # pitch -90
        t_bev = np.array([[0], [2.5], [0]])

        # --- Extrinsic for Front ---
        R_front = np.eye(3)
        t_front = np.array([[0], [1], [0]])

        new_pixels = []
        for x, y in self.points:
            # (1) BEV 像素 → 相機座標 (假設 Z=0 在地面)
            # 這裡需要 scale，先假設像素直接對應 (X, Z)，Y=0
            P_bev_cam = np.array([[x], [0], [y]])

            # (2) 轉到世界座標
            P_world = R_bev.T @ (P_bev_cam - t_bev)

            # (3) 轉到 Front 相機座標
            P_front_cam = R_front @ P_world + t_front

            # (4) 投影到影像
            uvw = K @ P_front_cam
            u, v = int(uvw[0]/uvw[2]), int(uvw[1]/uvw[2])
            new_pixels.append([u, v])

        print(f"Projected {len(new_pixels)} points to front view image.")
        print("Sample projected points (u, v):", new_pixels[:5])  # 顯示前5個點
        return new_pixels


    def show_image(self, new_pixels, img_name='projection.png', color=(0, 0, 255), alpha=0.4):
        """
            Show the projection result and fill the selected area on perspective(front) view image.
        """
        if not new_pixels:
            print("Warning: No points were projected into the front view image.")
            # 為了讓程式不報錯，即使沒有點也要顯示原圖
            cv2.imshow(f'Top to front view projection {img_name}', self.image)
            cv2.imwrite(img_name, self.image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return self.image

        new_image = self.image.copy()
        
        # 繪製多邊形 (要求用多邊形填充區域)
        # 由於點的順序是依據滑鼠點擊順序，通常可以直接連成多邊形。
        new_image = cv2.fillPoly(
            new_image, [np.array(new_pixels, dtype=np.int32)], color)
        
        # 疊加
        new_image = cv2.addWeighted(
            new_image, alpha, self.image, (1.0 - alpha), 0) # 修正: (1 - alpha) 應為 (1.0 - alpha)

        cv2.imshow(
            f'Top to front view projection {img_name}', new_image)
        cv2.imwrite(img_name, new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return new_image


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        print(f'Clicked: x={x}, y={y}')
        points.append([x, y])
        
        # 繪製點和座標 (可選)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        # cv2.putText(img, str(x) + ',' + str(y), (x+5, y+5), font, 0.5, (0, 0, 255), 1)
        cv2.imshow('image', img)

    # checking for right mouse clicks (通常用於取消或結束，這裡保持原樣)
    if event == cv2.EVENT_RBUTTONDOWN:

        # print(x, ' ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        # cv2.putText(img, str(b) + ',' + str(g) + ',' + str(r), (x, y), font, 1, (255, 255, 0), 2)
        cv2.imshow('image', img)


if __name__ == "__main__":

    # 注意: theta=-90 是 BEV 相機的俯仰角，但在投影公式中已經被隱含使用。
    # 這裡的 theta, phi, gamma, dx, dy, dz 主要是作為參數預留給更複雜的投影，
    # 在我們的簡化場景中，主要邏輯已在 top_to_front 內部硬編碼。
    pitch_ang = -90 

    # 確保這些路徑正確
    front_rgb = "bev_data/front2.png"
    top_rgb = "bev_data/bev2.png"
    
    # 檢查文件是否存在
    try:
        img = cv2.imread(top_rgb, 1)
        if img is None:
            raise FileNotFoundError(f"BEV image file not found or is empty: {top_rgb}")
        cv2.imshow('image', img)
    except Exception as e:
        print(f"Error loading BEV image: {e}. Please ensure you have the image file in the correct path.")
        exit()

    print("\n--- Task 1: BEV Projection ---")
    print(f"1. Displaying BEV image: {top_rgb}")
    print("2. Please click on the ground area in the image window using the LEFT mouse button.")
    print("3. Click ENTER/SPACE/any other key to close the window and proceed with projection.")
    
    # 點擊像素在視窗上ㄙㄟ
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if len(points) < 3:
        print(f"Warning: Only {len(points)} points clicked. You need at least 3 points to form a polygon. Exiting.")
    else:
        print(f"\nSelected {len(points)} points in BEV image.")
        
        # 載入 Front View 圖像
        try:
            projection = Projection(front_rgb, points)
        except FileNotFoundError as e:
            print(f"Error: {e}. Please ensure the front view image is at the correct path.")
            exit()
            
        print(f"3. Projecting selected area to Front View image: {front_rgb}")
        new_pixels = projection.top_to_front(theta=pitch_ang)
        
        # 顯示並儲存結果
        projection.show_image(new_pixels)
        print("4. Projection complete. Result saved as 'projection.png'")
        print("Please check the generated image for the result.")