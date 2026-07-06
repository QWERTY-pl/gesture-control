import cv2
import mediapipe as mp
import pyautogui
import math
import time
import os
import numpy as np
import json
import joblib
from PIL import Image, ImageDraw, ImageFont
import ctypes
import ctypes.wintypes

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

SMOOTHING_FACTOR = 0.4
VOLUME_STEP = 2
HAND_LANDMARKER_TASK_FILE = "hand_landmarker.task"
SCREENSHOT_DIR = "screenshots"

GESTURES_EN = ['Fist', 'Index', 'Two', 'Three', 'Pinky', 'Thumb', 'Palm']
GESTURES_CN = ['握拳', '食指', '两指', '三指', '小指', '拇指', '手掌']
GESTURE_MAP = dict(zip(GESTURES_EN, GESTURES_CN))

#MediaPipe Hand Landmarker模型
class HandGestureController:
    def __init__(self):
        """初始化手势控制器
        
        设置鼠标位置、音量、状态标志等参数，加载字体和机器学习模型。
        """
        self.last_mouse_x = SCREEN_WIDTH // 2
        self.last_mouse_y = SCREEN_HEIGHT // 2
        self.current_volume = 50
        self.screenshot_taken = False
        self.click_mode = 'single'
        self.last_palm_y = 0
        self.last_click_time = 0
        self.last_volume_time = 0
        self.gesture_history = []
        self.gesture_stability = 3
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.font = self._get_font()
        self.landmarker = None
        self.cap = None
        self.ml_model = None
        self.use_ml = False
        self._load_ml_model()

    def _get_font(self):
        """获取系统中文字体路径
        按优先级查找Windows/macOS系统中的中文字体，用于PIL绘制中文。
        Returns:
            ImageFont对象，支持中文显示
        """
        font_paths = [
            "C:/Windows/Fonts/simsun.ttc",
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/msyhbd.ttc",
            "/Library/Fonts/Songti.ttc",
            "/System/Library/Fonts/PingFang.ttc"
        ]
        for path in font_paths:
            if os.path.exists(path):
                return ImageFont.truetype(path, 24)
        return ImageFont.load_default()

    def _load_ml_model(self):
        """加载机器学习模型
        从文件加载训练好的随机森林模型，用于手势识别。
        如果模型不存在或加载失败，使用规则识别。
        """
        if os.path.exists('gesture_model.pkl'):
            try:
                self.ml_model = joblib.load('gesture_model.pkl')
                print("机器学习模型加载成功")
            except Exception as e:
                print(f"模型加载失败: {e}")
                self.ml_model = None
        else:
            print("未找到机器学习模型，使用规则识别")
            self.ml_model = None

    def _extract_features(self, landmarks):
        """从手部关键点提取特征向量
        提取关键点坐标、手指方向差异、手掌宽度等特征，用于机器学习模型输入。
        Args:
            landmarks: MediaPipe手部关键点列表
        Returns:
            numpy数组，形状为(1, n_features)，包含所有提取的特征
        """
        features = []

        for lm in landmarks:
            features.append(lm.x)
            features.append(lm.y)
            features.append(lm.z)

        if len(landmarks) < 21:
            features.extend([0] * (63 - len(features)))

        fingers = [
            (8, 6, 5),
            (12, 10, 9),
            (16, 14, 13),
            (20, 18, 17),
            (4, 3, 2)
        ]

        for tip, pip, mcp in fingers:
            if tip < len(landmarks):
                tip_y = landmarks[tip].y
                mcp_y = landmarks[mcp].y
                features.append(tip_y - mcp_y)

                tip_x = landmarks[tip].x
                mcp_x = landmarks[mcp].x
                features.append(tip_x - mcp_x)
            else:
                features.extend([0, 0])

        if len(landmarks) >= 21:
            palm_width = math.sqrt(
                (landmarks[8].x - landmarks[20].x) ** 2 +
                (landmarks[8].y - landmarks[20].y) ** 2
            )
            wrist_to_middle = math.sqrt(
                (landmarks[0].x - landmarks[9].x) ** 2 +
                (landmarks[0].y - landmarks[9].y) ** 2
            )
            features.append(palm_width)
            features.append(wrist_to_middle)
            features.append(palm_width / wrist_to_middle if wrist_to_middle > 0 else 0)
        else:
            features.extend([0, 0, 0])

        return np.array(features).reshape(1, -1)

    def _draw_chinese_text(self, image, text, position, color, font_size=24):
        """使用PIL在图像上绘制中文文本
        解决OpenCV cv2.putText不支持中文的问题。
        Args:
            image: OpenCV图像（BGR格式）
            text: 要绘制的中文文本
            position: 文本位置(x, y)
            color: 文本颜色(BGR格式)
            font_size: 字体大小，默认24
            
        Returns:
            绘制了中文文本的图像
        """
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text(position, text, font=self.font, fill=tuple(color))
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def _download_model(self):
        """下载MediaPipe手部检测模型
        如果本地不存在模型文件，从Google服务器下载hand_landmarker.task。
        """
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        if not os.path.exists(HAND_LANDMARKER_TASK_FILE):
            print("正在下载手部检测模型...")
            try:
                urllib.request.urlretrieve(url, HAND_LANDMARKER_TASK_FILE)
                print("模型下载完成")
            except Exception as e:
                print(f"模型下载失败: {e}")
                raise

    def _get_distance(self, point1, point2):
        """计算两个关键点之间的欧氏距离
        Args:
            point1: 第一个关键点
            point2: 第二个关键点
            
        Returns:
            两点之间的距离
        """
        return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

    def _get_palm_center(self, landmarks):
        """计算掌心中心点坐标
        根据手指根部关键点(5, 9, 13, 17)计算掌心中心，用于鼠标移动定位。
        支持部分关键点缺失的情况。   
        Args:
            landmarks: 手部关键点列表
            
        Returns:
            字典，包含'x'和'y'坐标
        """
        palm_points = []
        if len(landmarks) >= 6:
            palm_points.append(landmarks[5])
        if len(landmarks) >= 10:
            palm_points.append(landmarks[9])
        if len(landmarks) >= 14:
            palm_points.append(landmarks[13])
        if len(landmarks) >= 18:
            palm_points.append(landmarks[17])

        if palm_points:
            x = sum(p.x for p in palm_points) / len(palm_points)
            y = sum(p.y for p in palm_points) / len(palm_points)
            return {'x': x, 'y': y}
        elif len(landmarks) >= 1:
            return {'x': landmarks[0].x, 'y': landmarks[0].y}
        else:
            return {'x': 0.5, 'y': 0.5}

    def _detect_finger_extended(self, landmarks, tip_idx, pip_idx, mcp_idx, is_thumb=False):
        """判断单根手指是否伸出
        拇指使用特殊的水平偏移和距离比例判断，其他手指使用指尖到手腕距离与指根到手腕距离的比例判断。
        Args:
            landmarks: 手部关键点列表
            tip_idx: 指尖关键点索引
            pip_idx: 近端指间关节索引
            mcp_idx: 掌指关节索引
            is_thumb: 是否为拇指，默认False
            
        Returns:
            True表示手指伸出，False表示手指弯曲
        """
        if len(landmarks) <= tip_idx:
            return False

        tip = landmarks[tip_idx]
        pip = landmarks[pip_idx]
        mcp = landmarks[mcp_idx]
        wrist = landmarks[0]

        if is_thumb:
            thumb_tip = landmarks[4]
            thumb_ip = landmarks[3]
            thumb_mcp = landmarks[2]
            index_mcp = landmarks[5]
            pinky_mcp = landmarks[17]
            middle_mcp = landmarks[9]

            palm_width = self._get_distance(index_mcp, pinky_mcp)
            thumb_tip_to_middle = self._get_distance(thumb_tip, middle_mcp)
            thumb_mcp_to_middle = self._get_distance(thumb_mcp, middle_mcp)

            if thumb_mcp_to_middle > 0 and thumb_tip_to_middle / thumb_mcp_to_middle > 1.5:
                return True

            thumb_tip_x = thumb_tip.x
            index_mcp_x = index_mcp.x
            pinky_mcp_x = pinky_mcp.x
            hand_center_x = (index_mcp_x + pinky_mcp_x) / 2

            thumb_tip_to_center = abs(thumb_tip_x - hand_center_x)
            palm_half_width = palm_width / 2

            if thumb_tip_to_center > palm_half_width * 1.2:
                return True

            return False
        else:
            dist_tip_to_wrist = self._get_distance(tip, wrist)
            dist_mcp_to_wrist = self._get_distance(mcp, wrist)

            if dist_mcp_to_wrist > 0 and dist_tip_to_wrist / dist_mcp_to_wrist > 1.5:
                return True

            return False

    def _detect_gesture_rule(self, landmarks):
        """使用几何规则识别手势（默认模式）
        根据手指伸出的组合判断手势类型：握拳、食指、两指、三指、小指、拇指、手掌。
        Args:
            landmarks: 手部关键点列表（至少21个）
        Returns:
            手势名称（中文），'未知'表示无法识别
        """
        if len(landmarks) < 21:
            return '未知'

        thumb_extended = self._detect_finger_extended(landmarks, 4, 3, 2, is_thumb=True)
        index_extended = self._detect_finger_extended(landmarks, 8, 6, 5)
        middle_extended = self._detect_finger_extended(landmarks, 12, 10, 9)
        ring_extended = self._detect_finger_extended(landmarks, 16, 14, 13)
        pinky_extended = self._detect_finger_extended(landmarks, 20, 18, 17)

        fingers_extended = sum([index_extended, middle_extended, ring_extended, pinky_extended])

        index_tip = landmarks[8]
        pinky_tip = landmarks[20]
        wrist = landmarks[0]
        middle_mcp = landmarks[9]

        palm_width = self._get_distance(index_tip, pinky_tip)
        wrist_to_middle = self._get_distance(wrist, middle_mcp)
        palm_open_ratio = palm_width / wrist_to_middle if wrist_to_middle > 0 else 0

        if fingers_extended == 4 and thumb_extended:
            return '手掌'
        elif index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return '食指'
        elif index_extended and middle_extended and not ring_extended and not pinky_extended:
            return '两指'
        elif index_extended and middle_extended and ring_extended and not pinky_extended:
            return '三指'
        elif pinky_extended and not index_extended and not middle_extended and not ring_extended:
            return '小指'
        elif thumb_extended and fingers_extended == 0:
            return '拇指'
        elif fingers_extended == 0 and not thumb_extended:
            return '握拳'
        else:
            return '未知'

    def _detect_gesture(self, landmarks):
        """手势识别入口
        根据当前模式选择规则识别或机器学习识别。
        ML模式下置信度低于60%时自动回退到规则识别。
        Args:
            landmarks: 手部关键点列表
        Returns:
            手势名称（中文）
        """
        if len(landmarks) < 6:
            return '未知'

        rule_gesture = self._detect_gesture_rule(landmarks)

        if self.use_ml and self.ml_model is not None:
            features = self._extract_features(landmarks)
            try:
                prediction = self.ml_model.predict(features)
                probs = self.ml_model.predict_proba(features)
                max_prob = np.max(probs)
                en_label = GESTURES_EN[prediction[0]]
                ml_gesture = GESTURE_MAP.get(en_label, '未知')

                if max_prob > 0.6 and ml_gesture != '未知':
                    return ml_gesture
                else:
                    return rule_gesture
            except Exception as e:
                return rule_gesture

        return rule_gesture

    def _get_stable_gesture(self, gesture):
        """手势稳定性处理
        取历史帧中出现次数最多的手势，减少识别抖动。
        Args:
            gesture: 当前帧识别的手势 
        Returns:
            稳定后的手势
        """
        self.gesture_history.append(gesture)
        if len(self.gesture_history) > self.gesture_stability:
            self.gesture_history.pop(0)

        if len(self.gesture_history) == self.gesture_stability:
            most_common = max(set(self.gesture_history), key=self.gesture_history.count)
            return most_common
        return gesture

    def _map_hand_to_screen(self, landmarks, image_width, image_height):
        """将掌心坐标映射到屏幕坐标
        使用非线性映射扩展边缘区域，解决手掌移动到画面边缘时鼠标无法到达屏幕边缘的问题。
        Args:
            landmarks: 手部关键点列表
            image_width: 图像宽度
            image_height: 图像高度
        Returns:
            (screen_x, screen_y): 屏幕坐标
        """
        palm_center = self._get_palm_center(landmarks)
        x = palm_center['x']
        y = palm_center['y']

        margin = 0.10
        y_min = margin
        y_max = 1.0 - margin

        if y < y_min:
            normalized_y = y / y_min * 0.2
        elif y > y_max:
            normalized_y = 0.8 + (y - y_max) / margin * 0.2
        else:
            normalized_y = 0.2 + (y - y_min) / (y_max - y_min) * 0.6

        screen_x = x * SCREEN_WIDTH
        screen_y = normalized_y * SCREEN_HEIGHT

        screen_x = max(0, min(SCREEN_WIDTH - 1, screen_x))
        screen_y = max(0, min(SCREEN_HEIGHT - 1, screen_y))

        return screen_x, screen_y



    def _draw_landmarks(self, image, landmarks, connections):
        """在画面上绘制手部关键点和连接线
        关键点用绿色圆点表示，连接线用蓝色线条表示。
        Args:
            image: OpenCV图像
            landmarks: 手部关键点列表
            connections: 关键点连接关系列表
        """
        image_height, image_width, _ = image.shape

        for landmark in landmarks:
            x = int(landmark.x * image_width)
            y = int(landmark.y * image_height)
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_point = landmarks[start_idx]
                end_point = landmarks[end_idx]
                start_x = int(start_point.x * image_width)
                start_y = int(start_point.y * image_height)
                end_x = int(end_point.x * image_width)
                end_y = int(end_point.y * image_height)
                cv2.line(image, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)

    def _handle_gesture(self, gesture, landmarks):
        """根据识别的手势执行对应操作
        手势映射:
        握拳 → 鼠标移动
        食指 → 左键点击
        两指 → 右键点击
        三指 → 增加音量
        小指 → 减少音量
        拇指 → 截图保存
        手掌 → 上下滚动
        Args:
            gesture: 手势名称（中文）
            landmarks: 手部关键点列表
        """
        if gesture == '握拳':
            screen_x, screen_y = self._map_hand_to_screen(landmarks, CAMERA_WIDTH, CAMERA_HEIGHT)

            self.last_mouse_x = self.last_mouse_x * (1 - SMOOTHING_FACTOR) + screen_x * SMOOTHING_FACTOR
            self.last_mouse_y = self.last_mouse_y * (1 - SMOOTHING_FACTOR) + screen_y * SMOOTHING_FACTOR

            pyautogui.moveTo(self.last_mouse_x, self.last_mouse_y, duration=0.01)

        elif gesture == '食指':
            current_time = time.time()
            if current_time - self.last_click_time > 0.5:
                if self.click_mode == 'double':
                    pyautogui.doubleClick()
                else:
                    pyautogui.click()
                self.last_click_time = current_time

        elif gesture == '两指':
            current_time = time.time()
            if current_time - self.last_click_time > 0.5:
                pyautogui.click(button='right')
                self.last_click_time = current_time

        elif gesture == '三指':
            current_time = time.time()
            if current_time - self.last_volume_time > 0.3:
                self.current_volume = min(100, self.current_volume + VOLUME_STEP)
                pyautogui.press('volumeup')
                self.last_volume_time = current_time

        elif gesture == '小指':
            current_time = time.time()
            if current_time - self.last_volume_time > 0.3:
                self.current_volume = max(0, self.current_volume - VOLUME_STEP)
                pyautogui.press('volumedown')
                self.last_volume_time = current_time

        elif gesture == '拇指':
            if not self.screenshot_taken:
                os.makedirs(SCREENSHOT_DIR, exist_ok=True)
                screenshot = pyautogui.screenshot()
                screenshot_path = f'{SCREENSHOT_DIR}/screenshot_{int(time.time())}.png'
                screenshot.save(screenshot_path)
                self.screenshot_taken = True
                print(f"截图已保存至: {screenshot_path}")

        elif gesture == '手掌':
            self.screenshot_taken = False
            wrist_y = landmarks[0].y
            if self.last_palm_y != 0:
                y_diff = wrist_y - self.last_palm_y
                if abs(y_diff) > 0.02:
                    scroll_amount = int(y_diff * 100)
                    pyautogui.scroll(scroll_amount)
            self.last_palm_y = wrist_y

    def _update_fps(self):
        """计算并更新帧率
        每秒计算一次帧率，用于界面显示。
        """
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_fps_time = current_time

    def _show_startup_screen(self):
        """显示启动界面（Tkinter）
        展示手势映射说明和启动/退出按钮。
        Returns:
            True表示用户点击了启动，False表示退出
        """
        import tkinter as tk

        self.startup_window = tk.Tk()
        self.startup_window.title('手势识别电脑控制程序')
        self.startup_window.geometry('500x600')
        self.startup_window.resizable(False, False)

        screen_width = self.startup_window.winfo_screenwidth()
        screen_height = self.startup_window.winfo_screenheight()
        x = (screen_width - 500) // 2
        y = (screen_height - 600) // 2
        self.startup_window.geometry(f'500x600+{x}+{y}')

        bg_color = '#222222'
        self.startup_window.configure(bg=bg_color)

        title_label = tk.Label(self.startup_window, text='手势识别电脑控制程序',
                              font=('SimSun', 20, 'bold'), bg=bg_color, fg='#00FFFF')
        title_label.pack(pady=(30, 15))

        separator = tk.Frame(self.startup_window, height=1, bg='#444444')
        separator.pack(fill=tk.X, padx=50, pady=(0, 15))

        gestures_frame = tk.Frame(self.startup_window, bg=bg_color)
        gestures_frame.pack(pady=10)

        gestures = [
            ('握拳', '鼠标移动'),
            ('食指', '左键点击'),
            ('两指', '右键点击'),
            ('三指', '增加音量'),
            ('小指', '减少音量'),
            ('拇指', '截图保存'),
            ('手掌', '上下滚动'),
        ]

        for gesture, action in gestures:
            row_frame = tk.Frame(gestures_frame, bg=bg_color)
            row_frame.pack(fill=tk.X, padx=50, pady=4)

            gesture_label = tk.Label(row_frame, text=gesture, font=('SimSun', 12),
                                    bg=bg_color, fg='#00FF00', width=6)
            gesture_label.pack(side=tk.LEFT)

            arrow_label = tk.Label(row_frame, text='→', font=('SimSun', 12),
                                  bg=bg_color, fg='#FFFF00')
            arrow_label.pack(side=tk.LEFT, padx=15)

            action_label = tk.Label(row_frame, text=action, font=('SimSun', 12),
                                   bg=bg_color, fg='#FFFFFF')
            action_label.pack(side=tk.LEFT)

        separator2 = tk.Frame(self.startup_window, height=1, bg='#444444')
        separator2.pack(fill=tk.X, padx=50, pady=(15, 25))

        button_frame = tk.Frame(self.startup_window, bg=bg_color)
        button_frame.pack(pady=20)

        self.start_button = tk.Button(button_frame, text='启动程序',
                                     font=('SimSun', 14, 'bold'),
                                     bg='#0066CC', fg='white',
                                     activebackground='#0088FF', activeforeground='white',
                                     command=self._on_start_click,
                                     width=10, height=2)
        self.start_button.pack(side=tk.LEFT, padx=30)

        self.exit_button = tk.Button(button_frame, text='退出',
                                    font=('SimSun', 12),
                                    bg='#666666', fg='white',
                                    activebackground='#888888', activeforeground='white',
                                    command=self._on_exit_click,
                                    width=8, height=2)
        self.exit_button.pack(side=tk.LEFT, padx=30)

        self.startup_result = None

        self.startup_window.mainloop()

        return self.startup_result == 'start'

    def _on_start_click(self):
        """启动按钮点击事件
        设置启动结果并关闭启动窗口。
        """
        self.startup_result = 'start'
        self.startup_window.destroy()

    def _on_exit_click(self):
        """退出按钮点击事件
        设置退出结果并关闭启动窗口。
        """
        self.startup_result = 'exit'
        self.startup_window.destroy()

    def run(self):
        """主程序入口，启动摄像头和手势识别循环
        
        流程:
        1. 下载MediaPipe模型
        2. 显示启动界面
        3. 初始化手部检测器
        4. 启动摄像头循环
        5. 实时检测手势并执行对应操作
        """
        try:
            self._download_model()

            if not self._show_startup_screen():
                return

            BaseOptions = mp.tasks.BaseOptions
            HandLandmarker = mp.tasks.vision.HandLandmarker
            HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode

            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=HAND_LANDMARKER_TASK_FILE),
                running_mode=VisionRunningMode.IMAGE,
                num_hands=1)

            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20),
                (5, 9), (9, 13), (13, 17)
            ]

            self.landmarker = HandLandmarker.create_from_options(options)
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("无法打开摄像头")
                return

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

            print("手势识别电脑控制程序已启动")
            print("快捷键: Q-退出, C-切换点击功能, M-切换ML/规则识别模式")
            print(f"当前模式: {'ML识别' if self.use_ml else '规则识别'}")

            cv2.namedWindow('GestureControl', cv2.WINDOW_NORMAL)
            user32 = ctypes.WinDLL('user32', use_last_error=True)
            FindWindowW = user32.FindWindowW
            SetWindowTextW = user32.SetWindowTextW

            FindWindowW.argtypes = [ctypes.wintypes.LPCWSTR, ctypes.wintypes.LPCWSTR]
            FindWindowW.restype = ctypes.wintypes.HWND

            SetWindowTextW.argtypes = [ctypes.wintypes.HWND, ctypes.wintypes.LPCWSTR]
            SetWindowTextW.restype = ctypes.wintypes.BOOL

            hwnd = FindWindowW(None, 'GestureControl')
            if hwnd:
                SetWindowTextW(hwnd, '手势识别电脑控制程序')

            while self.cap.isOpened():
                success, image = self.cap.read()
                if not success:
                    print("忽略空的摄像头帧")
                    continue

                image = cv2.flip(image, 1)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                result = self.landmarker.detect(mp_image)

                if result.hand_landmarks:
                    for landmarks in result.hand_landmarks:
                        self._draw_landmarks(image, landmarks, connections)

                        raw_gesture = self._detect_gesture(landmarks)
                        gesture = self._get_stable_gesture(raw_gesture)

                        self._handle_gesture(gesture, landmarks)

                        image = self._draw_chinese_text(image, f'手势: {gesture}', (10, 30), (0, 255, 0), 24)

                        thumb_ext = self._detect_finger_extended(landmarks, 4, 3, 2, is_thumb=True)
                        index_ext = self._detect_finger_extended(landmarks, 8, 6, 5)
                        middle_ext = self._detect_finger_extended(landmarks, 12, 10, 9)
                        ring_ext = self._detect_finger_extended(landmarks, 16, 14, 13)
                        pinky_ext = self._detect_finger_extended(landmarks, 20, 18, 17)

                        finger_status = f'拇指:{"伸" if thumb_ext else "弯"} 食指:{"伸" if index_ext else "弯"} 中指:{"伸" if middle_ext else "弯"} 无名指:{"伸" if ring_ext else "弯"} 小指:{"伸" if pinky_ext else "弯"}'
                        image = self._draw_chinese_text(image, finger_status, (10, 60), (255, 255, 0), 14)
                else:
                    self.gesture_history = []
                    image = self._draw_chinese_text(image, '手势: 未检测到手', (10, 30), (0, 0, 255), 24)

                image = self._draw_chinese_text(image, f'音量: {self.current_volume}%', (450, 30), (255, 0, 0), 20)
                image = self._draw_chinese_text(image, f'帧率: {self.fps} FPS', (450, 60), (0, 128, 255), 20)
                image = self._draw_chinese_text(image, 'Q-退出', (10, CAMERA_HEIGHT - 40), (0, 0, 255), 18)
                image = self._draw_chinese_text(image, 'C-单击/双击', (110, CAMERA_HEIGHT - 40), (0, 0, 255), 18)
                image = self._draw_chinese_text(image, 'M-切换ML', (200, CAMERA_HEIGHT - 40), (0, 0, 255), 18)
                image = self._draw_chinese_text(image, f'模式: {"ML" if self.use_ml else "规则"}', (330, CAMERA_HEIGHT - 40), (0, 255, 255), 18)
                image = self._draw_chinese_text(image, 
                    f'点击: {"双击" if self.click_mode == "double" else "单击"}', 
                    (430, CAMERA_HEIGHT - 40), 
                    (0, 255, 0), 18)

                cv2.imshow('GestureControl', image)

                key = cv2.waitKey(5) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.click_mode = 'double' if self.click_mode == 'single' else 'single'
                    print(f'点击模式: {self.click_mode}')
                elif key == ord('m'):
                    self.use_ml = not self.use_ml
                    print(f'ML识别 {"开启" if self.use_ml else "关闭"}')

                self._update_fps()

        except Exception as e:
            print(f"程序运行出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
            print("程序已退出")


if __name__ == '__main__':
    """
    手势识别电脑控制程序
    ----------------------
    使用MediaPipe检测手部关键点，通过规则/ML模型识别手势，
    映射为鼠标移动、点击、音量调节、截图等电脑操作。
    手势映射:
    握拳 → 鼠标移动    食指 → 左键点击    两指 → 右键点击
    三指 → 增加音量    小指 → 减少音量    拇指 → 截图保存
    手掌 → 上下滚动

    快捷键: Q-退出  C-切换单击/双击  M-切换ML/规则识别
    """
    controller = HandGestureController()
    controller.run()