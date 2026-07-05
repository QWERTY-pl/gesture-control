import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import math
import time
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

SMOOTHING_FACTOR = 0.4
VOLUME_STEP = 2
HAND_LANDMARKER_TASK_FILE = "hand_landmarker.task"

class HandGestureController:
    def __init__(self):
        self.last_mouse_x = SCREEN_WIDTH // 2
        self.last_mouse_y = SCREEN_HEIGHT // 2
        self.current_volume = 50
        self.screenshot_taken = False
        self.click_enabled = True
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
        
    def _get_font(self):
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
    
    def _draw_chinese_text(self, image, text, position, color, font_size=24):
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text(position, text, font=self.font, fill=tuple(color))
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    def _download_model(self):
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
        return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)
    
    def _get_palm_center(self, landmarks):
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
    
    def _detect_finger_extended(self, landmarks, tip_idx, pip_idx, is_thumb=False):
        if len(landmarks) <= tip_idx:
            return False
        
        tip = landmarks[tip_idx]
        pip = landmarks[pip_idx]
        
        if is_thumb:
            return tip.x < pip.x
        else:
            return tip.y < pip.y - 0.02
    
    def _detect_gesture(self, landmarks):
        if len(landmarks) < 2:
            return '未知'
        
        thumb_extended = self._detect_finger_extended(landmarks, 4, 3, is_thumb=True)
        index_extended = self._detect_finger_extended(landmarks, 8, 6)
        middle_extended = self._detect_finger_extended(landmarks, 12, 10)
        ring_extended = self._detect_finger_extended(landmarks, 16, 14)
        pinky_extended = self._detect_finger_extended(landmarks, 20, 18)
        
        fingers_extended = sum([index_extended, middle_extended, ring_extended, pinky_extended])
        
        palm_open = False
        if len(landmarks) >= 9 and len(landmarks) >= 21:
            index_tip = landmarks[8]
            pinky_tip = landmarks[20]
            index_pip = landmarks[6]
            wrist = landmarks[0]
            palm_open = self._get_distance(index_tip, pinky_tip) > self._get_distance(wrist, index_pip) * 0.8
        
        if palm_open and fingers_extended == 4 and thumb_extended:
            return '手掌'
        elif pinky_extended and fingers_extended == 1:
            return '小指'
        elif index_extended and middle_extended and fingers_extended == 2:
            return '两指'
        elif fingers_extended == 3:
            return '三指'
        elif index_extended and fingers_extended == 1:
            return '食指'
        elif thumb_extended and not index_extended and fingers_extended == 0 and len(landmarks) >= 6:
            return '拇指'
        elif fingers_extended == 0 and not thumb_extended and len(landmarks) >= 6:
            return '握拳'
        else:
            return '握拳'
    
    def _get_stable_gesture(self, gesture):
        self.gesture_history.append(gesture)
        if len(self.gesture_history) > self.gesture_stability:
            self.gesture_history.pop(0)
        
        if len(self.gesture_history) == self.gesture_stability:
            most_common = max(set(self.gesture_history), key=self.gesture_history.count)
            return most_common
        return gesture
    
    def _map_hand_to_screen(self, landmarks, image_width, image_height):
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
    
    def _get_quadrant(self, landmarks):
        palm_center = self._get_palm_center(landmarks)
        x, y = palm_center['x'], palm_center['y']
        
        if x < 0.5 and y < 0.5:
            return '第二象限'
        elif x >= 0.5 and y < 0.5:
            return '第一象限'
        elif x < 0.5 and y >= 0.5:
            return '第三象限'
        else:
            return '第四象限'
    
    def _draw_quadrants(self, image):
        image_height, image_width, _ = image.shape
        center_x = image_width // 2
        center_y = image_height // 2
        
        cv2.line(image, (center_x, 0), (center_x, image_height), (255, 255, 0), 2)
        cv2.line(image, (0, center_y), (image_width, center_y), (255, 255, 0), 2)
        
        image = self._draw_chinese_text(image, '第一象限', (center_x + 10, 10), (255, 255, 0), 18)
        image = self._draw_chinese_text(image, '第二象限', (10, 10), (255, 255, 0), 18)
        image = self._draw_chinese_text(image, '第三象限', (10, center_y + 10), (255, 255, 0), 18)
        image = self._draw_chinese_text(image, '第四象限', (center_x + 10, center_y + 10), (255, 255, 0), 18)
        
        return image
    
    def _draw_landmarks(self, image, landmarks, connections):
        image_height, image_width, _ = image.shape
        
        for landmark in landmarks:
            x = int(landmark.x * image_width)
            y = int(landmark.y * image_height)
            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
        
        palm_center = self._get_palm_center(landmarks)
        palm_x = int(palm_center['x'] * image_width)
        palm_y = int(palm_center['y'] * image_height)
        cv2.circle(image, (palm_x, palm_y), 8, (0, 0, 255), -1)
        cv2.circle(image, (palm_x, palm_y), 10, (0, 0, 255), 2)
        
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
        if gesture == '握拳':
            screen_x, screen_y = self._map_hand_to_screen(landmarks, CAMERA_WIDTH, CAMERA_HEIGHT)
            
            self.last_mouse_x = self.last_mouse_x * (1 - SMOOTHING_FACTOR) + screen_x * SMOOTHING_FACTOR
            self.last_mouse_y = self.last_mouse_y * (1 - SMOOTHING_FACTOR) + screen_y * SMOOTHING_FACTOR
            
            pyautogui.moveTo(self.last_mouse_x, self.last_mouse_y, duration=0.01)

        elif gesture == '食指' and self.click_enabled:
            current_time = time.time()
            if current_time - self.last_click_time > 0.5:
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
                screenshot = pyautogui.screenshot()
                screenshot.save(f'screenshot_{int(time.time())}.png')
                self.screenshot_taken = True
                print("截图已保存")

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
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def run(self):
        try:
            self._download_model()
            
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
                (5, 9), (9, 10), (10, 11), (11, 12),
                (9, 13), (13, 14), (14, 15), (15, 16),
                (13, 17), (17, 18), (18, 19), (19, 20),
                (0, 17)
            ]

            with HandLandmarker.create_from_options(options) as self.landmarker:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    print("无法打开摄像头")
                    return
                
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

                print("手势识别电脑控制程序已启动")
                print("快捷键: Q-退出, C-切换点击功能")
                
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

                            quadrant = self._get_quadrant(landmarks)
                            image = self._draw_chinese_text(image, f'手势: {gesture}', (10, 30), (0, 255, 0))
                            image = self._draw_chinese_text(image, f'象限: {quadrant}', (150, 30), (0, 255, 255))
                    else:
                        self.gesture_history = []
                        image = self._draw_chinese_text(image, '手势: 未检测到手', (10, 30), (0, 0, 255))
                        image = self._draw_chinese_text(image, '象限: 无', (150, 30), (0, 255, 255))

                    image = self._draw_quadrants(image)
                    image = self._draw_chinese_text(image, f'音量: {self.current_volume}%', (10, 60), (255, 0, 0))
                    image = self._draw_chinese_text(image, f'帧率: {self.fps} FPS', (150, 60), (0, 128, 255))
                    image = self._draw_chinese_text(image, 'Q-退出', (10, CAMERA_HEIGHT - 40), (0, 0, 255), 20)
                    image = self._draw_chinese_text(image, 'C-切换点击', (150, CAMERA_HEIGHT - 40), (0, 0, 255), 20)
                    image = self._draw_chinese_text(image, f'点击: {"开启" if self.click_enabled else "关闭"}', (300, CAMERA_HEIGHT - 40), (0, 255, 255), 20)

                    cv2.imshow('手势识别电脑控制程序', image)

                    key = cv2.waitKey(5) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('c'):
                        self.click_enabled = not self.click_enabled
                        print(f'点击功能 {"开启" if self.click_enabled else "关闭"}')
                    
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
    controller = HandGestureController()
    controller.run()