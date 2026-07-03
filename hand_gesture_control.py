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

SMOOTHING_FACTOR = 0.5
last_mouse_x, last_mouse_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2

VOLUME_STEP = 2
current_volume = 50

screenshot_taken = False
click_enabled = True

last_palm_y = 0

HAND_LANDMARKER_TASK_FILE = "hand_landmarker.task"

def get_font():
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

font = get_font()

def draw_chinese_text(image, text, position, color, font_size=24):
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=tuple(color))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def download_model():
    import urllib.request
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    if not os.path.exists(HAND_LANDMARKER_TASK_FILE):
        print("正在下载手部检测模型...")
        urllib.request.urlretrieve(url, HAND_LANDMARKER_TASK_FILE)
        print("模型下载完成")

def get_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

def get_palm_center(landmarks):
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

def detect_gesture(landmarks):
    if len(landmarks) < 2:
        return '未知'
    
    wrist = landmarks[0]
    
    thumb_extended = False
    index_extended = False
    middle_extended = False
    ring_extended = False
    pinky_extended = False
    
    if len(landmarks) >= 5:
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_extended = thumb_tip.x < thumb_ip.x
    
    if len(landmarks) >= 9:
        index_tip = landmarks[8]
        index_pip = landmarks[6]
        index_extended = index_tip.y < index_pip.y
    
    if len(landmarks) >= 13:
        middle_tip = landmarks[12]
        middle_pip = landmarks[10]
        middle_extended = middle_tip.y < middle_pip.y
    
    if len(landmarks) >= 17:
        ring_tip = landmarks[16]
        ring_pip = landmarks[14]
        ring_extended = ring_tip.y < ring_pip.y
    
    if len(landmarks) >= 21:
        pinky_tip = landmarks[20]
        pinky_pip = landmarks[18]
        pinky_extended = pinky_tip.y < pinky_pip.y

    fingers_extended = sum([index_extended, middle_extended, ring_extended, pinky_extended])

    palm_open = False
    if len(landmarks) >= 9 and len(landmarks) >= 21:
        index_tip = landmarks[8]
        pinky_tip = landmarks[20]
        index_pip = landmarks[6]
        palm_open = get_distance(index_tip, pinky_tip) > get_distance(wrist, index_pip) * 0.8

    if palm_open and fingers_extended == 4 and thumb_extended:
        return '手掌'
    elif fingers_extended == 0 and not thumb_extended and len(landmarks) >= 6:
        return '握拳'
    elif index_extended and fingers_extended == 1:
        return '食指'
    elif index_extended and middle_extended and fingers_extended == 2:
        return '两指'
    elif fingers_extended == 3:
        return '三指'
    elif pinky_extended and fingers_extended == 1:
        return '小指'
    elif thumb_extended and not index_extended and fingers_extended == 0 and len(landmarks) >= 6:
        return '拇指'
    else:
        return '握拳'

def map_hand_to_screen(landmarks, image_width, image_height):
    palm_center = get_palm_center(landmarks)
    x = palm_center['x']
    y = palm_center['y']
    
    margin = 0.15
    y_min = margin
    y_max = 1.0 - margin
    
    if y < y_min:
        normalized_y = y / y_min * 0.3
    elif y > y_max:
        normalized_y = 0.7 + (y - y_max) / margin * 0.3
    else:
        normalized_y = 0.3 + (y - y_min) / (y_max - y_min) * 0.4
    
    screen_x = x * SCREEN_WIDTH
    screen_y = normalized_y * SCREEN_HEIGHT
    
    screen_x = max(0, min(SCREEN_WIDTH - 1, screen_x))
    screen_y = max(0, min(SCREEN_HEIGHT - 1, screen_y))
    
    return screen_x, screen_y

def draw_landmarks(image, landmarks, connections):
    image_height, image_width, _ = image.shape
    
    for landmark in landmarks:
        x = int(landmark.x * image_width)
        y = int(landmark.y * image_height)
        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
    
    palm_center = get_palm_center(landmarks)
    palm_x = int(palm_center['x'] * image_width)
    palm_y = int(palm_center['y'] * image_height)
    cv2.circle(image, (palm_x, palm_y), 8, (0, 0, 255), -1)
    cv2.circle(image, (palm_x, palm_y), 10, (0, 0, 255), 2)
    
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]
        start_point = landmarks[start_idx]
        end_point = landmarks[end_idx]
        start_x = int(start_point.x * image_width)
        start_y = int(start_point.y * image_height)
        end_x = int(end_point.x * image_width)
        end_y = int(end_point.y * image_height)
        cv2.line(image, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)

def main():
    global last_mouse_x, last_mouse_y, current_volume, screenshot_taken, click_enabled, last_palm_y

    download_model()

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

    with HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        last_click_time = 0
        last_volume_time = 0
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("忽略空的摄像头帧")
                continue

            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            result = landmarker.detect(mp_image)

            if result.hand_landmarks:
                for landmarks in result.hand_landmarks:
                    draw_landmarks(image, landmarks, connections)

                    gesture = detect_gesture(landmarks)
                    
                    if gesture == '握拳':
                        screen_x, screen_y = map_hand_to_screen(landmarks, CAMERA_WIDTH, CAMERA_HEIGHT)
                        
                        last_mouse_x = last_mouse_x * (1 - SMOOTHING_FACTOR) + screen_x * SMOOTHING_FACTOR
                        last_mouse_y = last_mouse_y * (1 - SMOOTHING_FACTOR) + screen_y * SMOOTHING_FACTOR
                        
                        pyautogui.moveTo(last_mouse_x, last_mouse_y, duration=0.01)

                    elif gesture == '食指' and click_enabled:
                        current_time = time.time()
                        if current_time - last_click_time > 0.5:
                            pyautogui.click()
                            last_click_time = current_time

                    elif gesture == '两指':
                        current_time = time.time()
                        if current_time - last_click_time > 0.5:
                            pyautogui.click(button='right')
                            last_click_time = current_time

                    elif gesture == '三指':
                        current_time = time.time()
                        if current_time - last_volume_time > 0.3:
                            current_volume = min(100, current_volume + VOLUME_STEP)
                            pyautogui.press('volumeup')
                            last_volume_time = current_time

                    elif gesture == '小指':
                        current_time = time.time()
                        if current_time - last_volume_time > 0.3:
                            current_volume = max(0, current_volume - VOLUME_STEP)
                            pyautogui.press('volumedown')
                            last_volume_time = current_time

                    elif gesture == '拇指':
                        if not screenshot_taken:
                            screenshot = pyautogui.screenshot()
                            screenshot.save(f'screenshot_{int(time.time())}.png')
                            screenshot_taken = True
                            print("截图已保存")

                    elif gesture == '手掌':
                        screenshot_taken = False
                        wrist_y = landmarks[0].y
                        if last_palm_y != 0:
                            y_diff = wrist_y - last_palm_y
                            if abs(y_diff) > 0.02:
                                scroll_amount = int(y_diff * 100)
                                pyautogui.scroll(scroll_amount)
                        last_palm_y = wrist_y

                    image = draw_chinese_text(image, f'手势: {gesture}', (10, 30), (0, 255, 0))

            image = draw_chinese_text(image, f'音量: {current_volume}%', (10, 60), (255, 0, 0))
            image = draw_chinese_text(image, 'Q-退出', (10, CAMERA_HEIGHT - 40), (0, 0, 255), 20)
            image = draw_chinese_text(image, 'C-切换点击', (150, CAMERA_HEIGHT - 40), (0, 0, 255), 20)
            image = draw_chinese_text(image, f'点击: {"开启" if click_enabled else "关闭"}', (300, CAMERA_HEIGHT - 40), (0, 255, 255), 20)

            cv2.imshow('手势识别电脑控制程序', image)

            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                click_enabled = not click_enabled
                print(f'点击功能 {"开启" if click_enabled else "关闭"}')

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()