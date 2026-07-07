import math

class MockLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def create_fist_landmarks(x=0.5, y=0.5):
    wrist = MockLandmark(x, y)
    
    thumb_base = MockLandmark(x - 0.08, y - 0.02)
    thumb_ip = MockLandmark(x - 0.12, y + 0.02)
    thumb_tip = MockLandmark(x - 0.10, y + 0.08)
    
    index_base = MockLandmark(x - 0.02, y - 0.05)
    index_pip = MockLandmark(x - 0.02, y + 0.02)
    index_dip = MockLandmark(x - 0.02, y + 0.05)
    index_tip = MockLandmark(x - 0.02, y + 0.08)
    
    middle_base = MockLandmark(x + 0.02, y - 0.05)
    middle_pip = MockLandmark(x + 0.02, y + 0.02)
    middle_dip = MockLandmark(x + 0.02, y + 0.05)
    middle_tip = MockLandmark(x + 0.02, y + 0.08)
    
    ring_base = MockLandmark(x + 0.06, y - 0.03)
    ring_pip = MockLandmark(x + 0.06, y + 0.02)
    ring_dip = MockLandmark(x + 0.06, y + 0.05)
    ring_tip = MockLandmark(x + 0.06, y + 0.08)
    
    pinky_base = MockLandmark(x + 0.10, y - 0.02)
    pinky_pip = MockLandmark(x + 0.10, y + 0.02)
    pinky_dip = MockLandmark(x + 0.10, y + 0.05)
    pinky_tip = MockLandmark(x + 0.10, y + 0.08)
    
    return [
        wrist,
        thumb_base, thumb_ip, thumb_ip, thumb_tip,
        index_base, index_pip, index_dip, index_tip,
        middle_base, middle_pip, middle_dip, middle_tip,
        ring_base, ring_pip, ring_dip, ring_tip,
        pinky_base, pinky_pip, pinky_dip, pinky_tip
    ]

def create_index_finger_landmarks(x=0.5, y=0.5):
    wrist = MockLandmark(x, y)
    
    thumb_base = MockLandmark(x - 0.08, y - 0.02)
    thumb_ip = MockLandmark(x - 0.12, y + 0.02)
    thumb_tip = MockLandmark(x - 0.10, y + 0.08)
    
    index_base = MockLandmark(x - 0.02, y - 0.05)
    index_pip = MockLandmark(x - 0.02, y - 0.12)
    index_dip = MockLandmark(x - 0.02, y - 0.18)
    index_tip = MockLandmark(x - 0.02, y - 0.25)
    
    middle_base = MockLandmark(x + 0.02, y - 0.05)
    middle_pip = MockLandmark(x + 0.02, y + 0.02)
    middle_dip = MockLandmark(x + 0.02, y + 0.05)
    middle_tip = MockLandmark(x + 0.02, y + 0.08)
    
    ring_base = MockLandmark(x + 0.06, y - 0.03)
    ring_pip = MockLandmark(x + 0.06, y + 0.02)
    ring_dip = MockLandmark(x + 0.06, y + 0.05)
    ring_tip = MockLandmark(x + 0.06, y + 0.08)
    
    pinky_base = MockLandmark(x + 0.10, y - 0.02)
    pinky_pip = MockLandmark(x + 0.10, y + 0.02)
    pinky_dip = MockLandmark(x + 0.10, y + 0.05)
    pinky_tip = MockLandmark(x + 0.10, y + 0.08)
    
    return [
        wrist,
        thumb_base, thumb_ip, thumb_ip, thumb_tip,
        index_base, index_pip, index_dip, index_tip,
        middle_base, middle_pip, middle_dip, middle_tip,
        ring_base, ring_pip, ring_dip, ring_tip,
        pinky_base, pinky_pip, pinky_dip, pinky_tip
    ]

def create_two_fingers_landmarks(x=0.5, y=0.5):
    wrist = MockLandmark(x, y)
    
    thumb_base = MockLandmark(x - 0.08, y - 0.02)
    thumb_ip = MockLandmark(x - 0.12, y + 0.02)
    thumb_tip = MockLandmark(x - 0.10, y + 0.08)
    
    index_base = MockLandmark(x - 0.02, y - 0.05)
    index_pip = MockLandmark(x - 0.02, y - 0.12)
    index_dip = MockLandmark(x - 0.02, y - 0.18)
    index_tip = MockLandmark(x - 0.02, y - 0.25)
    
    middle_base = MockLandmark(x + 0.02, y - 0.05)
    middle_pip = MockLandmark(x + 0.02, y - 0.12)
    middle_dip = MockLandmark(x + 0.02, y - 0.18)
    middle_tip = MockLandmark(x + 0.02, y - 0.25)
    
    ring_base = MockLandmark(x + 0.06, y - 0.03)
    ring_pip = MockLandmark(x + 0.06, y + 0.02)
    ring_dip = MockLandmark(x + 0.06, y + 0.05)
    ring_tip = MockLandmark(x + 0.06, y + 0.08)
    
    pinky_base = MockLandmark(x + 0.10, y - 0.02)
    pinky_pip = MockLandmark(x + 0.10, y + 0.02)
    pinky_dip = MockLandmark(x + 0.10, y + 0.05)
    pinky_tip = MockLandmark(x + 0.10, y + 0.08)
    
    return [
        wrist,
        thumb_base, thumb_ip, thumb_ip, thumb_tip,
        index_base, index_pip, index_dip, index_tip,
        middle_base, middle_pip, middle_dip, middle_tip,
        ring_base, ring_pip, ring_dip, ring_tip,
        pinky_base, pinky_pip, pinky_dip, pinky_tip
    ]

def create_three_fingers_landmarks(x=0.5, y=0.5):
    wrist = MockLandmark(x, y)
    
    thumb_base = MockLandmark(x - 0.08, y - 0.02)
    thumb_ip = MockLandmark(x - 0.12, y + 0.02)
    thumb_tip = MockLandmark(x - 0.10, y + 0.08)
    
    index_base = MockLandmark(x - 0.02, y - 0.05)
    index_pip = MockLandmark(x - 0.02, y - 0.12)
    index_dip = MockLandmark(x - 0.02, y - 0.18)
    index_tip = MockLandmark(x - 0.02, y - 0.25)
    
    middle_base = MockLandmark(x + 0.02, y - 0.05)
    middle_pip = MockLandmark(x + 0.02, y - 0.12)
    middle_dip = MockLandmark(x + 0.02, y - 0.18)
    middle_tip = MockLandmark(x + 0.02, y - 0.25)
    
    ring_base = MockLandmark(x + 0.06, y - 0.03)
    ring_pip = MockLandmark(x + 0.06, y - 0.10)
    ring_dip = MockLandmark(x + 0.06, y - 0.16)
    ring_tip = MockLandmark(x + 0.06, y - 0.22)
    
    pinky_base = MockLandmark(x + 0.10, y - 0.02)
    pinky_pip = MockLandmark(x + 0.10, y + 0.02)
    pinky_dip = MockLandmark(x + 0.10, y + 0.05)
    pinky_tip = MockLandmark(x + 0.10, y + 0.08)
    
    return [
        wrist,
        thumb_base, thumb_ip, thumb_ip, thumb_tip,
        index_base, index_pip, index_dip, index_tip,
        middle_base, middle_pip, middle_dip, middle_tip,
        ring_base, ring_pip, ring_dip, ring_tip,
        pinky_base, pinky_pip, pinky_dip, pinky_tip
    ]

def create_pinky_landmarks(x=0.5, y=0.5):
    wrist = MockLandmark(x, y)
    
    thumb_base = MockLandmark(x - 0.08, y - 0.02)
    thumb_ip = MockLandmark(x - 0.12, y + 0.02)
    thumb_tip = MockLandmark(x - 0.10, y + 0.08)
    
    index_base = MockLandmark(x - 0.02, y - 0.05)
    index_pip = MockLandmark(x - 0.02, y + 0.02)
    index_dip = MockLandmark(x - 0.02, y + 0.05)
    index_tip = MockLandmark(x - 0.02, y + 0.08)
    
    middle_base = MockLandmark(x + 0.02, y - 0.05)
    middle_pip = MockLandmark(x + 0.02, y + 0.02)
    middle_dip = MockLandmark(x + 0.02, y + 0.05)
    middle_tip = MockLandmark(x + 0.02, y + 0.08)
    
    ring_base = MockLandmark(x + 0.06, y - 0.03)
    ring_pip = MockLandmark(x + 0.06, y + 0.02)
    ring_dip = MockLandmark(x + 0.06, y + 0.05)
    ring_tip = MockLandmark(x + 0.06, y + 0.08)
    
    pinky_base = MockLandmark(x + 0.10, y - 0.02)
    pinky_pip = MockLandmark(x + 0.10, y - 0.10)
    pinky_dip = MockLandmark(x + 0.10, y - 0.16)
    pinky_tip = MockLandmark(x + 0.10, y - 0.22)
    
    return [
        wrist,
        thumb_base, thumb_ip, thumb_ip, thumb_tip,
        index_base, index_pip, index_dip, index_tip,
        middle_base, middle_pip, middle_dip, middle_tip,
        ring_base, ring_pip, ring_dip, ring_tip,
        pinky_base, pinky_pip, pinky_dip, pinky_tip
    ]

def create_thumb_landmarks(x=0.5, y=0.5):
    wrist = MockLandmark(x, y)
    
    thumb_base = MockLandmark(x - 0.08, y - 0.02)
    thumb_ip = MockLandmark(x - 0.18, y - 0.05)
    thumb_tip = MockLandmark(x - 0.25, y - 0.08)
    
    index_base = MockLandmark(x - 0.02, y - 0.05)
    index_pip = MockLandmark(x - 0.02, y + 0.02)
    index_dip = MockLandmark(x - 0.02, y + 0.05)
    index_tip = MockLandmark(x - 0.02, y + 0.08)
    
    middle_base = MockLandmark(x + 0.02, y - 0.05)
    middle_pip = MockLandmark(x + 0.02, y + 0.02)
    middle_dip = MockLandmark(x + 0.02, y + 0.05)
    middle_tip = MockLandmark(x + 0.02, y + 0.08)
    
    ring_base = MockLandmark(x + 0.06, y - 0.03)
    ring_pip = MockLandmark(x + 0.06, y + 0.02)
    ring_dip = MockLandmark(x + 0.06, y + 0.05)
    ring_tip = MockLandmark(x + 0.06, y + 0.08)
    
    pinky_base = MockLandmark(x + 0.10, y - 0.02)
    pinky_pip = MockLandmark(x + 0.10, y + 0.02)
    pinky_dip = MockLandmark(x + 0.10, y + 0.05)
    pinky_tip = MockLandmark(x + 0.10, y + 0.08)
    
    return [
        wrist,
        thumb_base, thumb_ip, thumb_ip, thumb_tip,
        index_base, index_pip, index_dip, index_tip,
        middle_base, middle_pip, middle_dip, middle_tip,
        ring_base, ring_pip, ring_dip, ring_tip,
        pinky_base, pinky_pip, pinky_dip, pinky_tip
    ]

def create_open_palm_landmarks(x=0.5, y=0.5):
    wrist = MockLandmark(x, y)
    
    thumb_base = MockLandmark(x - 0.08, y - 0.02)
    thumb_ip = MockLandmark(x - 0.20, y - 0.08)
    thumb_tip = MockLandmark(x - 0.28, y - 0.12)
    
    index_base = MockLandmark(x - 0.02, y - 0.05)
    index_pip = MockLandmark(x - 0.04, y - 0.15)
    index_dip = MockLandmark(x - 0.03, y - 0.25)
    index_tip = MockLandmark(x - 0.02, y - 0.35)
    
    middle_base = MockLandmark(x + 0.02, y - 0.05)
    middle_pip = MockLandmark(x + 0.03, y - 0.15)
    middle_dip = MockLandmark(x + 0.04, y - 0.25)
    middle_tip = MockLandmark(x + 0.05, y - 0.35)
    
    ring_base = MockLandmark(x + 0.06, y - 0.03)
    ring_pip = MockLandmark(x + 0.08, y - 0.13)
    ring_dip = MockLandmark(x + 0.10, y - 0.23)
    ring_tip = MockLandmark(x + 0.11, y - 0.33)
    
    pinky_base = MockLandmark(x + 0.10, y - 0.02)
    pinky_pip = MockLandmark(x + 0.14, y - 0.11)
    pinky_dip = MockLandmark(x + 0.18, y - 0.20)
    pinky_tip = MockLandmark(x + 0.20, y - 0.28)
    
    return [
        wrist,
        thumb_base, thumb_ip, thumb_ip, thumb_tip,
        index_base, index_pip, index_dip, index_tip,
        middle_base, middle_pip, middle_dip, middle_tip,
        ring_base, ring_pip, ring_dip, ring_tip,
        pinky_base, pinky_pip, pinky_dip, pinky_tip
    ]

def test_gesture_detection():
    print("=" * 50)
    print("手势识别功能测试")
    print("=" * 50)
    
    from hand_gesture_control import HandGestureController
    
    controller = HandGestureController()
    
    test_cases = [
        ("握拳", create_fist_landmarks),
        ("食指", create_index_finger_landmarks),
        ("两指", create_two_fingers_landmarks),
        ("三指", create_three_fingers_landmarks),
        ("小指", create_pinky_landmarks),
        ("拇指", create_thumb_landmarks),
        ("手掌", create_open_palm_landmarks),
    ]
    
    for expected_gesture, create_func in test_cases:
        landmarks = create_func(0.5, 0.5)
        detected = controller._detect_gesture(landmarks)
        
        status = "✓" if detected == expected_gesture else "✗"
        print(f"{status} 预期: {expected_gesture:<4} | 实际: {detected:<4}")
    
    print("\n" + "=" * 50)
    print("掌心坐标计算测试")
    print("=" * 50)
    
    for gesture_name, create_func in test_cases:
        landmarks = create_func(0.3, 0.4)
        palm = controller._get_palm_center(landmarks)
        quadrant = controller._get_quadrant(landmarks)
        print(f"{gesture_name:<4}: 掌心=({palm['x']:.2f}, {palm['y']:.2f}), 象限={quadrant}")
    
    print("\n" + "=" * 50)
    print("坐标映射测试")
    print("=" * 50)
    
    test_positions = [
        (0.2, 0.2, "左上角"),
        (0.8, 0.2, "右上角"),
        (0.2, 0.8, "左下角"),
        (0.8, 0.8, "右下角"),
        (0.5, 0.5, "中心"),
    ]
    
    for x, y, desc in test_positions:
        landmarks = create_fist_landmarks(x, y)
        screen_x, screen_y = controller._map_hand_to_screen(landmarks, 640, 480)
        print(f"{desc:<6}: 输入=({x}, {y}) -> 屏幕=({screen_x:.0f}, {screen_y:.0f})")

if __name__ == '__main__':
    test_gesture_detection()