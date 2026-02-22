import cv2
import numpy as np
import mediapipe as mp
import subprocess

# ---------------- Mediapipe ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ---------------- Camera ----------------
cap = cv2.VideoCapture(0)
canvas = None
draw_mode = False

points = []  # track all points while drawing
alpha = 0.15
smooth_x, smooth_y = None, None

# One-time trigger flags
music_played = False
settings_opened = False

# ---------------- Main Loop ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        h, w, _ = frame.shape

        # Index finger tip
        ix = int(hand.landmark[8].x * w)
        iy = int(hand.landmark[8].y * h)

        # Thumb tip
        tx = int(hand.landmark[4].x * w)
        ty = int(hand.landmark[4].y * h)

        distance = np.hypot(ix - tx, iy - ty)

        # DRAW MODE (pinch)
        if distance < 35:
            draw_mode = True
        elif distance > 55:
            draw_mode = False
            smooth_x, smooth_y = None, None

            # When finished drawing, check shape
            if len(points) > 5:  # ignore tiny dots
                pts = np.array(points)
                x_range = pts[:, 0].max() - pts[:, 0].min()
                y_range = pts[:, 1].max() - pts[:, 1].min()

                # --------- Straight line detection ---------
                if x_range > y_range * 3 and not music_played:
                    music_played = True
                    print("Straight line detected → Apple Music!")
                    applescript = '''
                    tell application "Music"
                        activate
                        play (first playlist whose name contains "Pop")
                    end tell
                    '''
                    subprocess.Popen(["osascript", "-e", applescript])

                # --------- Square detection ---------
                elif abs(x_range - y_range) / max(x_range, y_range) < 0.2 and not settings_opened:
                    settings_opened = True
                    print("Square detected → System Settings!")
                    subprocess.Popen(["open", "/System/Applications/System Settings.app"])

            points = []  # reset points after checking

        if draw_mode:
            # Smooth the points
            if smooth_x is None:
                smooth_x, smooth_y = ix, iy
            else:
                smooth_x = int(alpha * ix + (1 - alpha) * smooth_x)
                smooth_y = int(alpha * iy + (1 - alpha) * smooth_y)

            points.append([smooth_x, smooth_y])

            # Draw on canvas
            if len(points) > 1:
                cv2.line(canvas, tuple(points[-2]), tuple(points[-1]), (255, 0, 255), 6, cv2.LINE_AA)

    # Merge frame + canvas
    frame = cv2.add(frame, canvas)

    cv2.putText(frame, "Pinch & draw: line=Music | square=Settings | C=clear | Q=quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Control", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros_like(frame)
        points = []
        music_played = False
        settings_opened = False

cap.release()
cv2.destroyAllWindows()
