import cv2
import numpy as np
import os
import json
import time
from collections import deque
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load label classes ─────────────────────────────────────────
with open(os.path.join(BASE_DIR, 'label_classes.json'), 'r') as f:
    labels = json.load(f)

NUM_CLASSES = len(labels)
print(f'Loaded {NUM_CLASSES} classes: {labels}')

# ── Build model ────────────────────────────────────────────────
isl_model = Sequential([
    Dense(256, activation='relu', input_shape=(126,)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(NUM_CLASSES, activation='softmax')
])

isl_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
isl_model.load_weights(os.path.join(BASE_DIR, 'isl_mediapipe.weights.h5'))
print('Model loaded!')

# ── MediaPipe setup ────────────────────────────────────────────
base_options = mp_python.BaseOptions(
    model_asset_path=os.path.join(BASE_DIR, 'hand_landmarker.task')
)
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,                          # ← 2 hands
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=mp_vision.RunningMode.IMAGE
)
detector = mp_vision.HandLandmarker.create_from_options(options)

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

# ── Smoothing buffer ───────────────────────────────────────────
SMOOTH_FRAMES = 20
pred_buffer = deque(maxlen=SMOOTH_FRAMES)

# ── Word/sentence recording ────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.50
HOLD_SECONDS = 0.8
NO_HAND_TIMEOUT = 2.0

current_letter = None
letter_hold_start = None
last_hand_time = time.time()

sentence = []
current_word = ""

# Output text file
output_file = os.path.join(BASE_DIR, 'output.txt')
with open(output_file, 'w') as f:
    f.write("=== ISL Session Started ===\n")

def save_to_file(text):
    with open(output_file, 'a') as f:
        f.write(text + '\n')
    print(text)

print('=== ISL Detection Started ===')
print('Controls: q=quit | b=reset buffer | c=clear sentence | SPACE=add space')
print(f'Output saved to: {output_file}')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    best_label = "..."
    best_confidence = 0.0
    top3_display = []

    if result.hand_landmarks:
        last_hand_time = time.time()

        # Draw skeleton for all detected hands
        for hand_lms in result.hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
            for a, b in CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
            for pt in pts:
                cv2.circle(frame, pt, 4, (255, 0, 0), -1)

        # ── Extract 2-hand coords (always 126 features) ───────
        coords = []
        for hand_idx in range(2):
            if hand_idx < len(result.hand_landmarks):
                lms = result.hand_landmarks[hand_idx]
                wrist = lms[0]
                hand_coords = []
                for lm in lms:
                    hand_coords.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

                # Scale normalize by middle finger base distance
                hand_coords = np.array(hand_coords)
                scale = np.sqrt(hand_coords[9*3]**2 + hand_coords[9*3+1]**2)
                if scale > 0:
                    hand_coords = hand_coords / scale
                coords.extend(hand_coords.tolist())
            else:
                coords.extend([0.0] * 63)  # pad missing hand with zeros

        coords = np.array(coords).reshape(1, 126)

        # Predict and add to buffer
        pred = isl_model.predict(coords, verbose=0)[0]
        pred_buffer.append(pred)

        # Smooth predictions
        avg_pred = np.mean(pred_buffer, axis=0)

        # Pick highest confidence among top 3
        top3_idx = np.argsort(avg_pred)[::-1][:3]
        top3_display = [(labels[i], avg_pred[i]) for i in top3_idx]

        best_idx = top3_idx[0]
        best_confidence = avg_pred[best_idx]
        best_label = labels[best_idx] if best_confidence >= CONFIDENCE_THRESHOLD else "..."

        # ── Letter hold logic ──────────────────────────────────
        if best_label != "...":
            if best_label == current_letter:
                held_time = time.time() - letter_hold_start
                progress = min(held_time / HOLD_SECONDS, 1.0)

                # Hold progress bar
                bar_x, bar_y = 30, h - 60
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 300, bar_y + 20), (50, 50, 50), -1)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(300 * progress), bar_y + 20), (0, 255, 255), -1)
                cv2.putText(frame, f"Hold: {best_label.upper()}", (bar_x, bar_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                if held_time >= HOLD_SECONDS:
                    current_word += best_label.upper()
                    sentence.append(best_label.upper())
                    save_to_file(f"Letter recorded: {best_label.upper()} | Word so far: {current_word}")
                    current_letter = None
                    letter_hold_start = None
            else:
                current_letter = best_label
                letter_hold_start = time.time()
        else:
            current_letter = None
            letter_hold_start = None

    else:
        pred_buffer.clear()
        current_letter = None
        letter_hold_start = None

        # Auto space after no hand timeout
        if time.time() - last_hand_time > NO_HAND_TIMEOUT and current_word:
            sentence.append(' ')
            save_to_file(f"Word completed: {current_word}")
            current_word = ""                          # ← fixed reset
            last_hand_time = time.time()

    # ── Build sentence ─────────────────────────────────────────
    full_sentence = ''.join(sentence)

    # ── Display ────────────────────────────────────────────────
    color = (0, 255, 0) if best_confidence >= CONFIDENCE_THRESHOLD else (0, 0, 255)
    cv2.putText(frame, f"{best_label.upper()}", (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 3)
    cv2.putText(frame, f"Conf: {best_confidence:.2f}", (30, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Top 3
    cv2.putText(frame, "Top 3:", (w - 220, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    for i, (lbl, conf) in enumerate(top3_display):
        col = (0, 255, 255) if i == 0 else (180, 180, 180)
        cv2.putText(frame, f"{lbl.upper()}: {conf:.2f}", (w - 220, 60 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)

    # Sentence bar
    cv2.rectangle(frame, (0, h - 100), (w, h), (30, 30, 30), -1)
    cv2.putText(frame, "Sentence:", (10, h - 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    display_sentence = full_sentence[-40:] if len(full_sentence) > 40 else full_sentence
    cv2.putText(frame, display_sentence, (10, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, "q=quit | c=clear | SPACE=space", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    cv2.imshow('ISL Detection', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('b'):
        pred_buffer.clear()
        print('Buffer reset!')
    elif key == ord('c'):
        sentence.clear()
        current_word = ""
        save_to_file("--- Sentence cleared ---")
        print('Sentence cleared!')
    elif key == ord(' '):
        sentence.append(' ')
        save_to_file(f"Word completed (manual): {current_word}")
        current_word = ""

# ── Save final sentence ────────────────────────────────────────
final = ''.join(sentence)
save_to_file("\n=== Session Ended ===")
save_to_file(f"Full sentence: {final}")
print(f'\nFull sentence: {final}')

detector.close()
cap.release()
cv2.destroyAllWindows()
