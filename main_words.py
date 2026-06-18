"""
main_words.py  -  ASL Word Translator (Windows lag-fixed)
FIXES: subprocess TTS, frame-skip, capped FPS, buffer flush
"""

import cv2
import mediapipe as mp
import pickle
import time
import numpy as np
import math
import subprocess

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
MODEL_PATH    = "word_model.pkl"
STABLE_FRAMES = 20
SMOOTH_BUFFER = 15
WORD_DELAY    = 2.0
PROCESS_EVERY = 3       # run MediaPipe every 3rd frame
DISPLAY_FPS   = 20      # cap UI redraw

# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
model     = pickle.load(open(MODEL_PATH, "rb"))
ALL_WORDS = list(model.classes_)
print(f"[OK] Model loaded — {len(ALL_WORDS)} words: {ALL_WORDS}")

# ─────────────────────────────────────────────
#  MEDIAPIPE
# ─────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

# ─────────────────────────────────────────────
#  TTS  — subprocess approach (no freeze on Windows)
# ─────────────────────────────────────────────
_tts_proc = None

def speak(text):
    global _tts_proc
    # kill previous speech if still running
    if _tts_proc and _tts_proc.poll() is None:
        _tts_proc.terminate()
    safe = text.replace("'", "")   # avoid quote issues in -c string
    script = (
        f"import pyttsx3; e=pyttsx3.init(); "
        f"e.setProperty('rate',145); "
        f"e.setProperty('volume',1.0); "
        f"e.say('{safe}'); e.runAndWait()"
    )
    _tts_proc = subprocess.Popen(
        ["python", "-c", script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

# ─────────────────────────────────────────────
#  FEATURE EXTRACTION  (89 features)
# ─────────────────────────────────────────────
WRIST      = 0
THUMB_TIP  = 4
INDEX_TIP  = 8
MIDDLE_TIP = 12
RING_TIP   = 16
PINKY_TIP  = 20
INDEX_MCP  = 5
MIDDLE_MCP = 9
RING_MCP   = 13
PINKY_MCP  = 17

FINGER_JOINTS = [
    (1,  2,  3,  4),
    (5,  6,  7,  8),
    (9,  10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
]

def _dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def _angle_3pts(a, b, c):
    v1 = (a[0]-b[0], a[1]-b[1])
    v2 = (c[0]-b[0], c[1]-b[1])
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag = math.sqrt(v1[0]**2+v1[1]**2) * math.sqrt(v2[0]**2+v2[1]**2)
    if mag == 0:
        return 0.0
    return math.acos(max(-1.0, min(1.0, dot/mag)))

def extract(lm_list):
    base_x, base_y = lm_list[0]
    coords = []
    for x, y in lm_list:
        coords.append(x - base_x)
        coords.append(y - base_y)
    max_val = max(abs(v) for v in coords) or 1.0
    coords  = [v / max_val for v in coords]

    curls = []
    for mcp, pip, dip, tip in FINGER_JOINTS:
        a_mcp = _angle_3pts(lm_list[WRIST], lm_list[mcp], lm_list[pip])
        a_pip = _angle_3pts(lm_list[mcp],   lm_list[pip], lm_list[dip])
        a_dip = _angle_3pts(lm_list[pip],   lm_list[dip], lm_list[tip])
        curl_ratio = lm_list[tip][1] - lm_list[mcp][1]
        curls.extend([a_mcp, a_pip, a_dip, curl_ratio])

    tips      = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    hand_size = _dist(lm_list[WRIST], lm_list[MIDDLE_MCP]) or 1.0
    tip_dists = []
    for i in range(len(tips)):
        for j in range(i+1, len(tips)):
            tip_dists.append(_dist(lm_list[tips[i]], lm_list[tips[j]]) / hand_size)

    tip_wrist = [_dist(lm_list[t], lm_list[WRIST]) / hand_size for t in tips]
    mcps      = [INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP, 1]
    mcp_wrist = [_dist(lm_list[m], lm_list[WRIST]) / hand_size for m in mcps]

    angles = [
        _angle_3pts(lm_list[INDEX_TIP],  lm_list[WRIST], lm_list[PINKY_TIP]),
        _angle_3pts(lm_list[THUMB_TIP],  lm_list[WRIST], lm_list[INDEX_TIP]),
        _angle_3pts(lm_list[THUMB_TIP],  lm_list[WRIST], lm_list[MIDDLE_TIP]),
        _angle_3pts(lm_list[THUMB_TIP],  lm_list[WRIST], lm_list[RING_TIP]),
        _angle_3pts(lm_list[THUMB_TIP],  lm_list[WRIST], lm_list[PINKY_TIP]),
        _angle_3pts(lm_list[INDEX_MCP],  lm_list[WRIST], lm_list[PINKY_MCP]),
        _angle_3pts(lm_list[MIDDLE_MCP], lm_list[WRIST], lm_list[RING_MCP]),
    ]

    return coords + curls + tip_dists + tip_wrist + mcp_wrist + angles

# ─────────────────────────────────────────────
#  COLORS & FONT
# ─────────────────────────────────────────────
C_BG     = (15, 15, 15)
C_PANEL  = (30, 30, 30)
C_GREEN  = (50, 220, 80)
C_ACCENT = (0, 200, 180)
C_YELLOW = (30, 210, 230)
C_WHITE  = (240, 240, 240)
C_GRAY   = (120, 120, 120)
FONT     = cv2.FONT_HERSHEY_SIMPLEX

def draw_rounded_rect(img, x1, y1, x2, y2, color, radius=10, thickness=-1):
    cv2.rectangle(img, (x1+radius, y1), (x2-radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1+radius), (x2, y2-radius), color, thickness)
    for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                   (x1+radius, y2-radius), (x2-radius, y2-radius)]:
        cv2.circle(img, (cx, cy), radius, color, thickness)

# ─────────────────────────────────────────────
#  DRAW UI
# ─────────────────────────────────────────────
def draw_ui(frame, detected_word, stable_progress, confidence,
            sentence, status_msg, top3):
    h, w    = frame.shape[:2]
    panel_h = 260
    canvas  = np.zeros((h + panel_h, w, 3), dtype=np.uint8)
    canvas[:] = C_BG
    canvas[:h, :w] = frame

    bar_w = int(w * stable_progress)
    cv2.rectangle(canvas, (0, 0), (bar_w, 7), C_ACCENT, -1)
    if stable_progress >= 1.0:
        cv2.rectangle(canvas, (0, 0), (w, 7), C_GREEN, -1)

    draw_rounded_rect(canvas, 6, h+6, w-6, h+panel_h-6, C_PANEL, radius=14)

    draw_rounded_rect(canvas, 16, h+14, w//2-10, h+130, (40,40,40), radius=10)
    cv2.putText(canvas, "DETECTED", (26, h+34), FONT, 0.42, C_GRAY, 1)
    word_display = detected_word if detected_word else "..."
    fs = 1.8 if len(word_display) <= 8 else (1.2 if len(word_display) <= 12 else 0.9)
    cv2.putText(canvas, word_display, (26, h+105), FONT, fs, C_GREEN, 3)
    cv2.rectangle(canvas, (16, h+116), (w//2-10, h+128), (55,55,55), -1)
    cv2.rectangle(canvas, (16, h+116),
                  (int(16 + (w//2-26) * confidence), h+128), C_ACCENT, -1)
    cv2.putText(canvas, f"{int(confidence*100)}%",
                (w//2-46, h+126), FONT, 0.38, C_GRAY, 1)

    draw_rounded_rect(canvas, w//2+4, h+14, w-16, h+130, (40,40,40), radius=10)
    cv2.putText(canvas, "TOP PREDICTIONS", (w//2+14, h+34), FONT, 0.42, C_GRAY, 1)
    colors_top = [C_GREEN, C_YELLOW, C_GRAY]
    for i, (word, prob) in enumerate(top3[:3]):
        y_pos   = h + 60 + i * 24
        bar_len = int((w//2 - 40) * prob)
        cv2.rectangle(canvas, (w//2+14, y_pos-12),
                      (w//2+14+bar_len, y_pos+2), (50,70,60), -1)
        cv2.putText(canvas, f"{word[:14]:<14} {prob*100:4.0f}%",
                    (w//2+18, y_pos), FONT, 0.42, colors_top[i], 1)

    draw_rounded_rect(canvas, 16, h+136, w-16, h+190, (40,40,40), radius=10)
    cv2.putText(canvas, "SENTENCE", (26, h+152), FONT, 0.42, C_GRAY, 1)
    sent_display = sentence[-52:] if sentence else "( sign words to build a sentence )"
    cv2.putText(canvas, sent_display, (26, h+178), FONT, 0.65, C_WHITE, 2)

    cv2.putText(canvas,
                "[SPACE] add word   [ENTER] speak   [BKSP] undo   [C] clear   [ESC] quit",
                (16, h+210), FONT, 0.38, C_GRAY, 1)
    cv2.putText(canvas, status_msg, (16, h+235), FONT, 0.52, C_ACCENT, 1)

    return canvas

# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS,          20)   # tell camera to capture 20fps
cap.set(cv2.CAP_PROP_BUFFERSIZE,    1)   # flush stale frames

smooth_buf    = []
stable_count  = 0
prev_pred     = ""
detected_word = ""
confidence    = 0.0
top3          = []
sentence      = ""
last_added    = ""
last_add_time = 0.0
status_msg    = "Ready  —  Show an ASL sign"
frame_count   = 0
last_display  = time.time()
display_interval = 1.0 / DISPLAY_FPS
canvas        = None

print("\n[RUNNING] ASL Word Translator — press ESC to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_count += 1
    raw_pred = ""

    # ── only run MediaPipe every Nth frame ──────────────────────
    if frame_count % PROCESS_EVERY == 0:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            for handLms in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, handLms, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0,200,180), thickness=2, circle_radius=3),
                    mp_draw.DrawingSpec(color=(200,200,200), thickness=1))

                lm_list = [[lm.x, lm.y] for lm in handLms.landmark]
                flat    = extract(lm_list)

                p = model.predict([flat])[0]
                smooth_buf.append(p)
                if len(smooth_buf) > SMOOTH_BUFFER:
                    smooth_buf.pop(0)
                raw_pred   = max(set(smooth_buf), key=smooth_buf.count)
                confidence = smooth_buf.count(raw_pred) / len(smooth_buf)

                proba   = model.predict_proba([flat])[0]
                top_idx = np.argsort(proba)[::-1][:3]
                top3    = [(ALL_WORDS[i], proba[i]) for i in top_idx]
        else:
            if smooth_buf:
                smooth_buf.pop(0)

    # ── stability check ─────────────────────────────────────────
    if raw_pred and raw_pred == prev_pred:
        stable_count = min(stable_count + 1, STABLE_FRAMES)
    else:
        stable_count = 0
    prev_pred       = raw_pred
    stable_progress = stable_count / STABLE_FRAMES

    if raw_pred:
        detected_word = raw_pred

    # ── cap display FPS ─────────────────────────────────────────
    now = time.time()
    if now - last_display >= display_interval:
        canvas = draw_ui(frame, detected_word, stable_progress,
                         confidence, sentence, status_msg, top3)
        cv2.imshow("ASL Word Translator", canvas)
        last_display = now

    # ── key handling ────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        if detected_word:
            sentence      += detected_word + " "
            last_added     = detected_word
            last_add_time  = time.time()
            status_msg     = f"Added: '{detected_word}'"
            speak(detected_word)

    elif key == 13:
        full = sentence.strip()
        if full:
            status_msg = f"Speaking: '{full}'"
            speak(full)
        else:
            status_msg = "Nothing to speak yet."

    elif key == 8:
        words = sentence.strip().split()
        if words:
            removed   = words[-1]
            sentence  = " ".join(words[:-1]) + (" " if len(words) > 1 else "")
            status_msg = f"Removed: '{removed}'"
        else:
            status_msg = "Nothing to remove."

    elif key == ord('c'):
        sentence      = ""
        detected_word = ""
        last_added    = ""
        smooth_buf    = []
        status_msg    = "Cleared"

    elif key == 27:
        break

# ── cleanup ─────────────────────────────────────────────────────
if _tts_proc and _tts_proc.poll() is None:
    _tts_proc.terminate()
cap.release()
cv2.destroyAllWindows()
print("\n[DONE] Translator closed.")