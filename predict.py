"""
predict.py  —  ASL Real-Time Predictor  (lag-free, competition UI)

FIX: Inference runs in a background thread so camera NEVER lags.
     The ensemble (RF+SVM+KNN) predict_proba was blocking the frame loop.

Controls
  R    reset buffer
  ESC  quit
"""

import cv2, pickle, collections, time, threading
import mediapipe as mp
import numpy as np
from features import extract

MODEL_PATH    = "model.pkl"
SMOOTH_BUFFER = 12
STABLE_FRAMES = 18
MIN_CONF      = 0.40
TOP_N         = 3

HARD = {"M","N","R","U","G","H","D","E","F","K","P","A","S","T"}

model   = pickle.load(open(MODEL_PATH, "rb"))
classes = list(model.classes_)
print(f"[OK] Model loaded — {len(classes)} classes")

mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.65,
    model_complexity=0          # ← complexity=0 is much faster, still accurate
)
mp_draw = mp.solutions.drawing_utils

# ── Colors (BGR) ──────────────────────────────────────────────────
BG        = (8,   8,  12)
PANEL     = (18,  18,  26)
CARD      = (26,  26,  38)
TEAL      = (180, 220,  80)    # main accent (BGR: greenish-teal)
GOLD      = ( 40, 190, 230)    # gold (BGR)
LAVENDER  = (200, 140, 180)
WHITE     = (245, 245, 250)
GRAY      = (100, 100, 115)
DIMGRAY   = ( 50,  50,  60)
RED_WARN  = ( 60,  60, 210)
FONT      = cv2.FONT_HERSHEY_SIMPLEX

# ─────────────────────────────────────────────
#  THREADED INFERENCE — eliminates camera lag
# ─────────────────────────────────────────────
_infer_lock     = threading.Lock()
_pending_flat   = None          # latest feature vector to classify
_result_pred    = ""
_result_probs   = []
_infer_running  = False

def _inference_worker():
    global _pending_flat, _result_pred, _result_probs, _infer_running
    while True:
        with _infer_lock:
            flat = _pending_flat
            _pending_flat = None
        if flat is None:
            time.sleep(0.001)
            continue
        try:
            probs    = model.predict_proba([flat])[0]
            top_idx  = np.argsort(probs)[::-1]
            pred     = classes[top_idx[0]]
            top_list = [(classes[i], float(probs[i])) for i in top_idx[:TOP_N]]
            with _infer_lock:
                _result_pred  = pred
                _result_probs = top_list
        except Exception:
            pass

_thread = threading.Thread(target=_inference_worker, daemon=True)
_thread.start()

def submit_inference(flat):
    global _pending_flat
    with _infer_lock:
        _pending_flat = flat   # always overwrite — only latest matters

def get_result():
    with _infer_lock:
        return _result_pred, list(_result_probs)

# ─────────────────────────────────────────────
#  UI HELPERS
# ─────────────────────────────────────────────
def filled_rect(img, x1, y1, x2, y2, color, alpha=1.0):
    if alpha >= 1.0:
        cv2.rectangle(img, (x1,y1), (x2,y2), color, -1)
    else:
        sub = img[y1:y2, x1:x2]
        overlay = sub.copy()
        overlay[:] = color
        cv2.addWeighted(overlay, alpha, sub, 1-alpha, 0, sub)

def corner_rect(img, x1, y1, x2, y2, color, t=2, sz=18):
    """Draw only corner brackets instead of full rectangle — sleek look."""
    pts = [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]
    for cx,cy,dx,dy in pts:
        cv2.line(img,(cx,cy),(cx+dx*sz,cy),color,t)
        cv2.line(img,(cx,cy),(cx,cy+dy*sz),color,t)

def hbar(img, x, y, w, h, val, color, bg=(40,40,55)):
    cv2.rectangle(img,(x,y),(x+w,y+h), bg,-1)
    filled = int(w * max(0.0, min(float(val),1.0)))
    if filled:
        cv2.rectangle(img,(x,y),(x+filled,y+h),color,-1)
    # subtle end cap
    cv2.rectangle(img,(x,y),(x+w,y+h),(60,60,80),1)

def glow_text(img, text, x, y, font, scale, color, thick, glow_color=None, glow_thick=None):
    """Draw text with a soft glow behind it."""
    if glow_color:
        cv2.putText(img, text, (x,y), font, scale, glow_color, glow_thick or thick+4)
    cv2.putText(img, text, (x,y), font, scale, color, thick)

# ─────────────────────────────────────────────
#  SIDEBAR DRAW
# ─────────────────────────────────────────────
SW = 260   # sidebar width

def draw_sidebar(canvas, top_preds, stable_prog, confirmed,
                 smooth_pred, confidence, hand_ok, fps, fh, fw):
    x0 = fw

    # sidebar background — dark panel with subtle gradient feel
    filled_rect(canvas, x0, 0, x0+SW, fh, PANEL)
    # thin accent border on left edge
    cv2.line(canvas,(x0,0),(x0,fh), TEAL, 1)

    # ── header ───────────────────────────────
    filled_rect(canvas, x0, 0, x0+SW, 44, CARD)
    cv2.putText(canvas, "ASL", (x0+10, 28), FONT, 0.65, TEAL, 1)
    cv2.putText(canvas, "PREDICTOR", (x0+52, 28), FONT, 0.65, WHITE, 1)
    cv2.putText(canvas, f"FPS {fps:02.0f}", (x0+SW-58, 28), FONT, 0.38, GRAY, 1)

    # ── big confirmed letter ──────────────────
    filled_rect(canvas, x0+10, 52, x0+SW-10, 160, CARD)
    corner_rect(canvas, x0+10, 52, x0+SW-10, 160, TEAL, t=1, sz=14)

    is_hard  = confirmed in HARD
    lcolor   = GOLD if is_hard else TEAL
    glow_col = (int(lcolor[0]*0.3), int(lcolor[1]*0.3), int(lcolor[2]*0.3))

    disp = confirmed if confirmed else "·"
    # center the letter
    (tw, th), _ = cv2.getTextSize(disp, FONT, 3.2, 5)
    lx = x0 + 10 + (SW-20-tw)//2
    glow_text(canvas, disp, lx, 148, FONT, 3.2, lcolor, 5,
              glow_color=glow_col, glow_thick=12)

    label_txt = "TRICKY SIGN" if is_hard and confirmed else ("CONFIRMED" if confirmed else "WAITING")
    label_col = GOLD if is_hard and confirmed else (TEAL if confirmed else GRAY)
    cv2.putText(canvas, label_txt, (x0+14, 172), FONT, 0.36, label_col, 1)

    # ── live prediction (smaller) ────────────
    cv2.putText(canvas, "LIVE", (x0+10, 196), FONT, 0.35, GRAY, 1)
    live_disp = smooth_pred if smooth_pred else "–"
    cv2.putText(canvas, live_disp, (x0+48, 196), FONT, 0.65, LAVENDER, 2)
    cv2.putText(canvas, f"{int(confidence*100)}%", (x0+SW-52, 196), FONT, 0.38, GRAY, 1)

    # ── stability bar ─────────────────────────
    cv2.putText(canvas, "STABILITY", (x0+10, 215), FONT, 0.34, GRAY, 1)
    bar_color = TEAL if stable_prog < 0.99 else (50, 240, 100)
    hbar(canvas, x0+10, 220, SW-20, 8, stable_prog, bar_color)

    # ── top-N predictions ─────────────────────
    cv2.line(canvas,(x0+10,238),(x0+SW-10,238),DIMGRAY,1)
    cv2.putText(canvas, "TOP PREDICTIONS", (x0+10, 256), FONT, 0.34, GRAY, 1)

    bar_colors = [TEAL, GOLD, LAVENDER]
    for i, (lbl, prob) in enumerate(top_preds[:TOP_N]):
        y      = 270 + i * 52
        bc     = GOLD if lbl in HARD else bar_colors[i]
        # row background
        filled_rect(canvas, x0+10, y, x0+SW-10, y+44, CARD, alpha=0.6)
        # letter
        glow_text(canvas, lbl, x0+18, y+32, FONT, 1.1, bc, 2,
                  glow_color=(int(bc[0]*0.2),int(bc[1]*0.2),int(bc[2]*0.2)),
                  glow_thick=6)
        # prob bar
        hbar(canvas, x0+46, y+14, SW-64, 10, prob, bc)
        cv2.putText(canvas, f"{int(prob*100):3d}%",
                    (x0+SW-40, y+24), FONT, 0.38, bc, 1)
        if lbl in HARD:
            cv2.putText(canvas, "tricky", (x0+46, y+38), FONT, 0.3, GOLD, 1)

    # ── hand status + controls ────────────────
    cv2.line(canvas,(x0+10, fh-70),(x0+SW-10, fh-70), DIMGRAY, 1)
    hc = (80,200,80) if hand_ok else (60,60,180)
    cv2.circle(canvas,(x0+20, fh-53), 6, hc, -1)
    cv2.putText(canvas,"HAND DETECTED" if hand_ok else "NO HAND",
                (x0+32, fh-48), FONT, 0.36, hc, 1)

    for i, t in enumerate(["[R] reset   [ESC] quit"]):
        cv2.putText(canvas, t, (x0+10, fh-24+i*18), FONT, 0.35, GRAY, 1)

# ─────────────────────────────────────────────
#  OVERLAY ON CAMERA FRAME
# ─────────────────────────────────────────────
def draw_camera_overlay(frame, stable_prog, confirmed, hand_ok):
    h, w = frame.shape[:2]

    # stability bar — top edge, glows green when full
    bar_color = (50, 240, 100) if stable_prog >= 1.0 else TEAL
    bw = int(w * stable_prog)
    cv2.rectangle(frame, (0,0), (bw, 5), bar_color, -1)
    cv2.rectangle(frame, (0,0), (w,  5), DIMGRAY,   1)

    # corner brackets on frame when confirmed
    if confirmed:
        corner_rect(frame, 8, 8, w-8, h-8,
                    GOLD if confirmed in HARD else TEAL, t=2, sz=22)

    # bottom-left: confirmed letter watermark on video
    if confirmed:
        alpha_layer = frame.copy()
        bc = GOLD if confirmed in HARD else TEAL
        (tw,_),_ = cv2.getTextSize(confirmed, FONT, 4.0, 6)
        cv2.putText(alpha_layer, confirmed, (w-tw-16, h-16),
                    FONT, 4.0,
                    (int(bc[0]*0.25),int(bc[1]*0.25),int(bc[2]*0.25)), 6)
        cv2.addWeighted(alpha_layer, 0.55, frame, 0.45, 0, frame)

# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
FH, FW = 480, 640

smooth_buf   = collections.deque(maxlen=SMOOTH_BUFFER)
stable_count = 0
prev_pred    = ""
confirmed    = ""
smooth_pred  = ""
confidence   = 0.0
top_preds    = [("–", 0.0)] * TOP_N
hand_ok      = False
fps_t = time.time(); fps_c = 0; fps_val = 0.0

print("[RUNNING]  predict.py — press ESC to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res   = hands.process(rgb)

    hand_ok = False

    if res.multi_hand_landmarks:
        for handLms in res.multi_hand_landmarks:
            # draw skeleton
            mp_draw.draw_landmarks(
                frame, handLms, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=TEAL,  thickness=2, circle_radius=4),
                mp_draw.DrawingSpec(color=WHITE, thickness=1))

            flat = extract([[lm.x, lm.y] for lm in handLms.landmark])
            submit_inference(flat)          # non-blocking — no lag!
            hand_ok = True
    else:
        if smooth_buf:
            smooth_buf.popleft()

    # pick up latest inference result
    raw_pred, top_preds_new = get_result()
    if top_preds_new:
        top_preds = top_preds_new

    if not top_preds or top_preds[0][1] < MIN_CONF:
        raw_pred = ""

    # smooth majority vote on latest predictions
    if raw_pred:
        smooth_buf.append(raw_pred)
    if smooth_buf:
        smooth_pred = collections.Counter(smooth_buf).most_common(1)[0][0]
        confidence  = smooth_buf.count(smooth_pred) / len(smooth_buf)
    else:
        smooth_pred = ""; confidence = 0.0

    # stability counter
    if smooth_pred and smooth_pred == prev_pred:
        stable_count = min(stable_count + 1, STABLE_FRAMES)
    else:
        stable_count = max(stable_count - 2, 0)
    prev_pred    = smooth_pred
    stable_prog  = stable_count / STABLE_FRAMES

    if stable_count >= STABLE_FRAMES and smooth_pred:
        confirmed = smooth_pred
    elif not hand_ok:
        confirmed = ""

    # FPS counter
    fps_c += 1
    if time.time() - fps_t >= 1.0:
        fps_val = fps_c; fps_c = 0; fps_t = time.time()

    # ── build canvas ──────────────────────────────────────────────
    draw_camera_overlay(frame, stable_prog, confirmed, hand_ok)

    canvas = np.zeros((FH, FW + SW, 3), dtype=np.uint8)
    canvas[:] = BG
    canvas[:FH, :FW] = frame

    draw_sidebar(canvas, top_preds, stable_prog, confirmed,
                 smooth_pred, confidence, hand_ok, fps_val, FH, FW)

    cv2.imshow("ASL Predictor", canvas)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('r'):
        smooth_buf.clear()
        stable_count = 0; confirmed = ""; prev_pred = ""
        print("[RESET] Buffer cleared")

cap.release()
cv2.destroyAllWindows()
print("[DONE]")