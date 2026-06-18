"""
main.py — SignLens ASL Translator
  GET /                    → index.html          (cover page)
  GET /translator          → asl_translator.html (translator UI)
  GET /asl_translator.html → asl_translator.html (direct fallback)
  GET /video               → MJPEG camera stream
  GET /ws                  → WebSocket upgrade

FPS optimisations active:
  model_complexity=0, 640×480, BUFFERSIZE=1, JPEG quality=72,
  async inference thread, adaptive frame-skip

Hand-tracking overlay: monochromatic lavender-purple palette
  (#A78BFA landmarks, #7C3AED connections) to match UI design.

FIX: A letter is added only once per hand appearance.
     The same letter can fire again only after the hand
     leaves the frame (or drops below confidence) and returns.
"""

import cv2
import mediapipe as mp
import pickle
import time
import threading
import collections
import numpy as np
import json
import base64
import hashlib
import struct
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from features import extract

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────
INDEX_PATH      = "index.html"           # cover / landing page
TRANSLATOR_PATH = "asl_translator.html"  # translator UI
HOST            = "localhost"
PORT            = 5000

STABLE_FRAMES   = 22
SMOOTH_BUFFER   = 20
MIN_CONF        = 0.40
JPEG_QUALITY    = 72
TARGET_FPS      = 60
INFER_EVERY     = 1      # raise to 2 if CPU struggles
CAP_W, CAP_H    = 640, 480

# ─────────────────────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────────────────────
model   = pickle.load(open("model.pkl", "rb"))
CLASSES = list(model.classes_)
print(f"[OK] Model — {len(CLASSES)} letters: {CLASSES}")

# ─────────────────────────────────────────────────────────────
#  MEDIAPIPE  (complexity=0 → ~3× faster than 1)
# ─────────────────────────────────────────────────────────────
mp_hands    = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils
mp_styles   = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.65,
    model_complexity=0,
)

# ── Monochromatic lavender-purple drawing specs ───────────────
_LANDMARK_SPEC = mp_drawing.DrawingSpec(
    color=(250, 139, 167),   # BGR for #A78BFA
    thickness=1,
    circle_radius=3,
)

_CONNECTION_SPEC = mp_drawing.DrawingSpec(
    color=(237, 58, 124),    # BGR for #7C3AED
    thickness=1,
    circle_radius=1,
)

# ─────────────────────────────────────────────────────────────
#  SHARED STATE
# ─────────────────────────────────────────────────────────────
state_lock = threading.Lock()
shared = {
    "frame_jpg":    b"",
    "live_letter":  "",
    "top3":         [["·", 0.0], ["·", 0.0], ["·", 0.0]],
    "stable_prog":  0.0,
    "hand_ok":      False,
    "fps":          0.0,
    "current_word": "",
    "sentence":     "",
    "status_msg":   "Show an ASL hand sign",
    "speak_text":   None,
}

ws_clients_lock = threading.Lock()
ws_clients      = []

# ─────────────────────────────────────────────────────────────
#  INFERENCE THREAD
# ─────────────────────────────────────────────────────────────
_inf_lock    = threading.Lock()
_inf_pending = None
_inf_pred    = ""
_inf_top3    = []

def _inference_worker():
    global _inf_pending, _inf_pred, _inf_top3
    while True:
        with _inf_lock:
            flat         = _inf_pending
            _inf_pending = None
        if flat is None:
            time.sleep(0.001)
            continue
        try:
            probs   = model.predict_proba([flat])[0]
            top_idx = np.argsort(probs)[::-1]
            top3    = [(CLASSES[i], float(probs[i])) for i in top_idx[:3]]
            with _inf_lock:
                _inf_pred = top3[0][0]
                _inf_top3 = top3
        except Exception:
            pass

threading.Thread(target=_inference_worker, daemon=True).start()

def submit_inference(flat):
    global _inf_pending
    with _inf_lock:
        _inf_pending = flat

def get_inference():
    with _inf_lock:
        return _inf_pred, list(_inf_top3)

# ─────────────────────────────────────────────────────────────
#  WEBSOCKET HELPERS
# ─────────────────────────────────────────────────────────────
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

def ws_handshake(conn, key: str):
    accept = base64.b64encode(
        hashlib.sha1((key + WS_GUID).encode()).digest()
    ).decode()
    conn.sendall((
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
    ).encode())

def ws_send(conn, data: str):
    payload = data.encode("utf-8")
    n       = len(payload)
    if n <= 125:
        header = bytes([0x81, n])
    elif n <= 65535:
        header = struct.pack(">BBH", 0x81, 126, n)
    else:
        header = struct.pack(">BBQ", 0x81, 127, n)
    try:
        conn.sendall(header + payload)
    except Exception:
        pass

def ws_recv(conn):
    try:
        h = _recvall(conn, 2)
        if not h:
            return None
        opcode = h[0] & 0x0F
        if opcode == 8:
            return None
        masked = (h[1] & 0x80) != 0
        n      = h[1] & 0x7F
        if n == 126:
            n = struct.unpack(">H", _recvall(conn, 2))[0]
        elif n == 127:
            n = struct.unpack(">Q", _recvall(conn, 8))[0]
        mask_key = _recvall(conn, 4) if masked else b""
        data     = bytearray(_recvall(conn, n))
        if masked:
            for i in range(n):
                data[i] ^= mask_key[i % 4]
        return data.decode("utf-8", errors="replace")
    except Exception:
        return None

def _recvall(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed")
        buf += chunk
    return buf

def ws_broadcast(obj: dict):
    msg = json.dumps(obj)
    with ws_clients_lock:
        dead = []
        for c in ws_clients:
            try:
                ws_send(c, msg)
            except Exception:
                dead.append(c)
        for c in dead:
            ws_clients.remove(c)

def handle_ws_client(conn):
    with ws_clients_lock:
        ws_clients.append(conn)
    print(f"[WS] connected  (total={len(ws_clients)})")
    try:
        while True:
            text = ws_recv(conn)
            if text is None:
                break
            try:
                _handle_browser_action(json.loads(text).get("action", ""))
            except Exception:
                pass
    finally:
        with ws_clients_lock:
            if conn in ws_clients:
                ws_clients.remove(conn)
        try:
            conn.close()
        except Exception:
            pass
        print(f"[WS] disconnected (total={len(ws_clients)})")

def _handle_browser_action(action: str):
    with state_lock:
        cw  = shared["current_word"]
        sen = shared["sentence"]

    if action == "confirm_word":
        if cw:
            with state_lock:
                shared["sentence"]    += cw + " "
                shared["status_msg"]   = f"Word added: '{cw}'"
                shared["current_word"] = ""
        else:
            with state_lock:
                shared["status_msg"] = "Nothing to confirm."

    elif action == "delete":
        if cw:
            with state_lock:
                shared["current_word"] = cw[:-1]
                shared["status_msg"]   = "Deleted last letter"
        elif sen:
            words   = sen.strip().split()
            removed = words[-1] if words else ""
            new_sen = (" ".join(words[:-1]) + " ") if len(words) > 1 else ""
            with state_lock:
                shared["sentence"]   = new_sen
                shared["status_msg"] = f"Removed: '{removed}'"

    elif action == "clear":
        with state_lock:
            shared["current_word"] = ""
            shared["sentence"]     = ""
            shared["status_msg"]   = "Cleared"

    elif action == "speak":
        with state_lock:
            full = (shared["sentence"] + shared["current_word"]).strip()
            shared["status_msg"] = f"Speaking: '{full}'" if full else "Nothing to speak."

# ─────────────────────────────────────────────────────────────
#  HTTP REQUEST HANDLER
# ─────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress per-request log noise

    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/":
            self._serve_html(INDEX_PATH)

        elif path in ("/translator", "/asl_translator.html"):
            self._serve_html(TRANSLATOR_PATH)

        elif path == "/video":
            self._serve_mjpeg()

        elif path == "/ws":
            self._upgrade_ws()

        else:
            self.send_error(404)

    def _serve_html(self, filepath):
        if not os.path.exists(filepath):
            self.send_error(404, f"File not found: {filepath}")
            return
        with open(filepath, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type",   "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control",  "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _serve_mjpeg(self):
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=--jpgboundary")
        self.send_header("Cache-Control",  "no-cache, no-store, must-revalidate")
        self.send_header("Pragma",         "no-cache")
        self.send_header("Expires",        "0")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try:
            while True:
                with state_lock:
                    jpg = shared["frame_jpg"]
                if jpg:
                    try:
                        self.wfile.write(
                            b"--jpgboundary\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n"
                            + jpg + b"\r\n"
                        )
                        self.wfile.flush()
                    except Exception:
                        break
                time.sleep(1 / 60)
        except Exception:
            pass

    def _upgrade_ws(self):
        key = self.headers.get("Sec-WebSocket-Key", "")
        if not key:
            self.send_error(400, "Missing Sec-WebSocket-Key")
            return
        ws_handshake(self.connection, key)
        t = threading.Thread(
            target=handle_ws_client, args=(self.connection,), daemon=True
        )
        t.start()
        t.join()


class ThreadingHTTPServer(HTTPServer):
    def process_request(self, request, client_address):
        threading.Thread(
            target=self._thread_target,
            args=(request, client_address),
            daemon=True,
        ).start()

    def _thread_target(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except Exception:
            pass
        finally:
            self.shutdown_request(request)

# ─────────────────────────────────────────────────────────────
#  BROADCAST LOOP  (30 Hz)
# ─────────────────────────────────────────────────────────────
def broadcast_loop():
    while True:
        with state_lock:
            payload = {
                "live_letter":  shared["live_letter"],
                "top3":         shared["top3"],
                "stable_prog":  shared["stable_prog"],
                "hand_ok":      shared["hand_ok"],
                "fps":          shared["fps"],
                "current_word": shared["current_word"],
                "sentence":     shared["sentence"],
                "status_msg":   shared["status_msg"],
                "speak_text":   shared.get("speak_text"),
            }
            shared["speak_text"] = None
        ws_broadcast(payload)
        time.sleep(1 / 30)

threading.Thread(target=broadcast_loop, daemon=True).start()

# ─────────────────────────────────────────────────────────────
#  CAMERA + INFERENCE LOOP
# ─────────────────────────────────────────────────────────────
def run_capture():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC,       cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
    cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[CAMERA] {actual_w}×{actual_h}")

    smooth_buf    = collections.deque(maxlen=SMOOTH_BUFFER)
    stable_count  = 0
    prev_pred     = ""
    last_added    = ""        # tracks the last letter that was committed
    frame_idx     = 0
    last_hand_ok  = False

    # ── How many consecutive no-hand frames before we reset last_added ──
    # ~0.25 s worth of frames at TARGET_FPS gives a natural "lift & reshoot"
    NO_HAND_RESET_FRAMES = max(8, TARGET_FPS // 4)
    no_hand_counter      = 0   # counts consecutive frames with no hand

    fps_t   = time.time()
    fps_c   = 0
    fps_val = 0.0

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame     = cv2.flip(frame, 1)
        frame_idx += 1

        # ── MediaPipe every INFER_EVERY frames ───────────────
        if frame_idx % INFER_EVERY == 0:
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res     = hands.process(rgb)
            hand_ok = False

            if res.multi_hand_landmarks:
                hand_ok         = True
                no_hand_counter = 0   # hand is present — reset absence counter

                for handLms in res.multi_hand_landmarks:
                    flat = extract([[lm.x, lm.y] for lm in handLms.landmark])
                    submit_inference(flat)

                    mp_drawing.draw_landmarks(
                        frame,
                        handLms,
                        mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=_LANDMARK_SPEC,
                        connection_drawing_spec=_CONNECTION_SPEC,
                    )
            else:
                # No hand detected this frame
                if smooth_buf:
                    smooth_buf.popleft()

                no_hand_counter += 1

                # After enough consecutive no-hand frames, allow the same
                # letter to be added again (user pulled hand away and returned)
                if no_hand_counter >= NO_HAND_RESET_FRAMES:
                    last_added = ""

            last_hand_ok = hand_ok
        else:
            hand_ok = last_hand_ok

        # ── Encode frame ──────────────────────────────────────
        ok, jpg_buf = cv2.imencode(".jpg", frame, encode_params)
        jpg_bytes   = jpg_buf.tobytes() if ok else b""

        # ── Inference result ──────────────────────────────────
        raw_pred, top3_new = get_inference()
        top3     = top3_new if top3_new else [("·", 0.0)] * 3
        conf_top = top3[0][1] if top3 else 0.0

        if conf_top < MIN_CONF or not hand_ok:
            raw_pred = ""

        if raw_pred:
            smooth_buf.append(raw_pred)

        if smooth_buf:
            ctr         = collections.Counter(smooth_buf)
            smooth_pred = ctr.most_common(1)[0][0]
            confidence  = ctr[smooth_pred] / len(smooth_buf)
        else:
            smooth_pred = ""
            confidence  = 0.0

        # Stability hysteresis
        if smooth_pred and smooth_pred == prev_pred:
            stable_count = min(stable_count + 1, STABLE_FRAMES)
        else:
            stable_count = max(stable_count - 2, 0)

        prev_pred   = smooth_pred
        stable_prog = stable_count / STABLE_FRAMES
        live_letter = smooth_pred if (smooth_pred and confidence >= 0.55) else ""

        # ── Auto-add letter when stable ──────────────────────
        # A letter is committed only when:
        #   1. It has been held stably for STABLE_FRAMES frames
        #   2. It is DIFFERENT from the last committed letter
        #      (same letter is allowed again only after hand was removed)
        if (stable_count >= STABLE_FRAMES
                and smooth_pred
                and confidence >= 0.55
                and smooth_pred != last_added):      # ← key guard

            with state_lock:
                shared["current_word"] += smooth_pred
                shared["status_msg"]    = f"Added: {smooth_pred}"

            last_added   = smooth_pred   # lock out this letter until hand leaves
            stable_count = 0             # reset so it doesn't fire again immediately

        # FPS counter
        fps_c += 1
        now_t  = time.time()
        if now_t - fps_t >= 1.0:
            fps_val = fps_c
            fps_c   = 0
            fps_t   = now_t
            print(f"[FPS] {fps_val}  hand={'Y' if hand_ok else 'N'}")

        # ── Write everything to shared in one lock ────────────
        with state_lock:
            shared["frame_jpg"]   = jpg_bytes
            shared["live_letter"] = live_letter
            shared["top3"]        = [[l, p] for l, p in top3[:3]]
            shared["stable_prog"] = stable_prog
            shared["hand_ok"]     = hand_ok
            shared["fps"]         = fps_val

    cap.release()

# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    missing = [f for f in [INDEX_PATH, TRANSLATOR_PATH] if not os.path.exists(f)]
    for f in missing:
        print(f"[WARN] '{f}' not found — place it next to main.py")

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    print(f"\n[SERVER]  http://{HOST}:{PORT}")
    print(f"          Cover page  → http://{HOST}:{PORT}/")
    print(f"          Translator  → http://{HOST}:{PORT}/translator")
    print(f"[PERF]    {CAP_W}×{CAP_H} · JPEG={JPEG_QUALITY} · " 
          f"skip_every={INFER_EVERY} · mp_complexity=0")
    print(f"[STYLE]   Hand overlay → monochromatic lavender-purple")
    print(f"[FIX]     Letter repeats disabled — remove & return hand to re-sign")
    print(f"          Ctrl+C to quit\n")

    try:
        run_capture()
    except KeyboardInterrupt:
        print("\n[DONE] Shutting down.")
        server.shutdown()
        sys.exit(0)