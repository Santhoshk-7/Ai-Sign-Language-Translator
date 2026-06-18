"""
data_words.py  -  ASL Word/Phrase Data Collector
Collects 89-feature vectors for whole words and phrases.

HOW TO USE:
  1. Run:  python data_words.py
  2. The current word/phrase is shown on screen
  3. Make the ASL sign for that word
  4. Press S to start/stop recording — hold the sign steady
  5. Press N to skip to next word
  6. Press A to add your own custom word at any time
  7. Press ESC to quit

WORD LIST (edit WORDS dict below to add/remove/change anything):
  - Greetings : Hello, Goodbye, Thank You, Sorry, Please, You're Welcome
  - Basic Needs: Help, Yes, No, Hungry, Thirsty, Tired, Pain, Bathroom
  - Emotions   : Happy, Sad, Angry, Scared, Excited, Confused, Fine, Love
  - Custom     : Add your own in the WORDS dict!

TIPS:
  - Collect 200+ samples per word for best accuracy
  - Vary lighting and hand angle between sessions
  - Keep the sign natural and centered in frame
"""

import csv, cv2, os, collections
import mediapipe as mp
import numpy as np
from features import extract

# ─────────────────────────────────────────────
#  WORD LIST  ←  Edit here to add/remove words
# ─────────────────────────────────────────────
WORDS = {
    # ── Greetings ──────────────────────────────
    "Hello":         "Open hand, wave / flat hand at forehead salute",
    "Goodbye":       "Open hand, wave fingers up and down",
    "Thank You":     "Flat hand from chin, move forward outward",
    "Sorry":         "Fist, rub circular motion on chest",
    "Please":        "Flat hand, rub circular motion on chest",
    "You're Welcome":"Flat hand sweep forward from chest",

    # ── Basic Needs ────────────────────────────
    "Help":          "Thumb up on flat palm, lift both upward",
    "Yes":           "Fist, nod hand up and down (like nodding)",
    "No":            "Index+middle tap thumb together quickly",
    "Hungry":        "C-hand, move down chest from throat",
    "Thirsty":       "Index finger, trace down throat",
    "Tired":         "Bent hands on chest, drop/slump forward",
    "Pain":          "Both index fingers tap together (at pain location)",
    "Bathroom":      "T-hand (thumb between index+middle), shake side to side",

    # ── Emotions ───────────────────────────────
    "Happy":         "Flat hand brush upward on chest (x2)",
    "Sad":           "Both open hands, drag down face slowly",
    "Angry":         "Claw hand on face, pull forward tense",
    "Scared":        "Both fists at chest, fingers burst open outward",
    "Excited":       "Both middle fingers brush upward on chest alternating",
    "Confused":      "One or both index fingers circle at temple",
    "Fine":          "5-hand (open), thumb tap chest once",
    "Love":          "Cross arms over chest (hug yourself)",

    # ── Custom  ←  Add your own words here! ────
    # "Water":       "W-hand (3 fingers), tap chin twice",
    # "Food":        "Flat O-hand, tap lips twice",
    # "More":        "Both flat O-hands, tap fingertips together",
    # "Stop":        "Flat hand chop down onto palm",
    # "Good":        "Flat hand from chin, move to open palm",
}

ALL_LABELS     = list(WORDS.keys())
CSV_FILE       = "data_words.csv"
SAMPLES_NEEDED = 200

# ─────────────────────────────────────────────
#  MEDIAPIPE
# ─────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(max_num_hands=1,
                          min_detection_confidence=0.75,
                          min_tracking_confidence=0.65)
mp_draw  = mp.solutions.drawing_utils

# ─────────────────────────────────────────────
#  COLORS / FONT
# ─────────────────────────────────────────────
C_BG     = (18, 18, 18)
C_PANEL  = (32, 32, 32)
C_GREEN  = (50, 220, 100)
C_ACCENT = (0, 200, 180)
C_YELLOW = (30, 210, 230)
C_ORANGE = (30, 140, 255)
C_GRAY   = (120, 120, 120)
C_WHITE  = (240, 240, 240)
C_RED    = (60, 60, 220)
FONT     = cv2.FONT_HERSHEY_SIMPLEX

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def count_existing():
    counts = collections.Counter()
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE) as f:
            for row in csv.reader(f):
                if row:
                    counts[row[-1]] += 1
    return counts

def pick_label(counts):
    for lbl in ALL_LABELS:
        if counts.get(lbl, 0) < SAMPLES_NEEDED:
            return lbl
    return None

def draw_progress_bar(canvas, x, y, w, h, val, max_val, color=C_ACCENT):
    cv2.rectangle(canvas, (x, y), (x+w, y+h), (50, 50, 50), -1)
    filled = int(w * min(val / max(max_val, 1), 1.0))
    if filled:
        cv2.rectangle(canvas, (x, y), (x+filled, y+h), color, -1)
    cv2.rectangle(canvas, (x, y), (x+w, y+h), C_GRAY, 1)

def draw_word_grid(canvas, counts, current_label, x0, y0):
    """Draw all words as a compact grid with completion status."""
    cell_w, cell_h, pad = 148, 22, 4
    cols = 4
    for i, lbl in enumerate(ALL_LABELS):
        col = i % cols
        row = i // cols
        cx  = x0 + col * (cell_w + pad)
        cy  = y0 + row * (cell_h + pad)
        done   = counts.get(lbl, 0) >= SAMPLES_NEEDED
        is_cur = lbl == current_label
        pct    = min(counts.get(lbl, 0) / SAMPLES_NEEDED, 1.0)

        if is_cur:
            bg, tc = C_ACCENT, (10, 10, 10)
        elif done:
            bg, tc = (40, 100, 40), C_GREEN
        else:
            # partial fill
            bg, tc = (45, 45, 45), C_GRAY
            cv2.rectangle(canvas, (cx, cy),
                          (cx + int(cell_w * pct), cy + cell_h),
                          (30, 70, 60), -1)

        cv2.rectangle(canvas, (cx, cy), (cx+cell_w, cy+cell_h), bg, -1)
        short = lbl[:16]
        cv2.putText(canvas, short, (cx+4, cy+cell_h-6),
                    FONT, 0.38, tc, 1)

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counts     = count_existing()
label      = pick_label(counts)
collecting = False
flash_t    = 0
status_msg = "Press  S  to start collecting"

print("\n" + "="*56)
print("  ASL WORD DATA COLLECTOR  |  89 features")
print("="*56)
print(f"  CSV     : {CSV_FILE}")
print(f"  Goal    : {SAMPLES_NEEDED} samples / word")
print(f"  Words   : {len(ALL_LABELS)}")
print()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res   = hands.process(rgb)

    flat    = None
    hand_ok = False

    if res.multi_hand_landmarks:
        for handLms in res.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, handLms, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=C_ACCENT, thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=C_WHITE,  thickness=1))
            lm_list = [[lm.x, lm.y] for lm in handLms.landmark]
            flat    = extract(lm_list)
            hand_ok = True

    # auto-save
    if collecting and hand_ok and flat and label:
        with open(CSV_FILE, "a", newline="") as f:
            csv.writer(f).writerow(flat + [label])
        counts[label] = counts.get(label, 0) + 1
        flash_t = 3
        if counts[label] >= SAMPLES_NEEDED:
            collecting = False
            status_msg = f"'{label}' done!  Press S for next word."
            label = pick_label(counts)
        if label is None:
            status_msg = "ALL WORDS COMPLETE!  Press ESC."

    # ── canvas ───────────────────────────────────────────────────
    PANEL_H = 310
    canvas  = np.zeros((h + PANEL_H, w, 3), dtype=np.uint8)
    canvas[:] = C_BG

    if flash_t > 0:
        cv2.rectangle(frame, (0, 0), (w-1, h-1), C_GREEN, 6)
        flash_t -= 1
    canvas[:h, :w] = frame

    cv2.rectangle(canvas, (8, h+8), (w-8, h+PANEL_H-8), C_PANEL, -1)

    if label:
        # current word large
        cv2.putText(canvas, "COLLECTING WORD", (20, h+32), FONT, 0.45, C_GRAY, 1)
        cv2.putText(canvas, label, (20, h+70), FONT, 1.6, C_ACCENT, 3)

        # ASL hint
        tip = WORDS.get(label, "")
        # wrap hint text
        words_tip = tip.split()
        line, lines = "", []
        for ww in words_tip:
            if len(line + ww) < 55:
                line += ww + " "
            else:
                lines.append(line.strip())
                line = ww + " "
        lines.append(line.strip())
        for li, ln in enumerate(lines[:2]):
            cv2.putText(canvas, ln, (20, h+90+li*18),
                        FONT, 0.4, C_YELLOW, 1)

        # progress bar
        cur = counts.get(label, 0)
        draw_progress_bar(canvas, 20, h+128, 300, 14, cur, SAMPLES_NEEDED)
        cv2.putText(canvas, f"{cur}/{SAMPLES_NEEDED}",
                    (328, h+140), FONT, 0.42, C_GRAY, 1)
    else:
        cv2.putText(canvas, "ALL DONE!", (20, h+70), FONT, 1.5, C_GREEN, 3)

    # hand / recording indicators
    hc = C_GREEN if hand_ok else C_RED
    cv2.circle(canvas, (20, h+152), 7, hc, -1)
    cv2.putText(canvas, "HAND OK" if hand_ok else "NO HAND",
                (34, h+157), FONT, 0.42, hc, 1)
    rc = C_RED if collecting else C_GRAY
    cv2.circle(canvas, (140, h+152), 7, rc, -1)
    cv2.putText(canvas, "RECORDING" if collecting else "PAUSED",
                (154, h+157), FONT, 0.42, rc, 1)

    # word grid
    draw_word_grid(canvas, counts, label or "", 8, h+168)

    # controls
    cv2.putText(canvas, status_msg,
                (8, h+282), FONT, 0.45, C_YELLOW, 1)
    cv2.putText(canvas,
                "[S] start/stop   [N] skip   [A] add custom word   [ESC] quit",
                (8, h+300), FONT, 0.38, C_GRAY, 1)

    cv2.imshow("ASL Word Collector", canvas)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        if label is None:
            status_msg = "All words already complete!"
        else:
            collecting = not collecting
            status_msg = (f"Recording '{label}'... hold sign steady"
                          if collecting else "Paused.")

    elif key == ord('n'):
        if label:
            old = label
            tmp = dict(counts)
            tmp[old] = SAMPLES_NEEDED
            label = pick_label(tmp)
            collecting = False
            status_msg = (f"Skipped '{old}' -> '{label}'"
                          if label else "All done!")

    elif key == ord('a'):
        # add a custom word via terminal input
        cv2.destroyAllWindows()
        print("\n── ADD CUSTOM WORD ─────────────────────────────")
        new_word = input("  Enter word/phrase: ").strip()
        new_hint = input("  Enter ASL hint (optional): ").strip()
        if new_word and new_word not in WORDS:
            WORDS[new_word]     = new_hint or "Custom sign"
            ALL_LABELS.append(new_word)
            label      = new_word
            collecting = False
            status_msg = f"Added '{new_word}' — press S to record"
            print(f"  Added: '{new_word}'")
        else:
            print("  Skipped (empty or already exists).")
        cap = cv2.VideoCapture(0)

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()

# ── terminal summary ──────────────────────────────────────────────
print("\n── FINAL SUMMARY ───────────────────────────────────────")
counts = count_existing()
total  = 0
for lbl in ALL_LABELS:
    c    = counts.get(lbl, 0)
    done = "✓" if c >= SAMPLES_NEEDED else " "
    bar  = "█" * int(c*25//SAMPLES_NEEDED) + "░"*(25-int(c*25//SAMPLES_NEEDED))
    print(f"  {done} {lbl:<20} [{bar}]  {c:>4}/{SAMPLES_NEEDED}")
    total += c
print(f"\n  Total rows    : {total}")
print(f"  Words done    : {sum(1 for l in ALL_LABELS if counts.get(l,0)>=SAMPLES_NEEDED)}/{len(ALL_LABELS)}\n")