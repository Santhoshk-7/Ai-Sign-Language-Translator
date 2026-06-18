"""
data.py  —  ASL Data Collector  (A–Z, 500 samples/letter)

HOW TO USE
  python data.py
  - Show the ASL sign for the displayed letter
  - Press S to start/stop recording — hold sign steady
  - Press N to skip a letter
  - Press ESC to quit

KEY TIPS FOR ACCURACY
  - Collect 500 samples per letter (more for tricky ones)
  - Slightly vary hand angle each 50 samples
  - Do multiple sessions — lighting variation helps
  - For M/N/R/U/G/H: be very precise and deliberate
"""

import csv, cv2, os, collections
import mediapipe as mp
import numpy as np
from features import extract

CSV_FILE       = "data.csv"
SAMPLES_NEEDED = 500
ALL_LABELS     = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
HARD_PAIRS     = {"M","N","R","U","G","H","D","E","F","K","P","A","S","T"}

ASL_TIPS = {
    "A": "Fist. Thumb rests on the side (NOT over fingers like S)",
    "B": "All 4 fingers straight up, thumb tucked across palm",
    "C": "Curved hand forming a C — fingers together",
    "D": "Index finger up, other 3 fingers + thumb touch to form circle",
    "E": "ALL fingers curl down to palm, thumb tucked under fingers",
    "F": "Index + thumb touch forming circle, other 3 fingers spread up",
    "G": "Index + thumb point HORIZONTALLY to the side (like a gun sideways)",
    "H": "Index + middle BOTH point horizontally sideways, together",
    "I": "Pinky only pointing up, fist",
    "J": "Like I but draw J shape — collected as static pinky-up",
    "K": "Index up, middle finger angled out, thumb between index+middle",
    "L": "Index pointing up, thumb pointing out — classic L shape",
    "M": "3 fingers (index+middle+ring) folded OVER the thumb",
    "N": "2 fingers (index+middle) folded over thumb (one fewer than M)",
    "O": "All fingers + thumb curved together forming an O circle",
    "P": "Like K but the whole hand points DOWNWARD",
    "Q": "Like G but pointing DOWNWARD instead of sideways",
    "R": "Index + middle fingers CROSSED over each other",
    "S": "Fist with thumb across the FRONT of fingers (not side like A)",
    "T": "Fist with thumb tucked BETWEEN index and middle fingers",
    "U": "Index + middle fingers up TOGETHER side by side (NOT crossed)",
    "V": "Index + middle fingers up in V shape — fingers SPREAD apart",
    "W": "Index + middle + ring all spread up — 3 fingers",
    "X": "Index finger hooked/crooked like a hook",
    "Y": "Thumb + pinky extended out, other 3 fingers folded",
    "Z": "Index finger traces Z in air — collect as index pointing out",
}

mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(max_num_hands=1,
                          min_detection_confidence=0.75,
                          min_tracking_confidence=0.65)
mp_draw  = mp.solutions.drawing_utils

C_BG = (18,18,18); C_PANEL=(32,32,32); C_GREEN=(50,220,100)
C_ACCENT=(0,200,180); C_YELLOW=(30,210,230); C_ORANGE=(30,140,255)
C_GRAY=(120,120,120); C_WHITE=(240,240,240); C_RED=(60,60,220)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def count_existing():
    c = collections.Counter()
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE) as f:
            for row in csv.reader(f):
                if row: c[row[-1]] += 1
    return c


def pick_label(counts):
    for lbl in ALL_LABELS:
        if counts.get(lbl, 0) < SAMPLES_NEEDED:
            return lbl
    return None


def draw_bar(canvas, x, y, w, h, val, max_val, color=C_ACCENT):
    cv2.rectangle(canvas, (x,y), (x+w,y+h), (50,50,50), -1)
    filled = int(w * min(val/max(max_val,1), 1.0))
    if filled: cv2.rectangle(canvas, (x,y), (x+filled,y+h), color, -1)
    cv2.rectangle(canvas, (x,y), (x+w,y+h), C_GRAY, 1)


def draw_grid(canvas, counts, current, x0, y0, cell=28):
    for i, lbl in enumerate(ALL_LABELS):
        cx = x0 + (i%13)*(cell+4)
        cy = y0 + (i//13)*(cell+4)
        done   = counts.get(lbl,0) >= SAMPLES_NEEDED
        is_cur = lbl == current
        is_hard= lbl in HARD_PAIRS
        if is_cur:   bg,tc = C_ACCENT,(10,10,10)
        elif done:   bg,tc = (40,100,40),C_GREEN
        elif is_hard:bg,tc = (60,50,20),C_ORANGE
        else:        bg,tc = (45,45,45),C_GRAY
        cv2.rectangle(canvas,(cx,cy),(cx+cell,cy+cell),bg,-1)
        cv2.putText(canvas,lbl,(cx+6,cy+cell-7),FONT,0.5,tc,1)


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counts     = count_existing()
label      = pick_label(counts)
collecting = False
flash      = 0
status     = "Press S to start collecting"

print(f"\n  ASL Collector  |  goal: {SAMPLES_NEEDED}/letter  |  89 features")
print(f"  Orange = tricky ASL letters — be extra precise!\n")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res   = hands.process(rgb)

    flat    = None
    hand_ok = False

    if res.multi_hand_landmarks:
        for handLms in res.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=C_ACCENT, thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=C_WHITE,  thickness=1))
            flat    = extract([[lm.x,lm.y] for lm in handLms.landmark])
            hand_ok = True

    if collecting and hand_ok and flat and label:
        with open(CSV_FILE,"a",newline="") as f:
            csv.writer(f).writerow(flat+[label])
        counts[label] = counts.get(label,0)+1
        flash = 3
        if counts[label] >= SAMPLES_NEEDED:
            collecting = False
            status     = f"✓ '{label}' done! Press S for next."
            label      = pick_label(counts)
        if label is None:
            status = "🎉 ALL 26 LETTERS DONE! Press ESC."

    PH = 260
    canvas = np.zeros((h+PH, w, 3), dtype=np.uint8)
    canvas[:] = C_BG
    if flash > 0:
        cv2.rectangle(frame,(0,0),(w-1,h-1),C_GREEN,6); flash-=1
    canvas[:h,:w] = frame

    cv2.rectangle(canvas,(8,h+8),(w-8,h+PH-8),C_PANEL,-1)

    if label:
        lc = C_ORANGE if label in HARD_PAIRS else C_ACCENT
        cv2.putText(canvas,"COLLECTING",(20,h+34),FONT,0.45,C_GRAY,1)
        cv2.putText(canvas,label,(20,h+90),FONT,2.8,lc,4)
        cur = counts.get(label,0)
        draw_bar(canvas,90,h+60,240,16,cur,SAMPLES_NEEDED)
        cv2.putText(canvas,f"{cur}/{SAMPLES_NEEDED}",(90,h+54),FONT,0.42,C_GRAY,1)

        # tip — wrap into two lines if long
        tip = ASL_TIPS.get(label,"")
        if len(tip) > 52:
            mid = tip.rfind(" ", 0, 52)
            cv2.putText(canvas,tip[:mid],(20,h+108),FONT,0.38,C_YELLOW,1)
            cv2.putText(canvas,tip[mid+1:],(20,h+124),FONT,0.38,C_YELLOW,1)
        else:
            cv2.putText(canvas,tip,(20,h+108),FONT,0.38,C_YELLOW,1)

        if label in HARD_PAIRS:
            cv2.putText(canvas,"TRICKY — read tip carefully!",(20,h+138),FONT,0.38,C_ORANGE,1)
    else:
        cv2.putText(canvas,"ALL DONE!",(20,h+75),FONT,1.5,C_GREEN,3)

    hc = C_GREEN if hand_ok else C_RED
    cv2.circle(canvas,(20,h+152),7,hc,-1)
    cv2.putText(canvas,"HAND OK" if hand_ok else "NO HAND",(34,h+157),FONT,0.42,hc,1)
    rc = C_RED if collecting else C_GRAY
    cv2.circle(canvas,(125,h+152),7,rc,-1)
    cv2.putText(canvas,"RECORDING" if collecting else "PAUSED",(139,h+157),FONT,0.42,rc,1)

    draw_grid(canvas,counts,label or "",20,h+167)

    cv2.putText(canvas,status,(20,h+228),FONT,0.45,C_YELLOW,1)
    cv2.putText(canvas,"[S] start/stop   [N] skip   [ESC] quit",
                (20,h+248),FONT,0.38,C_GRAY,1)

    cv2.imshow("ASL Data Collector",canvas)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        if label is None: status="All letters done!"
        else:
            collecting = not collecting
            status = (f"Recording '{label}'... hold sign steady" if collecting else "Paused.")
    elif key == ord('n'):
        if label:
            old=label; tmp=dict(counts); tmp[old]=SAMPLES_NEEDED
            label=pick_label(tmp); collecting=False
            status=f"Skipped '{old}' → '{label}'" if label else "All done!"
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("\n── SUMMARY ─────────────────────────────────────────")
counts = count_existing()
total  = 0
for lbl in ALL_LABELS:
    c    = counts.get(lbl,0)
    done = c >= SAMPLES_NEEDED
    bar  = "█"*int(c*20//SAMPLES_NEEDED)+"░"*(20-int(c*20//SAMPLES_NEEDED))
    tag  = " ⚠ TRICKY" if lbl in HARD_PAIRS and not done else ""
    print(f"  {'✓' if done else ' '} {lbl}  [{bar}]  {c:>4}/{SAMPLES_NEEDED}{tag}")
    total += c
print(f"\n  Total rows : {total}  |  Complete : {sum(1 for l in ALL_LABELS if counts.get(l,0)>=SAMPLES_NEEDED)}/26\n")