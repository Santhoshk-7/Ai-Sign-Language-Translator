"""
features.py  —  Shared ASL feature extractor (89 values)
Import in data.py, train.py, predict.py — never copy-paste.

Feature breakdown
  42  normalised x,y per landmark      (position + shape)
  20  finger curl ratios               (how bent each segment is)
  10  fingertip pairwise distances     (C(5,2) = 10)
   5  tip-to-wrist distances
   5  MCP-to-wrist distances
   7  key joint angles
 ───
  89  total

Why this separates hard ASL pairs:
  M vs N   → curl[index/middle/ring over thumb] count differs
  R vs U   → tip pairwise distance index↔middle differs
  G vs H   → angle[index-pinky MCP spread] differs
  D vs E   → curl depth of index vs all-curl differs
  A vs S   → thumb position angle differs
  K vs P   → hand tilt captured by wrist-relative y values
"""

import math

WRIST   = 0
THUMB   = [1, 2, 3, 4]
INDEX   = [5, 6, 7, 8]
MIDDLE  = [9, 10, 11, 12]
RING    = [13, 14, 15, 16]
PINKY   = [17, 18, 19, 20]
FINGERS = [THUMB, INDEX, MIDDLE, RING, PINKY]
TIPS    = [4, 8, 12, 16, 20]
MCPS    = [1, 5, 9, 13, 17]


def _dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def _angle(a, b, c):
    """Angle at vertex b, normalised to [0,1]."""
    v1 = (a[0]-b[0], a[1]-b[1])
    v2 = (c[0]-b[0], c[1]-b[1])
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag = (math.sqrt(v1[0]**2+v1[1]**2) *
           math.sqrt(v2[0]**2+v2[1]**2))
    if mag < 1e-9:
        return 0.0
    return math.acos(max(-1.0, min(1.0, dot/mag))) / math.pi


def extract(landmarks):
    """
    landmarks : list of 21 [x, y] from MediaPipe
    returns   : flat list of 89 floats, fully scale+position invariant
    """
    lm = landmarks

    # 1. normalised x,y  (42) ────────────────────────────────────
    bx, by = lm[WRIST]
    norm = []
    for x, y in lm:
        norm.append(x - bx)
        norm.append(y - by)
    scale = max(abs(v) for v in norm) or 1.0
    norm  = [v / scale for v in norm]

    def p(i):
        return (norm[i*2], norm[i*2+1])

    # 2. finger curl ratios  (20) ─────────────────────────────────
    curl = []
    for finger in FINGERS:
        mcp, pip, dip, tip = finger
        full = _dist(p(mcp), p(tip)) or 1e-6
        for a, b in [(mcp, pip), (pip, dip), (dip, tip), (mcp, tip)]:
            curl.append(_dist(p(a), p(b)) / full)

    # 3. fingertip pairwise distances  (10) ───────────────────────
    tip_pts   = [p(t) for t in TIPS]
    tip_dists = []
    for i in range(5):
        for j in range(i+1, 5):
            tip_dists.append(_dist(tip_pts[i], tip_pts[j]))
    mx        = max(tip_dists) or 1.0
    tip_dists = [v/mx for v in tip_dists]

    # 4. tip-to-wrist distances  (5) ──────────────────────────────
    tip_wrist = [_dist(p(t), p(WRIST)) for t in TIPS]
    mx        = max(tip_wrist) or 1.0
    tip_wrist = [v/mx for v in tip_wrist]

    # 5. MCP-to-wrist distances  (5) ──────────────────────────────
    mcp_wrist = [_dist(p(m), p(WRIST)) for m in MCPS]
    mx        = max(mcp_wrist) or 1.0
    mcp_wrist = [v/mx for v in mcp_wrist]

    # 6. key angles  (7) ──────────────────────────────────────────
    angles = [
        _angle(p(THUMB[1]),  p(THUMB[2]),  p(THUMB[3])),   # thumb IP
        _angle(p(INDEX[0]),  p(INDEX[1]),  p(INDEX[2])),   # index PIP
        _angle(p(MIDDLE[0]), p(MIDDLE[1]), p(MIDDLE[2])),  # middle PIP
        _angle(p(RING[0]),   p(RING[1]),   p(RING[2])),    # ring PIP
        _angle(p(PINKY[0]),  p(PINKY[1]),  p(PINKY[2])),   # pinky PIP
        _angle(p(TIPS[1]),   p(WRIST),     p(TIPS[0])),    # index-thumb spread
        _angle(p(MCPS[1]),   p(WRIST),     p(MCPS[4])),    # knuckle spread
    ]

    # 42 + 20 + 10 + 5 + 5 + 7 = 89
    return norm + curl + tip_dists + tip_wrist + mcp_wrist + angles


if __name__ == "__main__":
    dummy = [[i*0.05 % 1, i*0.03 % 1] for i in range(21)]
    f = extract(dummy)
    assert len(f) == 89, f"Expected 89, got {len(f)}"
    print(f"✓  features.py OK  —  {len(f)} features per sample")