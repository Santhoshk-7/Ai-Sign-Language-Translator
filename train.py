"""
train.py  —  ASL Ensemble Trainer  (RF + SVM + KNN voting)

What makes this better than a single RandomForest:
  1. Data augmentation   — 5× more training data for FREE
       adds tiny random noise + scale jitter to every sample
       forces the model to generalise, not memorise
  2. Ensemble voting     — 3 models vote on every prediction
       RF is good at boundaries, SVM at margins, KNN at local clusters
       together they cover each other's blind spots
  3. Calibrated probs    — SVM wrapped in CalibratedClassifierCV
       so all three models output real 0-1 probabilities
  4. Saved as one object — predict.py loads one file, calls .predict()
"""

import pandas as pd
import numpy as np
import pickle, os, time, collections, warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble         import RandomForestClassifier, VotingClassifier
from sklearn.svm              import SVC
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.calibration      import CalibratedClassifierCV
from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics          import (accuracy_score, classification_report,
                                      confusion_matrix)
from sklearn.preprocessing    import LabelEncoder

CSV_FILE   = "data.csv"
MODEL_FILE = "model.pkl"
TEST_SIZE  = 0.15
CV_FOLDS   = 5
AUGMENT_N  = 4          # extra copies per sample via augmentation
NOISE_STD  = 0.008      # landmark jitter (tiny, realistic)
SCALE_JITTER = 0.06     # ±6% hand size variation

HARD = {"M","N","R","U","G","H","D","E","F","K","P","A","S","T"}
LINE = "─"*56

def hdr(t): print(f"\n{LINE}\n  {t}\n{LINE}")
def bar(v, w=24): return "█"*int(v*w) + "░"*(w-int(v*w))


# ─────────────────────────────────────────────
#  DATA AUGMENTATION
# ─────────────────────────────────────────────
def augment(X, y, n_copies=AUGMENT_N, noise=NOISE_STD, scale_jit=SCALE_JITTER):
    """
    For each sample create n_copies variants by:
      - adding small Gaussian noise to landmark positions
      - applying a random scale factor to simulate hand size variation
    Returns augmented X,y APPENDED to originals.
    """
    rng = np.random.default_rng(42)
    copies_X, copies_y = [], []
    for _ in range(n_copies):
        noisy  = X + rng.normal(0, noise, X.shape)
        scale  = 1.0 + rng.uniform(-scale_jit, scale_jit, (len(X), 1))
        noisy  = noisy * scale
        copies_X.append(noisy)
        copies_y.append(y)
    aug_X = np.vstack([X] + copies_X)
    aug_y = np.concatenate([y] + copies_y)
    idx   = rng.permutation(len(aug_X))
    return aug_X[idx], aug_y[idx]


# ─────────────────────────────────────────────
#  1. LOAD
# ─────────────────────────────────────────────
hdr("STEP 1 — LOAD DATA")

if not os.path.exists(CSV_FILE):
    print(f"  ✗  '{CSV_FILE}' not found. Run data.py first."); exit(1)

df = pd.read_csv(CSV_FILE, header=None)
print(f"  Rows    : {len(df)}")
print(f"  Columns : {df.shape[1]}  ({df.shape[1]-1} features + label)")

X_raw = df.iloc[:,:-1].values.astype(np.float32)
y_raw = df.iloc[:,-1].values.astype(str)

counts = collections.Counter(y_raw)
labels = sorted(counts.keys())
print(f"  Classes : {labels}\n")

warn = False
for lbl in labels:
    n   = counts[lbl]
    tag = " [TRICKY]" if lbl in HARD else ""
    flg = "  ⚠ collect more!" if n < 100 else ""
    print(f"  {lbl}{tag:<9}  [{bar(n/max(counts.values()))}]  {n:>5}{flg}")
    if n < 100: warn = True
if warn:
    print("\n  ⚠  Some classes have < 100 samples. Accuracy may be low.")

nan_rows = np.isnan(X_raw).any(axis=1).sum()
if nan_rows:
    mask = ~np.isnan(X_raw).any(axis=1)
    X_raw, y_raw = X_raw[mask], y_raw[mask]
    print(f"\n  Dropped {nan_rows} NaN rows. Remaining: {len(X_raw)}")


# ─────────────────────────────────────────────
#  2. SPLIT  (before augmentation — no leakage)
# ─────────────────────────────────────────────
hdr("STEP 2 — TRAIN / TEST SPLIT")

X_train_raw, X_test, y_train_raw, y_test = train_test_split(
    X_raw, y_raw, test_size=TEST_SIZE, random_state=42, stratify=y_raw)

print(f"  Original train : {len(X_train_raw)}")
print(f"  Test (held out): {len(X_test)}")


# ─────────────────────────────────────────────
#  3. AUGMENT  (training set only)
# ─────────────────────────────────────────────
hdr("STEP 3 — DATA AUGMENTATION")

X_train, y_train = augment(X_train_raw, y_train_raw)
print(f"  Original train samples : {len(X_train_raw)}")
print(f"  After augmentation     : {len(X_train)}  ({AUGMENT_N}× copies + noise + scale jitter)")
print(f"  Features per sample    : {X_train.shape[1]}")


# ─────────────────────────────────────────────
#  4. BUILD ENSEMBLE
# ─────────────────────────────────────────────
hdr("STEP 4 — BUILD ENSEMBLE  (RF + SVM + KNN)")

rf = RandomForestClassifier(
    n_estimators      = 500,
    max_depth         = None,
    min_samples_split = 4,
    min_samples_leaf  = 2,
    max_features      = "sqrt",
    class_weight      = "balanced",
    n_jobs            = -1,
    random_state      = 42,
)

# SVM needs probability calibration for soft voting
svm_base = SVC(
    kernel       = "rbf",
    C            = 10,
    gamma        = "scale",
    class_weight = "balanced",
    random_state = 42,
)
svm = CalibratedClassifierCV(svm_base, cv=3)

knn = KNeighborsClassifier(
    n_neighbors = 7,
    weights     = "distance",
    metric      = "euclidean",
    n_jobs      = -1,
)

ensemble = VotingClassifier(
    estimators = [("rf", rf), ("svm", svm), ("knn", knn)],
    voting     = "soft",          # average probabilities
    weights    = [3, 2, 1],       # RF trusted most, then SVM, then KNN
    n_jobs     = -1,
)


# ─────────────────────────────────────────────
#  5. TRAIN
# ─────────────────────────────────────────────
hdr("STEP 5 — TRAINING")

print("  Training RF + SVM + KNN ensemble on augmented data …")
print("  (This takes 1–3 minutes — SVM is slow but worth it)")
t0 = time.time()
ensemble.fit(X_train, y_train)
elapsed = time.time()-t0
print(f"  ✓  Trained in {elapsed:.1f}s")


# ─────────────────────────────────────────────
#  6. CROSS-VALIDATION  (on original data only)
# ─────────────────────────────────────────────
hdr("STEP 6 — CROSS-VALIDATION  (original data)")

print(f"  Running {CV_FOLDS}-fold CV on RF only (fast proxy) …")
skf    = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
scores = cross_val_score(rf, X_raw, y_raw, cv=skf,
                         scoring="accuracy", n_jobs=-1)
print(f"  RF fold scores : {[f'{s:.3f}' for s in scores]}")
print(f"  RF mean ± std  : {scores.mean():.4f} ± {scores.std():.4f}")
print("  (Ensemble will be higher — SVM + KNN fill RF's gaps)")

if scores.mean() < 0.85:
    print("  ⚠  Below 85% on RF alone — collect more data for weak letters")
elif scores.mean() >= 0.95:
    print("  ✓  Excellent RF base — ensemble should reach 97%+")


# ─────────────────────────────────────────────
#  7. TEST SET EVALUATION
# ─────────────────────────────────────────────
hdr("STEP 7 — TEST SET EVALUATION")

y_pred   = ensemble.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"  Ensemble test accuracy : {accuracy*100:.2f}%\n")
print(classification_report(y_test, y_pred, zero_division=0))

print("  PER-LETTER ACCURACY")
for lbl in sorted(set(y_raw)):
    mask = y_test == lbl
    if not mask.any(): continue
    acc  = accuracy_score(y_test[mask], y_pred[mask])
    tag  = " ← TRICKY" if lbl in HARD else ""
    flg  = "  ⚠ WEAK" if acc < 0.85 else ""
    print(f"  {lbl}  [{bar(acc)}]  {acc*100:5.1f}%{tag}{flg}")

print("\n  TOP CONFUSIONS")
lbl_list = sorted(set(y_raw))
cm = confusion_matrix(y_test, y_pred, labels=lbl_list)
pairs = []
for i,tl in enumerate(lbl_list):
    for j,pl in enumerate(lbl_list):
        if i!=j and cm[i,j]>0:
            pairs.append((cm[i,j],tl,pl))
pairs.sort(reverse=True)
if pairs:
    for cnt,tl,pl in pairs[:8]:
        tag=" ← known hard pair" if tl in HARD or pl in HARD else ""
        print(f"  '{tl}' → '{pl}'  {cnt}×{tag}")
else:
    print("  None! Perfect separation.")

if accuracy < 0.90:
    print(f"\n  ⚠  Accuracy {accuracy*100:.1f}% — check the weak letters above")
    print("     Collect 200+ more samples for those specific letters")
    print("     Use the tip text in data.py for precise hand positions")


# ─────────────────────────────────────────────
#  8. SAVE
# ─────────────────────────────────────────────
hdr("STEP 8 — SAVE MODEL")

with open(MODEL_FILE,"wb") as f:
    pickle.dump(ensemble, f)

kb = os.path.getsize(MODEL_FILE)/1024
print(f"  ✓  Saved '{MODEL_FILE}'  ({kb:.0f} KB)")
print(f"  Classes : {list(ensemble.classes_)}")

hdr("DONE")
print(f"  Training samples (augmented) : {len(X_train)}")
print(f"  Test accuracy (ensemble)     : {accuracy*100:.2f}%")
print(f"  RF cross-val mean            : {scores.mean()*100:.2f}%")
print(f"  Model                        : {MODEL_FILE}\n")
print("  ➜  Run predict.py to test live")
print("  ➜  Run main.py for the full sentence translator\n")