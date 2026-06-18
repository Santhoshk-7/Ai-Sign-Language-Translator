"""
train_words.py  -  ASL Word Model Trainer
Reads data_words.csv, trains RandomForest, saves word_model.pkl

Run after data_words.py:  python train_words.py
"""

import pandas as pd
import numpy as np
import pickle, os, time, collections, warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics         import classification_report, accuracy_score

CSV_FILE   = "data_words.csv"
MODEL_FILE = "word_model.pkl"
TEST_SIZE  = 0.15
CV_FOLDS   = 5
MIN_SAMPLES = 50

LINE = "─" * 56
def hdr(t): print(f"\n{LINE}\n  {t}\n{LINE}")
def bar(v, w=25): return "█"*int(v*w) + "░"*(w-int(v*w))

# ── 1. Load ──────────────────────────────────────────────────────
hdr("STEP 1 — LOAD DATA")

if not os.path.exists(CSV_FILE):
    print(f"  ✗  '{CSV_FILE}' not found. Run data_words.py first.")
    exit(1)

df = pd.read_csv(CSV_FILE, header=None)
print(f"  Rows     : {len(df)}")
print(f"  Features : {df.shape[1]-1}")

X = df.iloc[:, :-1].values.astype(np.float32)
y = df.iloc[:, -1].values.astype(str)

counts = collections.Counter(y)
labels = sorted(counts.keys())
print(f"  Words    : {labels}\n")

for lbl in labels:
    n    = counts[lbl]
    flag = "  ⚠  need more!" if n < MIN_SAMPLES else ""
    print(f"  {lbl:<22} [{bar(n/max(counts.values()))}]  {n:>4}{flag}")

# drop NaN rows
nan_rows = np.isnan(X).any(axis=1).sum()
if nan_rows:
    mask = ~np.isnan(X).any(axis=1)
    X, y = X[mask], y[mask]
    print(f"\n  Dropped {nan_rows} NaN rows.")

# ── 2. Split ─────────────────────────────────────────────────────
hdr("STEP 2 — TRAIN / TEST SPLIT")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=42, stratify=y)
print(f"  Train : {len(X_train)}  |  Test : {len(X_test)}")

# ── 3. Train ─────────────────────────────────────────────────────
hdr("STEP 3 — TRAIN MODEL")
model = RandomForestClassifier(
    n_estimators      = 500,
    max_depth         = None,
    min_samples_split = 4,
    min_samples_leaf  = 2,
    max_features      = "sqrt",
    class_weight      = "balanced",
    n_jobs            = -1,
    random_state      = 42,
)
print("  Training RandomForest (500 trees) ...")
t0 = time.time()
model.fit(X_train, y_train)
print(f"  Done in {time.time()-t0:.2f}s")

# ── 4. Cross-validation ──────────────────────────────────────────
hdr("STEP 4 — CROSS-VALIDATION")
skf    = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
print(f"  Fold scores : {[f'{s:.3f}' for s in scores]}")
print(f"  Mean ± Std  : {scores.mean():.4f} ± {scores.std():.4f}")
if scores.mean() >= 0.95:
    print("  ✓  Excellent!")
elif scores.mean() >= 0.88:
    print("  ~  Good — more data per word will improve further.")
else:
    print("  ⚠  Below 88% — collect more samples per word.")

# ── 5. Test set ──────────────────────────────────────────────────
hdr("STEP 5 — TEST SET EVALUATION")
y_pred   = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"  Overall accuracy : {accuracy*100:.2f}%\n")
print(classification_report(y_test, y_pred, zero_division=0))

print("  PER-WORD ACCURACY")
for lbl in sorted(set(y)):
    mask = y_test == lbl
    if not mask.any():
        continue
    acc  = accuracy_score(y_test[mask], y_pred[mask])
    flag = "  ⚠" if acc < 0.85 else ""
    print(f"  {lbl:<22} [{bar(acc)}]  {acc*100:5.1f}%{flag}")

# ── 6. Save ──────────────────────────────────────────────────────
hdr("STEP 6 — SAVE MODEL")
with open(MODEL_FILE, "wb") as f:
    pickle.dump(model, f)
size_kb = os.path.getsize(MODEL_FILE) / 1024
print(f"  ✓  Saved '{MODEL_FILE}'  ({size_kb:.0f} KB)")
print(f"  Words : {list(model.classes_)}")

hdr("DONE")
print(f"  Words    : {len(model.classes_)}")
print(f"  CV acc   : {scores.mean()*100:.2f}% ± {scores.std()*100:.2f}%")
print(f"  Test acc : {accuracy*100:.2f}%")
print(f"\n  Run main_words.py to start the translator!\n")