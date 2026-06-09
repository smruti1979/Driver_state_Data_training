import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_curve

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
DATASET_FILE = "driver_fatigue_dataset.csv"
MODEL_OUTPUT_FILE = "fatigue_classifier.pkl"

if not os.path.exists(DATASET_FILE):
    raise FileNotFoundError(f"❌ Could not find {DATASET_FILE}. Please log data from your dashboard first.")

print("🔍 Loading dataset...")
df = pd.read_csv(DATASET_FILE)

# Remove rows containing missing data points
df.dropna(inplace=True)

# Define our raw features and target label
features = ["ear", "mar", "head_deviation"]
X = df[features]
y = df["label"]

print(f"📊 Dataset successfully loaded! Shape: {df.shape}")
print(f"   - Alert samples (Label 0): {sum(y == 0)}")
print(f"   - Fatigued samples (Label 1): {sum(y == 1)}")

# ==========================================
# 2. TRAIN-TEST SPLIT
# ==========================================
# 'stratify=y' ensures equal ratios of 0 and 1 in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"✂️  Data split complete: {len(X_train)} training rows, {len(X_test)} verification rows.")

# ==========================================
# 3. MODEL INITIALIZATION & TRAINING
# ==========================================
print("\n🏋️‍♂️ Training Random Forest Classifier...")
# n_estimators=100 uses a forest of 100 decision trees to prevent over-fitting
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)
print("✅ Training complete.")

# ==========================================
# 4. METRICS EVALUATION (WITH CUSTOM THRESHOLD)
# ==========================================
# ==========================================
# AUTOMATED OPTIMAL THRESHOLD FINDER
# ==========================================

# Get raw fatigue probabilities
y_prob = model.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

# Find the threshold closest to your safety target (e.g., catching 92% of fatigue)
TARGET_RECALL = 0.92
best_idx = np.where(recalls >= TARGET_RECALL)[0][-1]
OPTIMAL_THRESHOLD = thresholds[best_idx]

# Apply the mathematically optimal threshold
y_pred = (y_prob >= OPTIMAL_THRESHOLD).astype(int)

print(f"🎯 Automatically selected optimal threshold: {OPTIMAL_THRESHOLD:.4f}")
print(f"📈 Achieved Target Fatigue Recall: {recalls[best_idx]*100:.1f}%")



# ==========================================
# 5. MODEL SERIALIZATION
# ==========================================
print(f"\n💾 Saving model weights...")
joblib.dump(model, MODEL_OUTPUT_FILE)
print(f"🎉 Success! Production model saved as: '{MODEL_OUTPUT_FILE}'")

# ==========================================
# 6. OPTIONAL VISUALIZATION (Saves a plot)
# ==========================================
try:
    plt.figure(figsize=(6, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Alert', 'Fatigued'], yticklabels=['Alert', 'Fatigued'])
    plt.ylabel('Actual Truth')
    plt.xlabel('AI Prediction')
    plt.title('Fatigue Detection Confusion Matrix')
    plt.savefig('model_confusion_matrix.png')
    print("📈 Saved validation diagnostic graph to 'model_confusion_matrix.png'")
except Exception:
    pass
