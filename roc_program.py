import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.utils.class_weight import compute_class_weight

# ── 1. LOAD ────────────────────────────────────────────────────────────────────
df = pd.read_csv("Technology in Life Survey.csv")

# ── 2. DROP LEAKY / UNINFORMATIVE COLUMNS ─────────────────────────────────────
columns_to_drop = [
    col for col in df.columns
    if any(kw in col for kw in ["Score", "Feedback", "Timestamp", "Total score",
                                 "Did you enjoy",     # unrelated to security
                                 "please explain why" # 53 % missing, free-text
                                ])
]
df = df.drop(columns=columns_to_drop)

# ── 3. TARGET  (3-class: No / recovered / lost) ────────────────────────────────
target_col = "Did you ever get hacked?"
target_map = {
    "No":                              0,
    "Yes, but I got the account back": 1,
    "Yes, I lost the account":         2,
}
df["target"] = df[target_col].map(target_map)
df = df.drop(columns=[target_col])

# ── 4. FEATURE ENGINEERING ────────────────────────────────────────────────────
# 4a. Fix grade typo: "Sophmore" → "Sophomore" so encoder sees one category
grade_col = "What grade are you in?"
df[grade_col] = df[grade_col].str.replace("Sophmore", "Sophomore", regex=False)

# 4b. Ordinal-encode grade (there is a natural order)
grade_order = [["Freshman(9th)", "Sophomore(10th)", "Junior(11th)",
                 "Senior (12th)", "Adult (graduated)"]]
oe = OrdinalEncoder(categories=grade_order, handle_unknown="use_encoded_value", unknown_value=-1)
df[[grade_col]] = oe.fit_transform(df[[grade_col]])

# 4c. Binary risk flag: reuses AND uses a short/easy password → highest risk
df["high_risk_pw"] = (
    (df["Do you reuse passwords for multiple accounts? "] == "Yes") &
    (df["If you reuse the same password, Is your password short/easy or long/complex?"] == "Short and Easy")
).astype(int)

# 4d. One-hot encode remaining categoricals
#     (avoids the label-leakage bug in the original where a SINGLE LabelEncoder
#      was refit on every column, making label assignments arbitrary)
remaining_cats = df.select_dtypes(include="object").columns.tolist()
df = pd.get_dummies(df, columns=remaining_cats, drop_first=True)

# ── 5. TRAIN / TEST SPLIT (stratified) ────────────────────────────────────────
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=4, stratify=y   # FIX: stratify preserves class balance
)

# ── 6. CLASS WEIGHTS (handles the 68 / 28 / 18 imbalance) ───────────────────
classes = np.unique(y_train)
weights = compute_class_weight("balanced", classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))

# ── 7. MODELS ─────────────────────────────────────────────────────────────────
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        max_depth=10,           
        min_samples_leaf=3,
        class_weight=class_weight_dict,
        random_state=4,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=500,
        max_depth=10,
        learning_rate=0.05,
        random_state=4,
    ),
}

# ── 8. CROSS-VALIDATED EVALUATION ─────────────────────────────────────────────
# FIX: with only ~116 training rows a single 80/20 split is unreliable.
# StratifiedKFold gives a much more honest estimate of generalisation.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=4)

print("=" * 55)
print("  5-Fold Cross-Validated ROC-AUC (OvR, macro)")
print("=" * 55)

best_model, best_auc, best_name = None, -1, ""

for name, clf in models.items():
    cv_scores = cross_val_score(clf, X_train, y_train,
                                cv=cv, scoring="roc_auc_ovr_weighted", n_jobs=-1)
    print(f"{name:25s}  {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    if cv_scores.mean() > best_auc:
        best_auc, best_model, best_name = cv_scores.mean(), clf, name

print()

# ── 9. FINAL FIT + HOLD-OUT EVALUATION ────────────────────────────────────────
best_model.fit(X_train, y_train)
y_pred  = best_model.predict(X_test)
y_probs = best_model.predict_proba(X_test)

test_auc = roc_auc_score(y_test, y_probs, multi_class="ovr", average="weighted")
print(f"Best model  : {best_name}")
print(f"Test ROC-AUC: {test_auc:.3f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=["Not hacked", "Recovered", "Lost account"]))

# ── 10. PLOTS ──────────────────────────────────────────────────────────────────
class_names = ["Not hacked", "Recovered", "Lost account"]
colors = ["steelblue", "darkorange", "green"]

# After model.fit and predict_proba, create a model confusion matrix
cm = confusion_matrix(y_test, y_pred)

#Set all important features into an importances series
importances = pd.Series(best_model.feature_importances_, index=X.columns)

# 10a. ROC Curves ---
fig1, ax1 = plt.subplots(figsize=(10, 6))
for i, (cname, color) in enumerate(zip(class_names, colors)):
    fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_probs[:, i])
    auc_i = roc_auc_score((y_test == i).astype(int), y_probs[:, i])
    ax1.plot(fpr, tpr, color=color, label=f"{cname} (AUC={auc_i:.2f})")
ax1.plot([0, 1], [0, 1], "k--", linewidth=0.8)
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.set_title("ROC Curves (One-vs-Rest)")
ax1.legend(fontsize=8)
fig1.savefig("roc_curves.png", dpi=150)
plt.close(fig1)

# 10b. Confusion Matrix ---
fig2, ax2 = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax2, colorbar=False)
ax2.set_title("Confusion Matrix")
fig2.savefig("confusion_matrix.png", dpi=150)
plt.close(fig2)

# 10c. Feature Importances ---
fig3, ax3 = plt.subplots(figsize=(25, 10))
importances.nlargest(10).sort_values().plot(kind="barh", ax=ax3, color="steelblue")
ax3.set_title("Top 10 Feature Importances")
ax3.set_xlabel("Importance")
fig3.savefig("feature_importances.png", dpi=300)
plt.close(fig3)