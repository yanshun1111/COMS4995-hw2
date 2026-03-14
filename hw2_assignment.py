"""
Assignment 2: From Trees to Neural Networks
Home Credit Default Risk Dataset
GBDT (XGBoost) vs MLP (scikit-learn) Comparison
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import time
import os
import json
import gc

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, average_precision_score,
                             precision_recall_curve)
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
import xgboost as xgb

warnings.filterwarnings('ignore')
np.random.seed(42)

SAVE_DIR = '/sessions/blissful-amazing-mendel/plots'
os.makedirs(SAVE_DIR, exist_ok=True)

DATA_PATH = '/sessions/blissful-amazing-mendel/mnt/hw2/home-credit-default-risk/application_train.csv'

# ============================================================
# 1. DATA PREPARATION
# ============================================================
print("=" * 60)
print("1. DATA PREPARATION")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"Original shape: {df.shape}")
print(f"Target distribution:\n{df['TARGET'].value_counts(normalize=True)}")

# Stratified subsample to 80K rows for computational feasibility
# while maintaining class balance
df_majority = df[df['TARGET'] == 0].sample(n=73600, random_state=42)
df_minority = df[df['TARGET'] == 1].sample(n=6400, random_state=42)
df = pd.concat([df_majority, df_minority]).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Subsampled shape: {df.shape}")
print(f"Subsampled target dist:\n{df['TARGET'].value_counts(normalize=True)}")

# Drop SK_ID_CURR (identifier)
df = df.drop('SK_ID_CURR', axis=1)

# --- Handle anomalous values ---
anomalous_mask = df['DAYS_EMPLOYED'] == 365243
print(f"\nDAYS_EMPLOYED anomalous (365243): {anomalous_mask.sum()} rows")
df['DAYS_EMPLOYED_ANOMALY'] = anomalous_mask.astype(int)
df.loc[anomalous_mask, 'DAYS_EMPLOYED'] = np.nan

# --- Drop columns with >60% missing ---
missing_pct = df.isnull().mean()
high_missing_cols = missing_pct[missing_pct > 0.6].index.tolist()
print(f"Dropping {len(high_missing_cols)} columns with >60% missing")
df = df.drop(columns=high_missing_cols)
print(f"Shape after drop: {df.shape}")

# --- Separate features and target ---
y = df['TARGET']
X = df.drop('TARGET', axis=1)

cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(f"\nCategorical: {len(cat_cols)}, Numerical: {len(num_cols)}")

# --- Feature Engineering ---
X['CREDIT_INCOME_RATIO'] = X['AMT_CREDIT'] / (X['AMT_INCOME_TOTAL'] + 1)
X['ANNUITY_INCOME_RATIO'] = X['AMT_ANNUITY'] / (X['AMT_INCOME_TOTAL'] + 1)
X['CREDIT_ANNUITY_RATIO'] = X['AMT_CREDIT'] / (X['AMT_ANNUITY'] + 1)
X['AGE_YEARS'] = (-X['DAYS_BIRTH']) / 365
X['EMPLOYED_YEARS'] = (-X['DAYS_EMPLOYED']) / 365
X['INCOME_PER_PERSON'] = X['AMT_INCOME_TOTAL'] / (X['CNT_FAM_MEMBERS'] + 1)
print(f"Shape after feature engineering: {X.shape}")

# --- Split 70/15/15 ---
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15/0.85, random_state=42, stratify=y_temp)
print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
del df, X, y, X_temp, y_temp
gc.collect()

# --- Encode categoricals (fit on train only) ---
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X_train[col] = X_train[col].fillna('Missing')
    le.fit(X_train[col])
    label_encoders[col] = le
    X_train[col] = le.transform(X_train[col])
    for sX in [X_val, X_test]:
        sX[col] = sX[col].fillna('Missing')
        mask = ~sX[col].isin(le.classes_)
        if mask.any():
            sX.loc[mask, col] = le.transform(['Missing'])[0]
        else:
            sX[col] = le.transform(sX[col])

# --- Impute missing numericals (fit on train only) ---
num_cols_updated = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
train_medians = X_train[num_cols_updated].median()
X_train[num_cols_updated] = X_train[num_cols_updated].fillna(train_medians)
X_val[num_cols_updated] = X_val[num_cols_updated].fillna(train_medians)
X_test[num_cols_updated] = X_test[num_cols_updated].fillna(train_medians)
print(f"Missing after imputation - Train: {X_train.isnull().sum().sum()}")

# --- Scale for MLP (fit on train only) ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# --- Oversample minority for MLP ---
X_tr_s = X_train_scaled.copy()
y_tr = y_train.values.copy()
idx_maj = np.where(y_tr == 0)[0]
idx_min = np.where(y_tr == 1)[0]
n_upsample = int(len(idx_maj) * 0.4)
idx_min_up = np.random.choice(idx_min, size=n_upsample, replace=True)
idx_combined = np.concatenate([idx_maj, idx_min_up])
np.random.shuffle(idx_combined)
X_train_mlp = X_tr_s[idx_combined]
y_train_mlp = y_tr[idx_combined]
print(f"\nMLP oversampled training: {X_train_mlp.shape}")
print(f"MLP target dist: 0={np.mean(y_train_mlp==0):.3f}, 1={np.mean(y_train_mlp==1):.3f}")

print("\nData preparation complete!")

# ============================================================
# 2. GRADIENT BOOSTED TREE (XGBoost)
# ============================================================
print("\n" + "=" * 60)
print("2. XGBoost")
print("=" * 60)

scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

# Baseline
print("\nTraining baseline XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=500, learning_rate=0.1, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    scale_pos_weight=scale_pos_weight,
    random_state=42, eval_metric='logloss',
    early_stopping_rounds=30, n_jobs=-1
)
t0 = time.time()
xgb_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
xgb_train_time = time.time() - t0
print(f"Time: {xgb_train_time:.2f}s, Best iter: {xgb_model.best_iteration}")

# Plot train vs val loss
results = xgb_model.evals_result()
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(results['validation_0']['logloss'], label='Training Loss', linewidth=2)
ax.plot(results['validation_1']['logloss'], label='Validation Loss', linewidth=2)
ax.axvline(x=xgb_model.best_iteration, color='red', linestyle='--', alpha=0.7,
           label=f'Early Stop ({xgb_model.best_iteration})')
ax.set_xlabel('Boosting Round', fontsize=12)
ax.set_ylabel('Log Loss', fontsize=12)
ax.set_title('XGBoost: Training vs Validation Loss', fontsize=14)
ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/xgb_train_val_loss.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: xgb_train_val_loss.png")

# Feature importance
fi = pd.DataFrame({'feature': X_train.columns, 'importance': xgb_model.feature_importances_}).sort_values('importance', ascending=False)
fig, ax = plt.subplots(figsize=(10, 8))
top = fi.head(20)
ax.barh(range(20), top['importance'].values[::-1], color='steelblue')
ax.set_yticks(range(20))
ax.set_yticklabels(top['feature'].values[::-1], fontsize=10)
ax.set_xlabel('Feature Importance (Gain)', fontsize=12)
ax.set_title('XGBoost: Top 20 Feature Importances', fontsize=14)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/xgb_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: xgb_feature_importance.png")

# Learning rate comparison
learning_rates = [0.01, 0.1, 0.3]
lr_results = {}
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, lr in enumerate(learning_rates):
    print(f"Training XGBoost LR={lr}...")
    m = xgb.XGBClassifier(
        n_estimators=500, learning_rate=lr, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42, eval_metric='logloss',
        early_stopping_rounds=30, n_jobs=-1
    )
    m.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
    r = m.evals_result()
    yp = m.predict(X_val)
    ypp = m.predict_proba(X_val)[:, 1]
    lr_results[lr] = {'best_iter': m.best_iteration, 'model': m,
                       'acc': accuracy_score(y_val, yp), 'f1': f1_score(y_val, yp),
                       'auc_pr': average_precision_score(y_val, ypp)}
    axes[idx].plot(r['validation_0']['logloss'], label='Train', linewidth=1.5)
    axes[idx].plot(r['validation_1']['logloss'], label='Val', linewidth=1.5)
    axes[idx].axvline(x=m.best_iteration, color='red', linestyle='--', alpha=0.7)
    axes[idx].set_title(f'LR={lr} (stop@{m.best_iteration})', fontsize=12)
    axes[idx].set_xlabel('Round'); axes[idx].set_ylabel('Log Loss')
    axes[idx].legend(); axes[idx].grid(True, alpha=0.3)
    print(f"  Acc={lr_results[lr]['acc']:.4f}, F1={lr_results[lr]['f1']:.4f}, AUC-PR={lr_results[lr]['auc_pr']:.4f}")

plt.suptitle('XGBoost: Effect of Learning Rate', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/xgb_learning_rate_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: xgb_learning_rate_comparison.png")

best_xgb = lr_results[0.1]['model']
# Free memory
for lr in [0.01, 0.3]:
    del lr_results[lr]['model']
gc.collect()

# ============================================================
# 3. MLP
# ============================================================
print("\n" + "=" * 60)
print("3. MLP")
print("=" * 60)

print("\nTraining baseline MLP (128,64)...")
mlp_model = MLPClassifier(
    hidden_layer_sizes=(128, 64), activation='relu',
    learning_rate_init=0.001, max_iter=300,
    random_state=42, early_stopping=True,
    validation_fraction=0.1, n_iter_no_change=20,
    batch_size=512, verbose=False
)
t0 = time.time()
mlp_model.fit(X_train_mlp, y_train_mlp)
mlp_train_time = time.time() - t0
print(f"Time: {mlp_train_time:.2f}s, Iters: {mlp_model.n_iter_}")

# Loss curve
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(mlp_model.loss_curve_, label='Training Loss', linewidth=2, color='darkorange')
if hasattr(mlp_model, 'validation_scores_') and mlp_model.validation_scores_:
    ax2 = ax.twinx()
    ax2.plot(mlp_model.validation_scores_, label='Validation Score', linewidth=2, color='green', linestyle='--')
    ax2.set_ylabel('Validation Score', fontsize=12, color='green')
    ax2.legend(loc='center right', fontsize=11)
ax.set_xlabel('Iteration', fontsize=12); ax.set_ylabel('Training Loss', fontsize=12)
ax.set_title('MLP: Training Loss Curve', fontsize=14)
ax.legend(loc='upper right', fontsize=11); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/mlp_loss_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: mlp_loss_curve.png")

# Architecture comparison
architectures = {'(64,)': (64,), '(128, 64)': (128, 64), '(256, 128, 64)': (256, 128, 64)}
arch_results = {}
fig, ax = plt.subplots(figsize=(10, 6))
for name, arch in architectures.items():
    print(f"Training MLP {name}...")
    m = MLPClassifier(hidden_layer_sizes=arch, activation='relu', learning_rate_init=0.001,
                      max_iter=300, random_state=42, early_stopping=True,
                      validation_fraction=0.1, n_iter_no_change=20, batch_size=512, verbose=False)
    m.fit(X_train_mlp, y_train_mlp)
    yp = m.predict(X_val_scaled); ypp = m.predict_proba(X_val_scaled)[:, 1]
    arch_results[name] = {'accuracy': accuracy_score(y_val, yp), 'f1': f1_score(y_val, yp),
                           'auc_pr': average_precision_score(y_val, ypp), 'n_iter': m.n_iter_}
    ax.plot(m.loss_curve_, label=f'{name} (F1={arch_results[name]["f1"]:.3f})', linewidth=2)
    print(f"  Acc={arch_results[name]['accuracy']:.4f}, F1={arch_results[name]['f1']:.4f}, AUC-PR={arch_results[name]['auc_pr']:.4f}")
    del m; gc.collect()

ax.set_xlabel('Iteration', fontsize=12); ax.set_ylabel('Training Loss', fontsize=12)
ax.set_title('MLP: Effect of Network Architecture', fontsize=14)
ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/mlp_architecture_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: mlp_architecture_comparison.png")

# Activation comparison
act_results = {}
for act in ['relu', 'tanh']:
    print(f"Training MLP activation={act}...")
    m = MLPClassifier(hidden_layer_sizes=(128, 64), activation=act, learning_rate_init=0.001,
                      max_iter=300, random_state=42, early_stopping=True,
                      validation_fraction=0.1, n_iter_no_change=20, batch_size=512, verbose=False)
    m.fit(X_train_mlp, y_train_mlp)
    yp = m.predict(X_val_scaled); ypp = m.predict_proba(X_val_scaled)[:, 1]
    act_results[act] = {'accuracy': accuracy_score(y_val, yp), 'f1': f1_score(y_val, yp),
                         'auc_pr': average_precision_score(y_val, ypp)}
    print(f"  Acc={act_results[act]['accuracy']:.4f}, F1={act_results[act]['f1']:.4f}, AUC-PR={act_results[act]['auc_pr']:.4f}")
    del m; gc.collect()

# Learning rate comparison
lr_inits = [0.001, 0.01, 0.1]
mlp_lr_results = {}
fig, ax = plt.subplots(figsize=(10, 6))
for lr_init in lr_inits:
    print(f"Training MLP LR={lr_init}...")
    m = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', learning_rate_init=lr_init,
                      max_iter=300, random_state=42, early_stopping=True,
                      validation_fraction=0.1, n_iter_no_change=20, batch_size=512, verbose=False)
    m.fit(X_train_mlp, y_train_mlp)
    yp = m.predict(X_val_scaled); ypp = m.predict_proba(X_val_scaled)[:, 1]
    mlp_lr_results[lr_init] = {'accuracy': accuracy_score(y_val, yp), 'f1': f1_score(y_val, yp),
                                'auc_pr': average_precision_score(y_val, ypp)}
    ax.plot(m.loss_curve_, label=f'LR={lr_init} (F1={mlp_lr_results[lr_init]["f1"]:.3f})', linewidth=2)
    print(f"  Acc={mlp_lr_results[lr_init]['accuracy']:.4f}, F1={mlp_lr_results[lr_init]['f1']:.4f}")
    del m; gc.collect()

ax.set_xlabel('Iteration', fontsize=12); ax.set_ylabel('Training Loss', fontsize=12)
ax.set_title('MLP: Effect of Learning Rate', fontsize=14)
ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/mlp_learning_rate_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: mlp_learning_rate_comparison.png")

# ============================================================
# 4. COMPARISON
# ============================================================
print("\n" + "=" * 60)
print("4. GBDT vs MLP COMPARISON")
print("=" * 60)

y_test_pred_xgb = best_xgb.predict(X_test)
y_test_proba_xgb = best_xgb.predict_proba(X_test)[:, 1]
y_test_pred_mlp = mlp_model.predict(X_test_scaled)
y_test_proba_mlp = mlp_model.predict_proba(X_test_scaled)[:, 1]

metrics = {}
for name, yp, ypp in [('XGBoost', y_test_pred_xgb, y_test_proba_xgb),
                       ('MLP', y_test_pred_mlp, y_test_proba_mlp)]:
    metrics[name] = {
        'Accuracy': accuracy_score(y_test, yp),
        'Precision': precision_score(y_test, yp, zero_division=0),
        'Recall': recall_score(y_test, yp),
        'F1-Score': f1_score(y_test, yp),
        'AUC-PR': average_precision_score(y_test, ypp),
    }

cdf = pd.DataFrame(metrics).T
print("\n--- Test Set Performance ---")
print(cdf.round(4).to_string())
print(f"\nTraining Times: XGBoost={xgb_train_time:.2f}s, MLP={mlp_train_time:.2f}s")

# Comparison plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
metric_names = list(metrics['XGBoost'].keys())
x = np.arange(len(metric_names)); w = 0.35
xv = [metrics['XGBoost'][m] for m in metric_names]
mv = [metrics['MLP'][m] for m in metric_names]
axes[0].bar(x - w/2, xv, w, label='XGBoost', color='steelblue', alpha=0.85)
axes[0].bar(x + w/2, mv, w, label='MLP', color='darkorange', alpha=0.85)
axes[0].set_xticks(x); axes[0].set_xticklabels(metric_names, rotation=20, fontsize=10)
axes[0].set_ylabel('Score', fontsize=12); axes[0].set_title('GBDT vs MLP: Test Metrics', fontsize=14)
axes[0].legend(fontsize=11); axes[0].set_ylim(0, 1.05); axes[0].grid(True, alpha=0.3, axis='y')
for i, (v1, v2) in enumerate(zip(xv, mv)):
    axes[0].text(i - w/2, v1 + 0.02, f'{v1:.3f}', ha='center', va='bottom', fontsize=8)
    axes[0].text(i + w/2, v2 + 0.02, f'{v2:.3f}', ha='center', va='bottom', fontsize=8)

prec_xgb, rec_xgb, _ = precision_recall_curve(y_test, y_test_proba_xgb)
prec_mlp, rec_mlp, _ = precision_recall_curve(y_test, y_test_proba_mlp)
axes[1].plot(rec_xgb, prec_xgb, label=f'XGBoost (AUC-PR={metrics["XGBoost"]["AUC-PR"]:.3f})', linewidth=2, color='steelblue')
axes[1].plot(rec_mlp, prec_mlp, label=f'MLP (AUC-PR={metrics["MLP"]["AUC-PR"]:.3f})', linewidth=2, color='darkorange')
axes[1].set_xlabel('Recall', fontsize=12); axes[1].set_ylabel('Precision', fontsize=12)
axes[1].set_title('Precision-Recall Curves', fontsize=14)
axes[1].legend(fontsize=11); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/gbdt_vs_mlp_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: gbdt_vs_mlp_comparison.png")

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(['XGBoost', 'MLP'], [xgb_train_time, mlp_train_time], color=['steelblue', 'darkorange'], alpha=0.85)
for bar, t in zip(bars, [xgb_train_time, mlp_train_time]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{t:.1f}s', ha='center', fontsize=12)
ax.set_ylabel('Training Time (seconds)', fontsize=12)
ax.set_title('Training Time Comparison', fontsize=14); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/training_time_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: training_time_comparison.png")

# Save data for report
comparison_data = {
    'metrics': metrics,
    'xgb_train_time': xgb_train_time, 'mlp_train_time': mlp_train_time,
    'xgb_best_iter': best_xgb.best_iteration, 'mlp_n_iter': mlp_model.n_iter_,
    'arch_results': arch_results, 'act_results': act_results,
    'lr_results_xgb': {str(lr): lr_results[lr] for lr in [0.01, 0.3]},
    'lr_results_xgb_01': {'best_iter': lr_results[0.1]['best_iter'],
                           'acc': lr_results[0.1]['acc'], 'f1': lr_results[0.1]['f1'],
                           'auc_pr': lr_results[0.1]['auc_pr']},
    'mlp_lr_results': {str(k): v for k, v in mlp_lr_results.items()},
    'top_features': fi.head(10).to_dict('records'),
    'n_features': int(X_train.shape[1]),
    'n_train': int(X_train.shape[0]), 'n_val': int(X_val.shape[0]), 'n_test': int(X_test.shape[0]),
    'n_dropped_cols': len(high_missing_cols), 'scale_pos_weight': float(scale_pos_weight),
    'cat_cols': cat_cols,
}
with open(f'{SAVE_DIR}/comparison_data.json', 'w') as f:
    json.dump(comparison_data, f, indent=2, default=str)

print("\n" + "=" * 60)
print("ALL COMPLETE!")
print("=" * 60)
