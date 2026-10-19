"""
probe_features.py
用 frozen EfficientNetB0 提取单帧特征，训练线性分类器
快速验证特征是否对 Confusion 有判别力
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

TRAIN_X      = 'train_X.mmap'
VAL_X        = 'val_X.mmap'
TRAIN_SHAPE  = (4885, 8, 224, 224, 3)
VAL_SHAPE    = (1429, 8, 224, 224, 3)
TRAIN_LABELS = 'train_labels4_filtered.npz'
VAL_LABELS   = 'val_labels4_filtered.npz'

print("加载 backbone...")
backbone = keras.applications.EfficientNetB0(
    include_top=False, weights='imagenet',
    pooling='avg', input_shape=(224, 224, 3)
)

print("加载 mmap...")
X_train = np.memmap(TRAIN_X, dtype='float32', mode='r', shape=TRAIN_SHAPE)
X_val   = np.memmap(VAL_X,   dtype='float32', mode='r', shape=VAL_SHAPE)

y_train = np.load(TRAIN_LABELS)['Confusion'].astype(np.int32)
y_val   = np.load(VAL_LABELS)['Confusion'].astype(np.int32)

# 取每个视频的中间帧（第4帧）作为代表帧
print("提取训练集特征...")
feats_train = []
for i in range(0, len(X_train), 64):
    batch = X_train[i:i+64, 4]  # 第4帧
    feats_train.append(backbone(batch, training=False).numpy())
feats_train = np.concatenate(feats_train, axis=0)

print("提取验证集特征...")
feats_val = []
for i in range(0, len(X_val), 64):
    batch = X_val[i:i+64, 4]
    feats_val.append(backbone(batch, training=False).numpy())
feats_val = np.concatenate(feats_val, axis=0)

print(f"Feature shape: {feats_train.shape}")

# 标准化
scaler = StandardScaler()
feats_train = scaler.fit_transform(feats_train)
feats_val   = scaler.transform(feats_val)

# 线性分类器
print("\n训练 Logistic Regression...")
clf = LogisticRegression(
    max_iter=1000, class_weight='balanced', C=1.0, random_state=42
)
clf.fit(feats_train, y_train)

y_pred = clf.predict(feats_val)
print(f"\nAccuracy: {accuracy_score(y_val, y_pred):.4f}")
print(f"Majority baseline: {(y_val==0).sum()/len(y_val):.4f}")
print(classification_report(y_val, y_pred,
      target_names=['VeryLow','Low','High','VeryHigh'], zero_division=0))
