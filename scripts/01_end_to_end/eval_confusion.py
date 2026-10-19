import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv('validation_ready.csv')
df.columns = df.columns.str.strip()
y_true = df['Confusion'].values

print("Val set distribution:")
for i in range(4):
    print(f"  class {i}: {(y_true==i).sum()}")

print(f"\nMajority baseline: {(y_true==0).sum()/len(y_true):.4f}")
print(f"Total samples: {len(y_true)}")
