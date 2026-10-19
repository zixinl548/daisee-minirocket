import numpy as np
import pandas as pd

for split, csv in [('train', 'train_ready.csv'), ('val', 'validation_ready.csv')]:
    df = pd.read_csv(csv)
    df.columns = df.columns.str.strip()
    np.savez(f'{split}_labels4.npz',
             Confusion=df['Confusion'].values.astype(np.int8),
             Engagement=df['Engagement'].values.astype(np.int8),
             Boredom=df['Boredom'].values.astype(np.int8),
             Frustration=df['Frustration'].values.astype(np.int8))
    print(f"{split} Confusion distribution:")
    for i in range(4):
        print(f"  class {i}: {(df['Confusion']==i).sum()}")
