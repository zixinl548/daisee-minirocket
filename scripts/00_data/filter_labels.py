import numpy as np
import pandas as pd
import cv2

for split, csv in [('train', 'train_ready.csv'), ('val', 'validation_ready.csv')]:
    df = pd.read_csv(csv)
    df.columns = df.columns.str.strip()
    
    valid_mask = []
    for _, row in df.iterrows():
        cap = cv2.VideoCapture(row['video_path'])
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        valid_mask.append(total > 0)
    
    valid_mask = np.array(valid_mask)
    df_valid = df[valid_mask].reset_index(drop=True)
    
    np.savez(f'{split}_labels4_filtered.npz',
             Confusion=df_valid['Confusion'].values.astype(np.int8),
             Engagement=df_valid['Engagement'].values.astype(np.int8),
             Boredom=df_valid['Boredom'].values.astype(np.int8),
             Frustration=df_valid['Frustration'].values.astype(np.int8))
    
    print(f"{split}: {valid_mask.sum()} valid / {len(df)} total")
    print(f"  Confusion distribution:")
    for i in range(4):
        print(f"    class {i}: {(df_valid['Confusion']==i).sum()}")
