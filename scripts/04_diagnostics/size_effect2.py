import numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.metrics import balanced_accuracy_score

ROOT = "/work/users/z/i/zixinl/Emotion"
P = pd.read_csv(f"{ROOT}/align_check/oof_preds.csv",     dtype={'clip': str, 'subj': str})
S = pd.read_csv(f"{ROOT}/align_check/face_size_all.csv", dtype={'clip': str, 'subj': str})
D = P.merge(S[['clip', 'w', 'h', 'area']], on='clip', how='inner')
SETS = ['ALL29', 'AU24', 'POSE3']
for s in SETS:
    D[f'ok_{s}'] = (D[f'{s}_binary'] == D['y2']).astype(int)
D['w10']  = D['w'] / 10
D['w10c'] = D['w10'] - D.groupby('subj')['w10'].transform('mean')

q = D['area'].quantile([1/3, 2/3]).values
D['terc'] = np.where(D['area'] <= q[0], '1_small',
             np.where(D['area'] <= q[1], '2_mid', '3_large'))

print("=== 各档的标签构成（关键：构成不同则 accuracy 不可比）===")
print(D.groupby('terc').agg(
    n=('clip', 'size'), w_med=('w', 'median'), conf_rate=('y2', 'mean'),
    lv0=('y4', lambda v: (v == 0).mean()), lv1=('y4', lambda v: (v == 1).mean()),
    lv2=('y4', lambda v: (v == 2).mean()), lv3=('y4', lambda v: (v == 3).mean())
).round(4))

print("\n=== 各档 binary BALANCED accuracy（对构成免疫）===")
for t, gg in D.groupby('terc'):
    vals = {s: balanced_accuracy_score(gg['y2'], gg[f'{s}_binary']) for s in SETS}
    print(f"  {t}  n={len(gg):5d}  " + "   ".join(f"{s} {vals[s]:.4f}" for s in SETS))

print("\n=== 人脸大小本身是不是行为信号（组内）===")
m = sm.Logit(D['y2'], sm.add_constant(D[['w10c']])).fit(
        disp=0, cov_type='cluster', cov_kwds={'groups': D['subj']})
b, se, p = m.params['w10c'], m.bse['w10c'], m.pvalues['w10c']
print(f"  logit(困惑) ~ 组内中心化宽度:  beta {b:+.4f}  se {se:.4f}  z {b/se:+.2f}  p {p:.4f}")
print("  正 = 同一个人靠近屏幕时更可能被标为困惑")

print("\n=== 交互检验：AU24 的尺寸效应是否强于 POSE3 ===")
L = pd.concat([
    D[['clip', 'subj', 'w10c']].assign(ok=D['ok_AU24'].values,  au=1.0),
    D[['clip', 'subj', 'w10c']].assign(ok=D['ok_POSE3'].values, au=0.0)],
    ignore_index=True)
L['inter'] = L['w10c'] * L['au']
m = sm.Logit(L['ok'], sm.add_constant(L[['w10c', 'au', 'inter']])).fit(
        disp=0, cov_type='cluster', cov_kwds={'groups': L['clip']})
print(m.summary().tables[1])
b, se, p = m.params['inter'], m.bse['inter'], m.pvalues['inter']
print(f"\n交互项  beta {b:+.4f}  se {se:.4f}  z {b/se:+.2f}  p {p:.4f}")
