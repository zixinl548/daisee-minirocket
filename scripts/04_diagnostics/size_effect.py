import numpy as np, pandas as pd
import statsmodels.api as sm

ROOT = "/work/users/z/i/zixinl/Emotion"
P = pd.read_csv(f"{ROOT}/align_check/oof_preds.csv",     dtype={'clip': str, 'subj': str})
S = pd.read_csv(f"{ROOT}/align_check/face_size_all.csv", dtype={'clip': str, 'subj': str})
D = P.merge(S[['clip', 'w', 'h', 'area']], on='clip', how='inner')
print(f"合并 {len(D)} clips / {D.subj.nunique()} subjects")

SETS = ['ALL29', 'AU24', 'POSE3']
for s in SETS:
    D[f'ok_{s}'] = (D[f'{s}_binary'] == D.y2).astype(int)

D['w10']  = D.w / 10
D['w10c'] = D.w10 - D.groupby('subj').w10.transform('mean')
print(f"组内中心化宽度(10px 单位): SD {D.w10c.std():.3f}  "
      f"IQR {D.w10c.quantile(.75) - D.w10c.quantile(.25):.3f}  "
      f"范围 {D.w10c.min():.2f} .. {D.w10c.max():.2f}")

q = D.area.quantile([1/3, 2/3]).values
D['terc'] = np.where(D.area <= q[0], '1_small',
             np.where(D.area <= q[1], '2_mid', '3_large'))
print("\n=== 描述性：按人脸三档的 binary 准确率（受试者混淆未控制，仅供观察）===")
print(D.groupby('terc').agg(n=('clip', 'size'), w_med=('w', 'median'),
      **{s: (f'ok_{s}', 'mean') for s in SETS}).round(4))

print("\n=== 组内效应：logit(判对) ~ 组内中心化人脸宽度，SE 按 subject 聚类 ===")
print("beta = 人脸每宽 10 像素，判对的 log-odds 变化")
for s in SETS:
    m = sm.Logit(D[f'ok_{s}'], sm.add_constant(D[['w10c']])).fit(
            disp=0, cov_type='cluster', cov_kwds={'groups': D.subj})
    b, se, p = m.params['w10c'], m.bse['w10c'], m.pvalues['w10c']
    print(f"  {s:6s}  beta {b:+.4f}  se {se:.4f}  z {b/se:+.2f}  p {p:.4f}")

print("\n=== 交互检验：AU24 的尺寸效应是否强于 POSE3 ===")
L = pd.concat([
    D[['clip', 'subj', 'w10c']].assign(ok=D.ok_AU24.values,  au=1.0),
    D[['clip', 'subj', 'w10c']].assign(ok=D.ok_POSE3.values, au=0.0)],
    ignore_index=True)
L['inter'] = L.w10c * L.au
m = sm.Logit(L.ok, sm.add_constant(L[['w10c', 'au', 'inter']])).fit(
        disp=0, cov_type='cluster', cov_kwds={'groups': L.clip})
print(m.summary().tables[1])
b, se, p = m.params['inter'], m.bse['inter'], m.pvalues['inter']
print(f"\n交互项  beta {b:+.4f}  se {se:.4f}  z {b/se:+.2f}  p {p:.4f}")
print("正且显著 -> AU 比头部姿态更依赖分辨率，支持解释一")
