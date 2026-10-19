# 下阶段任务（作业）· 草稿

> 待队友过目补充。标 **？** 的地方我没确认。

---

## 任务 1 · 把要比较的 baseline 方法列出来，写一个初步计划

**每个方法记四件事：**
- 怎么做的（一句话）
- 实现难度
- 有没有现成代码
- 用什么指标比

**候选方法（待补充）**
- Majority baseline（必须有，我们是 0.675）
- 我们现在的 EfficientNetB0 + BiGRU（作为现状对照，0.636 / balanced 0.311）
- MiniRocket + Ridge，跑在 AU 时间序列上 → 有代码（sktime），几分钟，不要 GPU
- Transformer（Zerveas 那套），跑在 AU 时间序列上 → 有代码，要 GPU
- ResNet + TCN（Abedi）→ 有代码但没有权重、没有 DAiSEE loader，重训要几天 GPU
- **？** 还要不要加别的

**比较哪些指标**
- Accuracy
- Balanced accuracy（也就是 UAR，每一类的准确率取平均）
- Macro F1
- 四类各自的混淆矩阵
- 注：**accuracy 单独看会误导**，因为多数类占 67.5%，什么都不学直接猜 Very Low 就有 0.675

---

## 任务 2 · 把要比较的 baseline 数据集列出来（三个）

1. **DAiSEE**
2. **ReactionNet？** 我不确定是哪个，可能是 EngageNet、EmotiW-EW 或 EmotioNet，**要问队友**
3. **？** 第三个还没定

**每个数据集记：**
- 多少段视频、多少个人
- 标签是什么、几个等级
- 标签粒度（一段一个标签，还是逐帧）
- 有没有官方划分
- 划分是不是 subject-disjoint（训练和测试的人有没有重叠）

---

## AU pipeline（会上提到的几点）

**结论性的几句**
- AU 提取有现成工具，**速度很快**
- AU 对表情的刻画已经比较成熟，不用我们自己造
- AU 怎么算的：有现成的，不需要自己实现

**流程**
```
抽帧 → 人脸检测 → align 对齐 → AU 提取 → AU 时间序列 → 分类
```

**要做的事**
- 把 DAiSEE 的数据放进 **LibreFace** 跑一遍
- **比较 workflow：OpenFace vs LibreFace**
  - 抽不出来的帧占多少，**而且要按 confusion 等级分开看**（如果困惑的片段更容易失败，就是和标签相关的缺失，比随机缺失严重）
  - 两个工具各能给哪些 AU（LibreFace 没有 AU27，AU7 只有有无不给强度）
  - 速度
  - 量得准不准：LibreFace 的 AU25 是 0.94，OpenFace 是 0.85，而 AU25 是 Borges 发现最关键的那个
- AU 时间序列出来之后进 **MiniRocket**

---

## 待确认
- 第二、第三个数据集具体是哪两个
- 方法清单是否完整
- 主指标用哪个
