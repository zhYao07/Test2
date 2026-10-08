# Baseline_v21_5fold：RadImageNet ResNet50 五折

基于当前 v21，保留模型、数据预处理、损失、优化器、EMA、可选 Top3 EMA SWA，以及统一的输入/header 缓存。模型与数据代码、label.csv 与 v21 完全相同。

弱标签 Study 按 v20 的多标签迭代分层方法分为五份。每折训练其中四份，另一份只用于训练结束后的 OOF 评估。每个 epoch 都在同一批完整的 **58 条官方 gold label** 上验证，以 gold Macro AUC 选择该折最佳 epoch。58 条金标始终不进入训练；coverage 阈值与正例权重仅由当前折训练成员拟合。这与 v20_v1 的选择规则相同。金标分数是参与选模型的验证分数，不是独立测试分数。

从本目录运行，默认顺序训练全部五折，每折重新初始化模型、优化器、调度器、EMA 和 SWA：

```bash
torchrun --standalone --nproc_per_node=4 train.py \
  --data-root /root/RSNA/rsna-knee-abnormality-detection \
  --output-dir ./outputs/cv5 \
  --cache-dir ../Baseline_v20/input_cache \
  --image-size 224 \
  --backbone-mode layer3+layer4 \
  --epochs 20 \
  --backbone-lr 2e-3 \
  --ema-decay 0.9995 \
  --split-seed 42
```

其余默认参数沿用 v21：batch size=2、梯度累积=2、head LR=3e-4、140 mm crop、24 个训练窗口、bf16、关闭 metadata。默认关闭 SWA，可追加 `--swa`；SWA 另外保存，不替换按金标最佳 epoch 选择的 fold 权重。每折从原始 RadImageNet 权重开始，默认 `../radimagenet/ResNet50.pt`，可用 `--radimagenet-weights` 指定。禁止 `--init-checkpoint`，以防旧模型见过当前折的 OOF 成员。

只训练第三折：追加 `--fold 3`。续训第三折：追加 `--fold 3 --resume ./outputs/cv5/fold_2/last.pt`。对外折号为 1..5；成员表、checkpoint 元数据和诊断子目录沿用 0..4，**fold1.pt 对应 fold_0，fold5.pt 对应 fold_4**。续训检查当前折成员、标签指纹、模型、预训练文件、训练参数和 coverage 配置；已有权重时不会静默重训覆盖。

```text
outputs/cv5/
  fold1.pt ... fold5.pt         # 每折金标 Macro AUC 最佳 checkpoint，训练中实时保存
  cv_manifest.json / folds.csv / cv_targets.csv / split_distribution.csv
  fold_0/ ... fold_4/
    last.pt                    # 完整训练状态，可续训
    swa.pt                     # 仅 --swa 且有三个有效 EMA epoch 时生成
    cv.json / data_split.csv / fold_metrics.json
    validation_history.json / loss_curve.png
    oof_predictions.csv / gold_predictions.csv
    series_quality.csv / series_selection.csv / series_selection_summary.json
  oof_predictions.csv          # 五折完成后汇总
  gold_ensemble_predictions.csv / cv_metrics.json
```

fold1.pt 至 fold5.pt 包含实际验证的 EMA 权重（`--no-ema` 时为原始模型），以及原始训练状态、折成员和预处理配置。SWA 沿用 v21 的 Top3 gold AUC EMA epoch 参数平均规则，另存各折目录的 swa.pt。

仅预览划分，不扫描 DICOM、不加载预训练权重：

```bash
python train.py --data-root /root/RSNA/rsna-knee-abnormality-detection \
  --output-dir ./outputs/split_preview --split-only
```

五折完成后自动合并 OOF 与金标等权概率。也可运行 `python cv_results.py --output-dir ./outputs/cv5`。

Kaggle 使用本目录 `Kaggle_Inference.ipynb`，挂载 fold1.pt 至 fold5.pt，默认自动发现五份文件；有多余 checkpoint 时显式填写 `CHECKPOINT_PATHS`。校验五折元数据、v21 backbone spec 和共同预处理参数，使用各折自己的 coverage 阈值，平均 sigmoid 后的概率。无需另挂原始 RadImageNet 权重。保留 v20 五折的共享 DICOM 读取与预取；五个 ResNet50 模型的实际 GPU 显存和推理耗时需在 Kaggle 确认。

本地验证：

```bash
python -m unittest discover -s Baseline_v21_5fold -p 'test_*.py' -v
python tools/check_line_endings.py
```

回归覆盖真实标签划分、58 条金标隔离、五折合成训练/EMA/保存/续训/OOF 汇总、不同折续训拒绝、Kaggle checkpoint 加载和等权融合。本地为 CPU 验证，未执行真实 DICOM 五折训练或多卡 CUDA 训练。
