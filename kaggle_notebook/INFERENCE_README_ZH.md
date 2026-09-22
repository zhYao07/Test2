# Kaggle 测试集推理与提交

使用 [Kaggle_RSNA_Inference.ipynb](./Kaggle_RSNA_Inference.ipynb)。它有三个代码单元：第 1 格是数据预处理，第 2 格是模型定义，第 3 格是推理配置与执行。三个单元格直接在 Notebook 内运行，不创建或执行 `.py` 文件。双 T4 各有独立模型，并由两个线程并行推理；唯一写入的输出是 `/kaggle/working/submission.csv`。此 Notebook 对应本目录 `Kaggle_RSNA_3cells.ipynb` 训练出的 checkpoint。

**权重版本注意**：如果 `best.pt` 来自旧版 `baseline_v1`（`torch.hub` DINOv2），这个 Notebook 不能直接加载。该权重的 backbone 参数名与当前 Hugging Face 版本不同；仅把 Hugging Face DINOv2 模型添加为 Input 不会转换旧权重。启动时会检查并给出明确错误。请使用 Kaggle 训练版或 `baseline_v2` 产出的 `best.pt`。

## Kaggle Notebook 需要挂载的输入

1. **比赛数据**：在 Notebook 的 `Add Input` 中添加 `RSNA Knee Abnormality Detection` 竞赛数据。推理会用到 `test.csv`、`test_series.csv`、`test_series/` 和 `sample_submission.csv`。不需要把 570 GB 的比赛数据重新上传。
2. **训练权重**：把训练 Notebook 输出的 `/kaggle/working/rsna_outputs/best.pt` 保存为 Kaggle Dataset，或将训练 Notebook 的输出作为 Input 添加到推理 Notebook。需要的是 `best.pt`；`last.pt`、`split.csv`、弱标签 CSV 和训练数据都不需要上传。
3. **DINOv2 Hugging Face 模型**：添加训练时使用的同一个 Kaggle Model/Dataset，目录内应有 `config.json` 和 `pytorch_model.bin`。虽然 `best.pt` 含完整模型权重，当前模型类在构建时仍需读取这两个文件，然后严格加载 `best.pt` 覆盖全部参数。
4. **推理 Notebook**：上传或导入 `Kaggle_RSNA_Inference.ipynb`。不需要额外上传 `rsna_data.py`、`rsna_model.py`、`infer.py`。

## 操作

1. 接受比赛规则，创建 Kaggle Notebook，导入推理 Notebook 并添加上述三个 Input。
2. 选择 `GPU T4 x2` Accelerator，关闭 Internet。
3. 第 3 格顶部已经填写你当前 Kaggle 日志确认的三个路径。如挂载名称改变，直接修改 `DATA_ROOT`、`CHECKPOINT_PATH`、`DINOV2_MODEL_DIR`；GPU 数和分块大小也在同一处。
4. 依次运行三个代码单元。推理会校验 UID、列顺序、预测形状、数值范围和有限性，并输出 `/kaggle/working/submission.csv`。
5. 在 Kaggle 中 `Save Version` → `Save & Run All`；版本运行成功后，从该版本选择 `Submit to Competition`。

Notebook 使用两个推理线程，各使用一张 T4、FP16 和 batch size 1；每个线程处理不同的 Study，最后在内存中合并为一个 `submission.csv`。当前路径已经显式填写，不需要遍历 DICOM 目录搜索输入。不在提交时训练模型。若 Kaggle 镜像缺少 `pydicom`，需提前把对应 wheel 作为离线输入安装，不能在关闭 Internet 后在线安装。

## 重新生成 Notebook

若修改了本目录下的数据、模型或推理源码，在本地运行 `python kaggle_notebook/build_inference_notebook.py` 重新生成 Notebook，再上传新版。
