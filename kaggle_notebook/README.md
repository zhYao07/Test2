# Kaggle RSNA notebook package

Open `Kaggle_RSNA_3cells.ipynb` in Kaggle and run its three cells in order. It is a self-contained notebook: the cells contain the actual data, model, and training code, and do not write or execute any `.py` file.

The third cell starts two in-notebook DDP worker processes, one per T4 GPU. It is configured for one study per GPU, gradient accumulation of two steps, and FP16 AMP. Checkpoints and the train/validation split are written to `/kaggle/working/rsna_outputs`.

The scripts auto-discover the attached RSNA data, weak-label CSV, and Hugging Face DINOv2 model directory from `/kaggle/input`. The DINOv2 input must contain `config.json` and `pytorch_model.bin` (the format shown in Kaggle's model browser). It is loaded with `transformers.AutoModel.from_pretrained(..., local_files_only=True)`, so it never downloads from GitHub or Hugging Face.

If more than one matching input is attached, change the arguments in `sys.argv = ["notebook_train"]` near the end of the third cell, for example:

```bash
sys.argv = ["notebook_train", "--data-root", "/kaggle/input/your-rsna-data", "--labels-csv", "/kaggle/input/your-labels/llm_labels_v4_blend.csv", "--dinov2-model-dir", "/kaggle/input/dinov2-model-directory"]
```

The Kaggle image normally already provides PyTorch, pandas, scikit-learn, and pydicom. If `pydicom` is missing, install it in a setup cell before running these three cells.
