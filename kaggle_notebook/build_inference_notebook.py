"""Build a three-cell, fully inline Kaggle inference notebook."""

import json
import re
from pathlib import Path


HERE = Path(__file__).parent
data_source = (HERE / "rsna_data.py").read_text(encoding="utf-8")
data_source = re.sub(
    r'\nif __name__ == "__main__":\n.*?(?=\n__all__ =)',
    "\n", data_source, flags=re.DOTALL,
)
model_source = (HERE / "rsna_model.py").read_text(encoding="utf-8")
infer_source = (HERE / "infer.py").read_text(encoding="utf-8")
model_source = model_source.replace("from rsna_data import LABELS, SERIES_FEATURES, SLOTS\n", "")
infer_source = infer_source.replace("from rsna_data import LABELS, KneeDataset, _prepare_series_df\n", "")
infer_source = infer_source.replace("from rsna_model import RSNADINOv2\n", "")


def code_cell(source):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}

notebook = {
    "cells": [
        code_cell(data_source),
        code_cell(model_source),
        code_cell(infer_source),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
(HERE / "Kaggle_RSNA_Inference.ipynb").write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8"
)
