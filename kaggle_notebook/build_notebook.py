"""Generate an exactly-three-cell, self-contained Kaggle notebook."""

import json
from pathlib import Path


HERE = Path(__file__).parent


def code_cell(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


data_source = (HERE / "rsna_data.py").read_text(encoding="utf-8")
model_source = (HERE / "rsna_model.py").read_text(encoding="utf-8").replace(
    "from rsna_data import LABELS, SERIES_FEATURES, SLOTS\n", ""
)
train_source = (HERE / "train.py").read_text(encoding="utf-8")
train_source = train_source.replace(
    "from rsna_data import DEFAULT_ROOT, LABELS, KneeDataset, find_data_root, load_metadata\n"
    "from rsna_model import RSNADINOv2\n", ""
)
train_source = train_source.replace(
    'if __name__ == "__main__":\n    main()\n',
    '''# This notebook cell itself starts two forked workers, one per T4 GPU.
# No .py file is created or executed.
def _notebook_ddp_worker(local_rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    import sys
    sys.argv = ["notebook_train"]  # ignore Kaggle/IPython command-line arguments
    main()


if torch.cuda.device_count() < 2:
    raise RuntimeError("Enable the Kaggle accelerator with 2 × T4 GPUs before running this cell.")
torch.multiprocessing.start_processes(
    _notebook_ddp_worker, args=(2,), nprocs=2, join=True, start_method="fork"
)
'''
)


notebook = {
    "cells": [code_cell(data_source), code_cell(model_source), code_cell(train_source)],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
(HERE / "Kaggle_RSNA_3cells.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
