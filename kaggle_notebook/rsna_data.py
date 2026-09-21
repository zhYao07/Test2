"""RSNA Knee 数据加载与 DICOM 预处理模块。"""

from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from pydicom.errors import InvalidDicomError
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


KAGGLE_INPUT = Path("/kaggle/input")


def find_data_root(root=None):
    """Return the mounted Kaggle dataset containing the RSNA CSV files.

    Pass ``--data-root`` to train.py if more than one mounted dataset contains
    a train.csv.  This keeps the notebook independent of Kaggle dataset slugs.
    """
    if root is not None:
        root = Path(root)
        if (root / "train.csv").is_file():
            return root
        raise FileNotFoundError(f"RSNA data root has no train.csv: {root}")
    candidates = [path.parent for path in KAGGLE_INPUT.rglob("train.csv")
                  if (path.parent / "train_series.csv").is_file()
                  and (path.parent / "test.csv").is_file()
                  and (path.parent / "test_series.csv").is_file()]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            "Could not find RSNA data under /kaggle/input. Attach the competition "
            "data, or provide --data-root /kaggle/input/<your-data-dataset>."
        )
    raise RuntimeError(
        "Multiple possible RSNA data roots found: " + ", ".join(map(str, candidates)) +
        ". Provide --data-root explicitly."
    )


DEFAULT_ROOT = None
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
SLOTS = ["Axial_FS", "Axial_nonFS", "Coronal_FS", "Coronal_nonFS", "Sagittal_FS", "Sagittal_nonFS"]
SERIES_FEATURES = ["fluid_sensitive", "fat_suppression", "t1", "pd", "t2", "contrast_unknown", "te", "tr", "pixel_spacing", "slice_spacing", "coverage"]


def _count_dicoms(path):
    return sum(1 for _ in Path(path).glob("*.dcm"))


def _prepare_series_df(series_df, image_root):
    series_df = series_df.copy()
    series_df["SeriesPath"] = [Path(image_root) / str(study) / str(series) for study, series in zip(series_df.StudyInstanceUID, series_df.SeriesInstanceUID)]
    series_df["SeriesSlot"] = series_df.Anatomical_Plane + "_" + np.where(series_df.Fat_Suppression.eq(1), "FS", "nonFS")
    series_df["NumSlices"] = [_count_dicoms(path) for path in series_df.SeriesPath]
    return series_df


def load_metadata(root=DEFAULT_ROOT):
    root = find_data_root(root)
    train_df, test_df = pd.read_csv(root / "train.csv"), pd.read_csv(root / "test.csv")
    train_series_df = _prepare_series_df(pd.read_csv(root / "train_series.csv"), root / "train_series")
    test_series_df = _prepare_series_df(pd.read_csv(root / "test_series.csv"), root / "test_series")
    return train_df, train_series_df, test_df, test_series_df, pd.read_csv(root / "sample_submission.csv")


def get_sorted_dicom_info(series_path):
    """使用患者坐标排序，并把主切片方向统一为患者坐标轴的正方向。"""
    files = list(Path(series_path).glob("*.dcm"))
    headers = [pydicom.dcmread(file, stop_before_pixels=True) for file in files]
    orientation = np.asarray(headers[0].ImageOrientationPatient, dtype=np.float32)
    normal = np.cross(orientation[:3], orientation[3:])
    normal /= np.linalg.norm(normal)
    if normal[np.argmax(np.abs(normal))] < 0:
        normal = -normal
    positions = np.asarray([np.dot(np.asarray(header.ImagePositionPatient, dtype=np.float32), normal) for header in headers])
    order = np.argsort(positions)
    return [files[i] for i in order], positions[order], [headers[i] for i in order]


def sort_dicom_files(series_path):
    return get_sorted_dicom_info(series_path)[0]


def read_dicom(path):
    dicom = pydicom.dcmread(path)
    try:
        image = dicom.pixel_array.astype(np.float32)
    except ValueError as error:
        # 少数文件的头信息写着16位，但未压缩PixelData实际是8位；按真实字节数恢复。
        rows, columns = int(dicom.Rows), int(dicom.Columns)
        frames = int(getattr(dicom, "NumberOfFrames", 1) or 1)
        samples = int(getattr(dicom, "SamplesPerPixel", 1) or 1)
        pixel_count = rows * columns * frames * samples
        raw = dicom.PixelData
        transfer_syntax = getattr(getattr(dicom, "file_meta", None), "TransferSyntaxUID", None)
        uncompressed = transfer_syntax is None or not transfer_syntax.is_compressed
        if not uncompressed or frames != 1 or samples != 1 or len(raw) not in (pixel_count, pixel_count + 1):
            raise ValueError(f"Cannot decode DICOM {path}: {error}") from error
        dtype = np.int8 if int(getattr(dicom, "PixelRepresentation", 0) or 0) else np.uint8
        image = np.frombuffer(raw[:pixel_count], dtype=dtype).reshape(rows, columns).astype(np.float32)
    image = image * float(getattr(dicom, "RescaleSlope", 1)) + float(getattr(dicom, "RescaleIntercept", 0))
    if getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1":
        image = image.max() + image.min() - image
    return image


def _read_sampled_slices(files, indices):
    """读取采样切片；单张损坏时用空间上最近的可读切片替代。"""
    cache, invalid = {}, set()
    images = []
    for index in indices:
        index = int(index)
        candidates = sorted(range(len(files)), key=lambda candidate: abs(candidate - index))
        for candidate in candidates:
            if candidate in cache:
                image = cache[candidate]
                break
            if candidate in invalid:
                continue
            try:
                image = read_dicom(files[candidate])
                cache[candidate] = image
                break
            except (ValueError, OSError, EOFError, InvalidDicomError):
                invalid.add(candidate)
        else:
            raise RuntimeError(f"No decodable DICOM slices in series: {Path(files[0]).parent}")
        images.append(image)
    return np.stack(images)


def _sample_by_position(positions, num_slices):
    """在真实物理覆盖范围内均匀采样，而不是只按切片编号采样。"""
    targets = np.linspace(positions[0], positions[-1], num_slices)
    return np.abs(positions[:, None] - targets[None, :]).argmin(axis=0)


def _contrast_features(te, tr, fat_suppression):
    """依据 EDA 使用的 TE/TR 启发式生成 T1、PD、T2、未知四维标记。"""
    if fat_suppression:
        return [0, 1, 1, 0]
    if te <= 0 or tr <= 0:
        return [0, 0, 0, 1]
    if tr < 1000 and te < 30:
        return [1, 0, 0, 0]
    if te < 60:
        return [0, 1, 0, 0]
    return [0, 0, 1, 0]


def load_series(series_path, num_slices=24, image_size=224, fluid_sensitive=0, fat_suppression=0, return_features=False):
    """空间排序→物理位置采样→DICOM 解码→序列级归一化→等比例缩放。"""
    files, positions, headers = get_sorted_dicom_info(series_path)
    indices = _sample_by_position(positions, num_slices)
    volume = _read_sampled_slices(files, indices)

    foreground = volume[np.isfinite(volume) & (volume != 0)]
    low, high = np.percentile(foreground, [0.5, 99.5])
    volume = np.clip((volume - low) / (high - low + 1e-6), 0, 1).astype(np.float32)
    volume = torch.from_numpy(volume)[:, None]
    height, width = volume.shape[-2:]
    pad_h, pad_w = max(height, width) - height, max(height, width) - width
    volume = F.pad(volume, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
    volume = F.interpolate(volume, size=(image_size, image_size), mode="bilinear", align_corners=False)

    header = headers[len(headers) // 2]
    te, tr = float(getattr(header, "EchoTime", 0) or 0), float(getattr(header, "RepetitionTime", 0) or 0)
    spacing = np.asarray(getattr(header, "PixelSpacing", [0, 0]), dtype=np.float32)
    slice_spacing = float(np.median(np.abs(np.diff(positions)))) if len(positions) > 1 else 0
    coverage = float(abs(positions[-1] - positions[0])) if len(positions) > 1 else 0
    contrast = _contrast_features(te, tr, int(fat_suppression))
    features = torch.tensor([float(fluid_sensitive), float(fat_suppression), *contrast, te / 100, tr / 5000, float(spacing.mean()), slice_spacing / 10, coverage / 200], dtype=torch.float32)
    return (volume, features) if return_features else volume


class KneeDataset(Dataset):
    """Study 级数据集，输出 images [6,S,1,H,W] 和六个槽位的元数据。"""

    def __init__(self, study_df, series_df, num_slices=24, image_size=224):
        self.study_df = study_df.reset_index(drop=True)
        self.num_slices, self.image_size = num_slices, image_size
        self.series_groups = {uid: group.reset_index(drop=True) for uid, group in series_df.groupby("StudyInstanceUID")}

    def __len__(self):
        return len(self.study_df)

    def __getitem__(self, index):
        row, uid = self.study_df.iloc[index], self.study_df.iloc[index].StudyInstanceUID
        series_df = self.series_groups[uid]
        images = torch.zeros(len(SLOTS), self.num_slices, 1, self.image_size, self.image_size)
        slot_mask = torch.zeros(len(SLOTS))
        series_features = torch.zeros(len(SLOTS), len(SERIES_FEATURES))

        for slot_index, slot in enumerate(SLOTS):
            candidates = series_df[series_df.SeriesSlot == slot]
            if len(candidates):
                selected = candidates.sort_values(["Fluid_Sensitive", "NumSlices"], ascending=[False, False]).iloc[0]
                volume, features = load_series(selected.SeriesPath, self.num_slices, self.image_size, selected.Fluid_Sensitive, selected.Fat_Suppression, True)
                images[slot_index], series_features[slot_index], slot_mask[slot_index] = volume, features, 1

        values = pd.to_numeric(row[LABELS], errors="coerce").to_numpy(dtype=np.float32) if set(LABELS).issubset(row.index) else np.full(len(LABELS), np.nan, dtype=np.float32)
        valid = ~np.isnan(values)
        label_weight = np.where(valid, 2 * np.abs(values - 0.5), 0).astype(np.float32)
        return {
            "study_uid": uid,
            "images": images,
            "slot_mask": slot_mask,
            "series_features": series_features,
            "targets": torch.from_numpy(np.nan_to_num(values, nan=0)),
            "label_mask": torch.from_numpy(valid.astype(np.float32)),
            "label_weight": torch.from_numpy(label_weight),
        }

def build_datasets(root=DEFAULT_ROOT, num_slices=24, image_size=224):
    metadata = load_metadata(root)
    train_df, train_series_df, test_df, test_series_df, sample_submission = metadata
    train_dataset = KneeDataset(train_df, train_series_df, num_slices, image_size)
    test_dataset = KneeDataset(test_df, test_series_df, num_slices, image_size)
    return train_dataset, test_dataset, sample_submission, metadata


def build_dataloaders(root=DEFAULT_ROOT, batch_size=2, num_workers=4, num_slices=24, image_size=224):
    train_dataset, test_dataset, sample_submission, metadata = build_datasets(root, num_slices, image_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, sample_submission, metadata


if __name__ == "__main__":
    train_dataset, test_dataset, _, _ = build_datasets()
    sample = train_dataset[0]
    print("train studies:", len(train_dataset), "test studies:", len(test_dataset))
    for key, value in sample.items():
        print(key, value.shape if isinstance(value, torch.Tensor) else value)


__all__ = ["DEFAULT_ROOT", "LABELS", "SLOTS", "SERIES_FEATURES", "find_data_root", "load_metadata", "get_sorted_dicom_info", "sort_dicom_files", "read_dicom", "load_series", "KneeDataset", "build_datasets", "build_dataloaders"]
