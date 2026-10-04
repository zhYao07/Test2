# -*- coding: utf-8 -*-
"""RSNA Knee 数据加载与 140 mm physical crop 预处理模块。"""

import hashlib
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from pydicom.errors import InvalidDicomError
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


DEFAULT_ROOT = Path("/root/RSNA/rsna-knee-abnormality-detection")
DEFAULT_IMAGE_SIZE = 336
DEFAULT_CROP_MM = 140.0
# 一个 Study 预测 12 个标签；五个 slot 按方位/Fluid Sensitive 偏好选不同序列。
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
SLOTS = ["Sagittal_fluid", "Sagittal_second", "Coronal_fluid", "Coronal_second", "Axial"]
SLOT_SPECS = [("Sagittal", 1, 23), ("Sagittal", 0, 17), ("Coronal", 1, 15), ("Coronal", 0, 10), ("Axial", None, 15)]
CANDIDATE_BUDGETS = [spec[2] for spec in SLOT_SPECS]
CACHE_SCHEMA = 1
SERIES_SELECTION_VERSION = "header_quality_v1"
INTENSITY_AUGMENTATION_VERSION = "series_gamma_gain_v1"
TRAINING_VIEWS = ["original", "augmented"]
QUALITY_COLUMNS = ["quality_usable", "quality_header_errors", "quality_geometry_errors",
                   "quality_unique_slices", "quality_coverage_mm", "quality_pixel_spacing_mm",
                   "quality_slice_spacing_mm", "quality_fov_mm"]
# 每个序列对应 11 个数值元数据，顺序必须与 load_series 构造 features 的顺序一致。
SERIES_FEATURES = ["fluid_sensitive", "fat_suppression", "t1", "pd", "t2", "contrast_unknown", "te", "tr", "pixel_spacing", "slice_spacing", "coverage"] #翻译：
#MRI 序列特征：      液体敏感；         脂肪抑制； T1 加权；质子密度加权；T2 加权；对比或加权类型未知；回波时间；重复时间；像素间距；相邻切片间距；扫描覆盖范围。

def _count_dicoms(path):
    return sum(1 for _ in Path(path).glob("*.dcm"))


def _prepare_series_df(series_df, image_root):
    # 在原始序列表上补路径、槽位和切片数，方便后续按 Study/槽位选序列。
    series_df = series_df.copy()
    series_df["SeriesPath"] = [Path(image_root) / str(study) / str(series) for study, series in zip(series_df.StudyInstanceUID, series_df.SeriesInstanceUID)]
    series_df["NumSlices"] = [_count_dicoms(path) for path in series_df.SeriesPath]
    return series_df


def load_metadata(root=DEFAULT_ROOT):
    # Study 标签表与序列描述表分别读取；实际 DICOM 只在 __getitem__ 时按需解码。
    root = Path(root)
    train_df, test_df = pd.read_csv(root / "train.csv"), pd.read_csv(root / "test.csv")
    train_series_df = _prepare_series_df(pd.read_csv(root / "train_series.csv"), root / "train_series")
    test_series_df = _prepare_series_df(pd.read_csv(root / "test_series.csv"), root / "test_series")
    return train_df, train_series_df, test_df, test_series_df, pd.read_csv(root / "sample_submission.csv")


def get_sorted_dicom_info(series_path):

    """使用患者坐标排序，并把主切片方向统一为患者坐标轴的正方向。"""

    # 获取当前MRI序列文件夹下所有DICOM文件路径
    files = list(Path(series_path).glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"No DICOM files found in series: {series_path}")
    # 读取所有DICOM文件的头信息，但不读取像素数据，节省内存
    headers = [pydicom.dcmread(file, stop_before_pixels=True) for file in files]
    # 取第一张切片的ImageOrientationPatient，表示该图片在患者空间中的行方向和列方向   #ImagePositionPatient表示这一张图片左上角在患者三维空间的位置
    orientation = np.asarray(headers[0].ImageOrientationPatient, dtype=np.float32)
    # 前3个数表示图片行方向，后3个数表示图片列方向，两个方向叉乘得到切片法向量
    normal = np.cross(orientation[:3], orientation[3:])
    # 将法向量归一化，使其长度变为1，方便后续进行空间投影计算
    normal /= np.linalg.norm(normal)
    # 判断法向量主要方向是否为负，如果是则翻转，保证所有序列方向统一
    if normal[np.argmax(np.abs(normal))] < 0:
        # 反转法向量方向
        normal = -normal
    # 将每张切片的患者空间坐标投影到切片法向量方向，得到每张slice的真实空间位置
    positions = np.asarray([np.dot(np.asarray(header.ImagePositionPatient, dtype=np.float32), normal) for header in headers])
    # 根据真实空间位置从小到大排序，得到正确的slice顺序
    order = np.argsort(positions)
    # 按照排序后的索引，同时返回：
    # 1. 排序后的DICOM文件 2. 对应空间位置 3. 对应header信息
    return [files[i] for i in order], positions[order], [headers[i] for i in order]


def sort_dicom_files(series_path):
    # 只需要已排序文件列表时的便捷包装。
    return get_sorted_dicom_info(series_path)[0]

#读取一张DICOM MRI切片，将其转换成标准的二维numpy图像矩阵，并处理少数异常DICOM文件和灰度方向问题。
def read_dicom(path):
    dicom = pydicom.dcmread(path)
    # 尝试直接通过pydicom解析PixelData得到图像矩阵
    try:
        # pixel_array会自动根据DICOM头信息解析像素，并转换成numpy数组
        image = dicom.pixel_array.astype(np.float32)
    # 如果标准解析失败，进入手动恢复流程
    except ValueError as error:
        # 获取图像高度和宽度，后续用于手动reshape像素数据
        rows, columns = int(dicom.Rows), int(dicom.Columns)
        # 获取帧数，多帧DICOM没有该字段时默认认为只有1帧
        frames = int(getattr(dicom, "NumberOfFrames", 1) or 1)
        # 获取每个像素包含几个采样值，例如RGB可能是3，MRI通常是1
        samples = int(getattr(dicom, "SamplesPerPixel", 1) or 1)
        # 计算理论上的像素总数量
        pixel_count = rows * columns * frames * samples
        # 直接读取DICOM中的原始像素字节数据
        raw = dicom.PixelData
        # 获取DICOM传输格式，用于判断PixelData是否经过压缩
        transfer_syntax = getattr(getattr(dicom, "file_meta", None), "TransferSyntaxUID", None)
        # 判断当前DICOM是否为未压缩格式
        uncompressed = transfer_syntax is None or not transfer_syntax.is_compressed

        # 只有满足以下条件才允许手动解析：
        # 1. 未压缩
        # 2. 单帧
        # 3. 单通道
        # 4. 原始字节数量符合预期
        if not uncompressed or frames != 1 or samples != 1 or len(raw) not in (pixel_count, pixel_count + 1):

            # 如果情况复杂，无法安全恢复，则直接抛出异常
            raise ValueError(f"Cannot decode DICOM {path}: {error}") from error

        # 根据PixelRepresentation判断像素是否为有符号整数
        # 0代表unsigned，1代表signed
        dtype = np.int8 if int(getattr(dicom, "PixelRepresentation", 0) or 0) else np.uint8

        # 从原始字节创建numpy数组，并恢复成二维图像
        image = np.frombuffer(raw[:pixel_count], dtype=dtype).reshape(rows, columns).astype(np.float32)

    # 根据DICOM标准应用灰度线性变换
    # 输出值 = 原始像素值 × RescaleSlope + RescaleIntercept
    image = image * float(getattr(dicom, "RescaleSlope", 1)) + float(getattr(dicom, "RescaleIntercept", 0))

    # MONOCHROME1表示像素越大越黑，需要反转成普通医学图像亮度方向
    if getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1":
        # 使用最大值+最小值减去原像素，实现灰度反转
        image = image.max() + image.min() - image

    # 返回最终处理好的二维MRI切片
    return image

#读取指定的MRI切片，并且在遇到损坏DICOM时，用距离最近的正常切片替代，最后把所有slice组合成3D volume。注意这里的volume是一个numpy数组，shape为[S,H,W]，其中S是切片数，H和W是每张切片的高度和宽度。不是channle为3
def _read_sampled_slices(files, indices):
    """读取采样切片；单张损坏时用空间上最近的可读切片替代。"""
    # cache 避免重复解码同一张；invalid 避免对已知坏文件反复尝试。
    cache, invalid = {}, set()
    images = []
    for index in indices:
        index = int(index)
        # 按切片索引距离从近到远尝试，直到找到可解码的替代切片。
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
    # targets 是等间隔物理坐标；每个目标位置选最近的真实 DICOM 切片。
    targets = np.linspace(positions[0], positions[-1], num_slices)
    return np.abs(positions[:, None] - targets[None, :]).argmin(axis=0)


#从一个MRI序列的DICOM header中读取像素空间大小（PixelSpacing），用于后续把“毫米尺度的crop”转换成“像素尺度的crop”。
def _get_pixel_spacing(headers, series_path):

    """读取有效的行、列 PixelSpacing；优先使用堆栈中央切片。"""

    # 计算DICOM序列中间位置的slice索引，优先检查中间切片
    middle = len(headers) // 2
    # 按距离中间切片的远近排序，生成检查顺序
    search_order = sorted(range(len(headers)), key=lambda index: abs(index - middle))
    # 按照从中间到两边的顺序遍历每一张slice的header
    for index in search_order:

        # 获取当前slice的PixelSpacing字段，如果不存在则返回None
        value = getattr(headers[index], "PixelSpacing", None)

        # 如果当前slice没有PixelSpacing信息，跳过这一张slice
        if value is None or len(value) < 2:
            continue

        # 将PixelSpacing转换成numpy数组，只保留行方向和列方向的两个spacing值
        spacing = np.asarray(value[:2], dtype=np.float32)

        # 检查spacing是否有效：1. 所有值不是NaN/inf 2. 所有值大于0
        if np.all(np.isfinite(spacing)) and np.all(spacing > 0):
            # 返回有效的像素间距，单位通常是mm/pixel
            return spacing
    # 如果遍历所有slice都没有找到有效PixelSpacing，则说明该序列无法确定物理尺寸
    raise ValueError(f"Series has no valid PixelSpacing: {series_path}")


def physical_center_crop(volume, pixel_spacing, crop_mm=DEFAULT_CROP_MM):

    """按真实毫米尺度中心裁剪；原始 FOV 不足时先对称补零以保持尺度。"""

    # 检查输入volume是否为三维MRI数据，格式应该是 [slice数量, height, width]
    if volume.ndim != 3:
        # 如果不是三维数据，说明输入格式错误，直接报错
        raise ValueError(f"Expected volume [S,H,W], got shape {volume.shape}")
    # 检查裁剪尺寸是否有效，crop_mm必须是正数
    if not np.isfinite(crop_mm) or crop_mm <= 0:
        # 如果crop_mm为负数、0或者无穷/NaN，则无法进行裁剪
        raise ValueError(f"crop_mm must be positive, got {crop_mm}")
    # 将pixel_spacing转换成numpy数组，方便后续数学计算
    spacing = np.asarray(pixel_spacing, dtype=np.float32)

    # 检查PixelSpacing是否合法：必须包含两个值，并且不能为NaN、inf，同时两个方向都必须大于0
    if spacing.shape != (2,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        # PixelSpacing异常时无法从毫米转换到像素，因此停止处理
        raise ValueError(f"Invalid PixelSpacing: {pixel_spacing}")
    # 根据真实毫米大小计算需要裁剪的像素高度 # 公式：目标像素数量 = crop毫米数 / 每个像素代表的毫米数
    target_height = max(1, int(round(crop_mm / float(spacing[0]))))
    # 根据真实毫米大小计算需要裁剪的像素宽度
    target_width = max(1, int(round(crop_mm / float(spacing[1]))))
    # 获取当前MRI volume的空间尺寸
    # volume形状：[slice,height,width]
    height, width = volume.shape[-2:]

    # 计算高度方向需要补多少像素 # 如果原图高度已经大于目标高度，则不需要补
    pad_height = max(0, target_height - height)
    # 计算宽度方向需要补多少像素 # 如果原图宽度已经大于目标宽度，则不需要补
    pad_width = max(0, target_width - width)

    # 如果原始视野(FOV)小于目标裁剪区域，则先进行padding
    if pad_height or pad_width:

        # 原图不足140mm时进行左右/上下对称补零
        # 避免直接放大较小视野导致空间尺度错误
        volume = np.pad(
            volume,

            # 三个维度分别对应：
            # slice维度不补，height方向补，width方向补
            (
                (0, 0),
                # height方向上下对称补零
                (pad_height // 2, pad_height - pad_height // 2),
                # width方向左右对称补零
                (pad_width // 2, pad_width - pad_width // 2),
            ),

            # 使用0填充，代表MRI背景区域
            mode="constant",
            # 指定填充值为0
            constant_values=0,
        )
        # padding之后重新获取新的height和width
        height, width = volume.shape[-2:]

    # 计算中心裁剪开始位置的高度坐标
    top = (height - target_height) // 2

    # 计算中心裁剪开始位置的宽度坐标
    left = (width - target_width) // 2

    # 对所有slice使用同一个中心区域进行裁剪
    # 保证整个3D volume的空间位置保持一致
    return volume[:, top:top + target_height, left:left + target_width]


def _contrast_features(te, tr, fat_suppression):
    """依据 EDA 使用的 TE/TR 启发式生成 T1、PD、T2、未知四维标记。"""
    # 这些是经验规则推断的扫描对比度类别，不是人工标注的真值。
    if fat_suppression:
        return [0, 1, 1, 0]
    if te <= 0 or tr <= 0:
        return [0, 0, 0, 1]
    if tr < 1000 and te < 30:
        return [1, 0, 0, 0]
    if te < 60:
        return [0, 1, 0, 0]
    return [0, 0, 1, 0]


def _binary_flag(value):
    value = pd.to_numeric(value, errors="coerce")
    return int(value) if pd.notna(value) and value in (0, 1) else None


def read_series_quality(series_path):
    """只读 header 估计几何质量；不解码像素，不把 header 完整性当像素损坏率。"""
    files = sorted(Path(series_path).glob("*.dcm"))
    quality = dict.fromkeys(QUALITY_COLUMNS, None)
    quality.update(quality_usable=0, quality_header_errors=0, quality_geometry_errors=0,
                   quality_unique_slices=0)
    headers, normals, locations = [], [], []
    for file in files:
        try:
            header = pydicom.dcmread(file, stop_before_pixels=True)
        except (OSError, ValueError, EOFError, InvalidDicomError):
            quality["quality_header_errors"] += 1
            continue
        try:
            orientation = np.asarray(header.ImageOrientationPatient, dtype=np.float64)
            location = np.asarray(header.ImagePositionPatient, dtype=np.float64)
            if orientation.shape != (6,) or location.shape != (3,):
                raise ValueError("Invalid geometry dimensions")
            if not np.isfinite(orientation).all() or not np.isfinite(location).all():
                raise ValueError("Nonfinite geometry")
            normal = np.cross(orientation[:3], orientation[3:])
            length = float(np.linalg.norm(normal))
            if length < 1e-6:
                raise ValueError("Degenerate orientation")
            normals.append(normal / length)
            locations.append(location)
            headers.append(header)
        except (AttributeError, TypeError, ValueError):
            quality["quality_geometry_errors"] += 1
    if not headers:
        return quality
    normal = normals[0]
    positions = np.asarray(locations) @ normal
    # 合并相同位置的重建/重复切片，不以文件数量奖励重复图像。
    unique = np.unique(np.round(positions, 4))
    quality["quality_unique_slices"] = len(unique)
    quality["quality_coverage_mm"] = float(unique[-1] - unique[0])
    if len(unique) > 1:
        quality["quality_slice_spacing_mm"] = float(np.median(np.diff(unique)))
    ordered_headers = [headers[i] for i in np.argsort(positions)]
    try:
        spacing = _get_pixel_spacing(ordered_headers, series_path)
        quality["quality_pixel_spacing_mm"] = float(spacing.max())
        middle = ordered_headers[len(ordered_headers) // 2]
        rows, columns = int(middle.Rows), int(middle.Columns)
        if rows <= 0 or columns <= 0:
            raise ValueError("Invalid image matrix")
        quality["quality_fov_mm"] = float(min(rows * spacing[0], columns * spacing[1]))
    except (AttributeError, TypeError, ValueError):
        quality["quality_geometry_errors"] += 1
    aligned = bool(np.all(np.abs(np.asarray(normals) @ normal) >= 0.999))
    quality["quality_usable"] = int(
        len(unique) >= 2 and aligned and quality["quality_header_errors"] == 0 and
        quality["quality_geometry_errors"] == 0 and quality["quality_fov_mm"] is not None
    )
    return quality


def _cached_series_quality(series_path, cache_dir):
    directory = Path(series_path)
    cache_path = None
    if cache_dir is not None:
        files = []
        for file in sorted(directory.glob("*.dcm")):
            stat = file.stat()
            files.append((file.name, stat.st_size, stat.st_mtime_ns))
        signature = json.dumps([SERIES_SELECTION_VERSION, str(directory.resolve()), files])
        cache_path = Path(cache_dir) / (hashlib.sha256(signature.encode()).hexdigest() + ".json")
        if cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if (isinstance(cached, dict) and cached.get("version") == SERIES_SELECTION_VERSION and
                    isinstance(cached.get("quality"), dict) and all(k in cached["quality"] for k in QUALITY_COLUMNS)):
                    return cached["quality"]
            except (OSError, ValueError, KeyError, TypeError):
                pass
    quality = read_series_quality(directory)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(dict(version=SERIES_SELECTION_VERSION, quality=quality), allow_nan=False),
                                 encoding="utf-8")
            os.replace(temporary, cache_path)
        finally:
            temporary.unlink(missing_ok=True)
    return quality


def add_series_quality(series_df, cache_dir=None, workers=8):
    """训练由 rank 0 扫描/读取缓存后广播；推理只扫描当前 GPU 分配的序列。"""
    if workers < 1:
        raise ValueError("Quality workers must be positive")
    rows = series_df.copy()
    print(f"Series quality: indexing {len(rows)} series (header-only, workers={workers})", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = []
        for index, quality in enumerate(pool.map(lambda p: _cached_series_quality(p, cache_dir), rows.SeriesPath), 1):
            results.append(quality)
            if index % 1000 == 0 or index == len(rows):
                print(f"Series quality: {index}/{len(rows)}", flush=True)
    for column in QUALITY_COLUMNS:
        rows[column] = [quality[column] for quality in results]
    return rows


def make_series_selection_config(train_series_df, coverage_quantile=0.2):
    """只使用弱标签训练 Study 的有效几何统计，确定各平面的覆盖参考阈值。"""
    if not 0 <= coverage_quantile <= 1:
        raise ValueError("Coverage quantile must be in [0,1]")
    thresholds = {}
    for plane in ("Sagittal", "Coronal", "Axial"):
        values = train_series_df.loc[
            train_series_df.Anatomical_Plane.eq(plane) & train_series_df.quality_usable.eq(1),
            "quality_coverage_mm",
        ]
        values = pd.to_numeric(values, errors="coerce")
        values = values[np.isfinite(values) & (values > 0)]
        thresholds[plane] = float(values.quantile(coverage_quantile)) if len(values) else 0.0
    config = dict(version=SERIES_SELECTION_VERSION, coverage_quantile=float(coverage_quantile),
                  coverage_thresholds=thresholds)
    validate_series_selection_config(config)
    return config


def validate_series_selection_config(config):
    if not isinstance(config, dict) or config.get("version") != SERIES_SELECTION_VERSION:
        raise ValueError("v14 requires its checkpoint's header-quality series selection config")
    quantile = config.get("coverage_quantile")
    if not isinstance(quantile, (int, float)) or not np.isfinite(quantile) or not 0 <= quantile <= 1:
        raise ValueError("Invalid coverage quantile")
    thresholds = config.get("coverage_thresholds", {})
    if not isinstance(thresholds, dict):
        raise ValueError("Invalid coverage thresholds")
    for plane in ("Sagittal", "Coronal", "Axial"):
        value = thresholds.get(plane)
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError(f"Invalid coverage threshold for {plane}")


def _series_quality_rank(record, config, crop_mm, image_size):
    threshold = config["coverage_thresholds"][record.Anatomical_Plane]
    coverage = float(record.quality_coverage_mm)
    # 覆盖/FOV 达标后不再奖励更大范围，避免优先选大视野而低分辨率的序列。
    coverage_score = min(coverage / threshold, 1.0) if threshold > 0 else 1.0
    fov_score = min(float(record.quality_fov_mm) / crop_mm, 1.0)
    effective_spacing = max(float(record.quality_pixel_spacing_mm), crop_mm / image_size)
    return (coverage_score, fov_score, -effective_spacing,
            -float(record.quality_slice_spacing_mm), int(record.quality_unique_slices))


def select_slot_series(rows, series_selection=None, crop_mm=DEFAULT_CROP_MM, image_size=DEFAULT_IMAGE_SIZE):
    """v10 候选范围不变；配置存在时仅在同一集合内按 header 几何质量排序。"""
    if series_selection is not None:
        validate_series_selection_config(series_selection)
        if any(column not in rows for column in QUALITY_COLUMNS):
            raise ValueError("Call add_series_quality before v14 series selection")
    used, selected = set(), []
    for plane, preference, _ in SLOT_SPECS:
        candidates = rows[(rows.Anatomical_Plane == plane) & (rows.NumSlices > 0)]
        candidates = candidates[~candidates.SeriesInstanceUID.astype(str).isin(used)]
        if preference is not None:
            preferred = candidates[candidates.Fluid_Sensitive.map(_binary_flag).eq(preference)]
            if len(preferred):
                candidates = preferred
        record = candidates.iloc[0] if len(candidates) else None
        if series_selection is not None and len(candidates) > 1:
            usable = candidates[candidates.quality_usable.eq(1)]
            if len(usable):
                # 同分保留首个元素；全部不可用时也回退 v10 CSV 顺序。
                record = max((row for _, row in usable.iterrows()),
                             key=lambda row: _series_quality_rank(row, series_selection, crop_mm, image_size))
        selected.append(record)
        if record is not None:
            used.add(str(record.SeriesInstanceUID))
    return selected


def series_selection_report(series_df, series_selection, crop_mm=DEFAULT_CROP_MM, image_size=DEFAULT_IMAGE_SIZE):
    records = []
    for uid, rows in series_df.groupby("StudyInstanceUID", sort=False):
        original = select_slot_series(rows)
        selected = select_slot_series(rows, series_selection, crop_mm, image_size)
        for slot, old, new in zip(SLOTS, original, selected):
            old_uid = str(old.SeriesInstanceUID) if old is not None else ""
            new_uid = str(new.SeriesInstanceUID) if new is not None else ""
            record = dict(StudyInstanceUID=str(uid), slot=slot, original_series_uid=old_uid,
                          selected_series_uid=new_uid, changed=int(old_uid != new_uid))
            for prefix, series in (("original", old), ("selected", new)):
                for column in QUALITY_COLUMNS:
                    record[f"{prefix}_{column}"] = series[column] if series is not None else None
            records.append(record)
    return pd.DataFrame(records)


def load_series(series_path, num_windows, image_size=DEFAULT_IMAGE_SIZE, crop_mm=DEFAULT_CROP_MM,
                fluid_sensitive=0, fat_suppression=0, span_lo=0.02, span_hi=0.98):
    """返回唯一切片缓存、窗口索引、原始位置及元数据；不跨序列构造窗口。"""
    if num_windows < 1 or not 0 <= span_lo < span_hi <= 1:
        raise ValueError("Invalid candidate count or span")
    files, positions, headers = get_sorted_dicom_info(series_path)
    first = int(len(files) * span_lo)
    last = max(first, int(len(files) * span_hi) - 1)
    # 覆盖 2%-98% 的原始索引范围；范围内继续采用 v1 物理位置采样。
    anchors = first + _sample_by_position(positions[first:last + 1], num_windows)
    neighbors = np.clip(anchors[:, None] + np.array([-1, 0, 1]), 0, len(files) - 1)
    needed = np.unique(neighbors)
    spacing = _get_pixel_spacing(headers, series_path)
    volume = physical_center_crop(_read_sampled_slices(files, needed), spacing, crop_mm)
    anchor_volume = volume[np.searchsorted(needed, anchors)]
    foreground = anchor_volume[np.isfinite(anchor_volume) & (anchor_volume != 0)]
    if not foreground.size:
        raise ValueError(f"Series contains no finite non-zero anchor pixels after crop: {series_path}")
    low, high = np.percentile(foreground, [0.5, 99.5])
    volume = np.clip((volume - low) / (high - low + 1e-6), 0, 1).astype(np.float32)
    slices = F.interpolate(torch.from_numpy(volume)[:, None], size=(image_size, image_size),
                           mode="bilinear", align_corners=False)[:, 0]
    window_indices = torch.from_numpy(np.searchsorted(needed, neighbors))
    coverage = float(positions[-1] - positions[0])
    anchor_positions = (2 * (positions[anchors] - positions[0]) / coverage - 1
                        if coverage > 0 else np.zeros(num_windows))
    header = headers[len(headers) // 2]
    te, tr = float(getattr(header, "EchoTime", 0) or 0), float(getattr(header, "RepetitionTime", 0) or 0)
    slice_spacing = float(np.median(np.abs(np.diff(positions)))) if len(positions) > 1 else 0
    features = torch.tensor([
        float(fluid_sensitive), float(fat_suppression), *_contrast_features(te, tr, int(fat_suppression)),
        te / 100, tr / 5000, float(spacing.mean()), slice_spacing / 10, coverage / 200,
    ], dtype=torch.float32)
    return slices, window_indices, torch.as_tensor(anchor_positions, dtype=torch.float32), features


def sample_training_windows(slot_indices, count, rng):
    """每个有效 slot 至少一个；剩余从未选候选随机抽取，候选不足时才重复。"""
    slot_indices = np.asarray(slot_indices)
    slots = np.unique(slot_indices)
    if not len(slots) or count < len(slots):
        raise ValueError("Training window count must cover every valid slot")
    mandatory = np.array([rng.choice(np.flatnonzero(slot_indices == slot)) for slot in slots], dtype=np.int64)
    remaining = np.setdiff1d(np.arange(len(slot_indices)), mandatory)
    rng.shuffle(remaining)
    take = min(count - len(mandatory), len(remaining))
    chosen = np.concatenate([mandatory, remaining[:take]])
    if len(chosen) < count:
        chosen = np.concatenate([chosen, rng.choice(len(slot_indices), count - len(chosen), replace=True)])
    return chosen


def make_intensity_augmentation_config(probability=0.5, gamma_range=(0.9, 1.1), gain_range=(0.9, 1.1)):
    config = dict(version=INTENSITY_AUGMENTATION_VERSION, probability=float(probability),
                  gamma_range=list(gamma_range), gain_range=list(gain_range))
    validate_intensity_augmentation_config(config)
    return config


def validate_intensity_augmentation_config(config):
    if not isinstance(config, dict) or config.get("version") != INTENSITY_AUGMENTATION_VERSION:
        raise ValueError("v17 requires its checkpoint's series intensity augmentation config")
    probability = config.get("probability")
    if not isinstance(probability, (int, float)) or not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Intensity augmentation probability must be in [0,1]")
    for name in ("gamma_range", "gain_range"):
        values = config.get(name)
        if (not isinstance(values, (list, tuple)) or len(values) != 2 or
            any(not isinstance(v, (int, float)) or not np.isfinite(v) for v in values) or
            not 0 < values[0] <= values[1]):
            raise ValueError(f"{name} must contain two finite positive bounds in ascending order")


def augment_series_intensity(images, slot_indices, study_uid, epoch, seed, config):
    """每序列一次 Gamma/gain；独立随机流保持抽样、worker 数及 rank 不变性。"""
    validate_intensity_augmentation_config(config)
    if images.ndim != 4 or images.shape[1] != 3 or slot_indices.shape != (len(images),):
        raise ValueError("Expected images [N,3,H,W] and slot_indices [N]")
    if config["probability"] == 0 or not len(images):
        return images
    # 不修改传入张量或 mmap 输入缓存；重复切片在同一 slot 中使用相同变换。
    output = images.clone()
    for slot in torch.unique(slot_indices).tolist():
        digest = hashlib.blake2b(
            f"{int(seed)}|{int(epoch)}|{study_uid}|{int(slot)}|intensity".encode("utf-8"),
            digest_size=8,
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest, "little"))
        if rng.random() >= config["probability"]:
            continue
        gamma = float(rng.uniform(*config["gamma_range"]))
        gain = float(rng.uniform(*config["gain_range"]))
        mask = slot_indices.eq(slot)
        output[mask] = (images[mask].pow(gamma) * gain).clamp(0, 1)
    return output


class KneeDataset(Dataset):
    """v17_v1 同一批随机24窗口返回原图与增强视图，验证/推理仅用原图。"""
    def __init__(self, study_df, series_df, image_size=DEFAULT_IMAGE_SIZE, crop_mm=DEFAULT_CROP_MM,
                 train=False, train_windows=24, span_lo=0.02, span_hi=0.98, cache_dir=None, seed=42,
                 series_selection=None, intensity_augmentation=None):
        if train_windows < len(SLOTS) or image_size < 1 or not 0 <= span_lo < span_hi <= 1:
            raise ValueError("train_windows must be >=5; image_size/span must be valid")
        self.study_df = study_df.reset_index(drop=True)
        self.series_groups = {str(uid): group.reset_index(drop=True)
                              for uid, group in series_df.groupby("StudyInstanceUID")}
        self.image_size, self.crop_mm = image_size, crop_mm
        self.train, self.train_windows, self.seed = train, train_windows, seed
        validate_series_selection_config(series_selection)
        self.series_selection = dict(series_selection, coverage_thresholds=dict(series_selection["coverage_thresholds"]))
        intensity_config = (make_intensity_augmentation_config() if intensity_augmentation is None
                            else intensity_augmentation)
        validate_intensity_augmentation_config(intensity_config)
        self.intensity_augmentation = dict(intensity_config,
                                          gamma_range=list(intensity_config["gamma_range"]),
                                          gain_range=list(intensity_config["gain_range"]))
        self.span_lo, self.span_hi = span_lo, span_hi
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        # persistent_workers 的 Dataset 副本共享 epoch，抽样不依赖 worker 数或访问顺序。
        self.epoch = torch.zeros((), dtype=torch.int64).share_memory_()

    def __len__(self):
        return len(self.study_df)

    def set_epoch(self, epoch):
        self.epoch.fill_(epoch)

    def _cache_path(self, uid, selected):
        if self.cache_dir is None:
            return None
        sources = []
        for record in selected:
            if record is None:
                sources.append(None)
                continue
            directory = Path(record.SeriesPath).resolve()
            files = []
            for file in sorted(directory.glob("*.dcm")):
                stat = file.stat()
                files.append((file.name, stat.st_size, stat.st_mtime_ns))
            sources.append((str(directory), _binary_flag(record.Fluid_Sensitive),
                            _binary_flag(record.Fat_Suppression), files))
        signature = json.dumps([CACHE_SCHEMA, str(uid), self.image_size, self.crop_mm,
                                self.span_lo, self.span_hi, SLOT_SPECS, sources], ensure_ascii=False)
        return self.cache_dir / (hashlib.sha256(signature.encode("utf-8")).hexdigest() + ".pt")

    def _build_bank(self, selected):
        pixels, indices, positions, slots = [], [], [], []
        metadata = torch.zeros(len(SLOTS), len(SERIES_FEATURES))
        slot_mask = torch.zeros(len(SLOTS))
        offset = 0
        for slot, (record, (_, _, budget)) in enumerate(zip(selected, SLOT_SPECS)):
            if record is None:
                continue
            slices, windows, depth, features = load_series(
                record.SeriesPath, budget, self.image_size, self.crop_mm,
                _binary_flag(record.Fluid_Sensitive) or 0, _binary_flag(record.Fat_Suppression) or 0,
                self.span_lo, self.span_hi,
            )
            pixels.append(slices)
            indices.append(windows + offset)
            positions.append(depth)
            slots.append(torch.full((budget,), slot, dtype=torch.long))
            metadata[slot], slot_mask[slot] = features, 1
            offset += len(slices)
        if not pixels:
            raise ValueError("Study has no usable series in the five slots")
        return dict(cache_schema=CACHE_SCHEMA, slice_images=torch.cat(pixels),
                    window_indices=torch.cat(indices), window_positions=torch.cat(positions),
                    window_slot_indices=torch.cat(slots), series_features=metadata, slot_mask=slot_mask)

    def candidate_bank(self, uid):
        selected = select_slot_series(self.series_groups[str(uid)], self.series_selection, self.crop_mm, self.image_size)
        path = self._cache_path(uid, selected)
        if path is not None and path.is_file():
            bank = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            if bank.get("cache_schema") != CACHE_SCHEMA:
                raise ValueError(f"Invalid input cache: {path}")
            return bank
        bank = self._build_bank(selected)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
            try:
                torch.save(bank, temporary)
                try:
                    os.replace(temporary, path)
                except OSError:
                    # 另一 DDP worker 已经写入并 mmap 时，Windows 可能拒绝覆盖。
                    if not path.is_file():
                        raise
            finally:
                temporary.unlink(missing_ok=True)
        return bank

    def __getitem__(self, index):
        row = self.study_df.iloc[index]
        uid = row.StudyInstanceUID
        bank = self.candidate_bank(uid)
        count = len(bank["window_indices"])
        slots = bank["window_slot_indices"].numpy()
        if self.train:
            digest = hashlib.blake2b(f"{self.seed}|{int(self.epoch)}|{uid}".encode(), digest_size=8).digest()
            rng = np.random.default_rng(int.from_bytes(digest, "little"))
            chosen = sample_training_windows(slots, self.train_windows, rng)
        else:
            chosen = np.arange(count)
        # 序列内按原始物理位置排序；不把随机抽取次序当作空间次序。
        positions = bank["window_positions"].numpy()
        chosen = chosen[np.lexsort((chosen, positions[chosen], slots[chosen]))]
        chosen = torch.as_tensor(chosen, dtype=torch.long)
        images = bank["slice_images"][bank["window_indices"][chosen]]
        augmented_images = None
        if self.train:
            # 同一批窗口保留原图，另外生成增强图；两组各自作为独立 Study 视图。
            augmented_images = augment_series_intensity(images, bank["window_slot_indices"][chosen], uid,
                                                        int(self.epoch), self.seed, self.intensity_augmentation)
        values = (pd.to_numeric(row[LABELS], errors="coerce").to_numpy(dtype=np.float32)
                  if set(LABELS).issubset(row.index) else np.full(len(LABELS), np.nan, dtype=np.float32))
        valid = ~np.isnan(values)
        sample = dict(study_uid=str(uid), images=images,
                    window_positions=bank["window_positions"][chosen],
                    window_slot_indices=bank["window_slot_indices"][chosen],
                    slot_mask=bank["slot_mask"], series_features=bank["series_features"],
                    candidate_count=torch.tensor(count),
                    targets=torch.from_numpy(np.nan_to_num(values, nan=0)),
                    label_mask=torch.from_numpy(valid.astype(np.float32)),
                    label_weight=torch.from_numpy(np.where(valid, 2 * np.abs(values - 0.5), 0).astype(np.float32)))
        if self.train:
            sample["augmented_images"] = augmented_images
        return sample


def collate_studies(samples):
    """训练按 Study 展开原图/增强两个视图，batch 索引隔离两组注意力。"""
    expanded, view_names = [], []
    for sample in samples:
        expanded.append(sample)
        view_names.append("original")
        if "augmented_images" in sample:
            expanded.append(dict(sample, images=sample["augmented_images"]))
            view_names.append("augmented")
    batch = {key: torch.cat([sample[key] for sample in expanded])
             for key in ("images", "window_positions", "window_slot_indices")}
    batch["window_batch_indices"] = torch.cat([
        torch.full((len(sample["images"]),), i, dtype=torch.long) for i, sample in enumerate(expanded)
    ])
    for key in ("slot_mask", "series_features", "targets", "label_mask", "label_weight", "candidate_count"):
        batch[key] = torch.stack([sample[key] for sample in expanded])
    batch["study_uid"] = [sample["study_uid"] for sample in expanded]
    batch["view_names"] = view_names
    batch["source_study_count"] = len(samples)
    batch["source_candidate_count"] = sum(int(sample["candidate_count"]) for sample in samples)
    return batch


def build_datasets(root=DEFAULT_ROOT, **kwargs):
    metadata = load_metadata(root)
    train_df, train_series_df, test_df, test_series_df, sample_submission = metadata
    quality_cache = Path(kwargs["cache_dir"]) / "series_quality" if kwargs.get("cache_dir") is not None else None
    train_series_df = add_series_quality(train_series_df, quality_cache)
    test_series_df = add_series_quality(test_series_df, quality_cache)
    if kwargs.get("series_selection") is None:
        weak_uids = train_df.loc[~train_df[LABELS].notna().all(axis=1), "StudyInstanceUID"]
        kwargs["series_selection"] = make_series_selection_config(train_series_df[train_series_df.StudyInstanceUID.isin(weak_uids)])
    metadata = (train_df, train_series_df, test_df, test_series_df, sample_submission)
    return (KneeDataset(train_df, train_series_df, train=True, **kwargs),
            KneeDataset(test_df, test_series_df, train=False, **kwargs), sample_submission, metadata)


__all__ = ["DEFAULT_ROOT", "DEFAULT_IMAGE_SIZE", "DEFAULT_CROP_MM", "LABELS", "SLOTS", "SLOT_SPECS",
           "CANDIDATE_BUDGETS", "load_metadata", "get_sorted_dicom_info", "read_dicom", "physical_center_crop",
           "load_series", "select_slot_series", "sample_training_windows", "KneeDataset", "collate_studies",
           "SERIES_SELECTION_VERSION", "QUALITY_COLUMNS", "read_series_quality", "add_series_quality",
           "make_series_selection_config", "validate_series_selection_config", "series_selection_report",
           "INTENSITY_AUGMENTATION_VERSION", "make_intensity_augmentation_config",
           "validate_intensity_augmentation_config", "augment_series_intensity", "TRAINING_VIEWS"]
