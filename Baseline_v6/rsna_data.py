"""RSNA Knee 数据加载与 140 mm physical crop 预处理模块。"""

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
DEFAULT_SLOT_WINDOW_BUDGET = 32
# 一个 Study 预测 12 个标签；SLOTS 固定六种扫描方向/脂肪抑制组合。
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
SLOTS = ["Axial_FS", "Axial_nonFS", "Coronal_FS", "Coronal_nonFS", "Sagittal_FS", "Sagittal_nonFS"]
# 每个序列对应 11 个数值元数据，顺序必须与 load_series 构造 features 的顺序一致。
SERIES_FEATURES = ["fluid_sensitive", "fat_suppression", "t1", "pd", "t2", "contrast_unknown", "te", "tr", "pixel_spacing", "slice_spacing", "coverage"] #翻译：
#MRI 序列特征：      液体敏感；         脂肪抑制； T1 加权；质子密度加权；T2 加权；对比或加权类型未知；回波时间；重复时间；像素间距；相邻切片间距；扫描覆盖范围。

def _prepare_series_df(series_df, image_root):
    # 保留全部序列，不再为择一规则扫描切片数。
    series_df = series_df.copy()
    series_df["SeriesPath"] = [Path(image_root) / str(study) / str(series) for study, series in zip(series_df.StudyInstanceUID, series_df.SeriesInstanceUID)]
    series_df["SeriesSlot"] = series_df.Anatomical_Plane + "_" + np.where(series_df.Fat_Suppression.eq(1), "FS", "nonFS")
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


def allocate_slot_windows(slice_counts, budget=DEFAULT_SLOT_WINDOW_BUDGET):
    """均衡分配 slot 总窗口预算；每序列至少一组，且不超过原始切片数。"""
    capacities = np.asarray(slice_counts, dtype=np.int64)
    if budget < 1 or capacities.ndim != 1 or np.any(capacities < 1):
        raise ValueError("Window budget and series slice counts must be positive")
    if len(capacities) > budget:
        raise ValueError(f"Slot has {len(capacities)} series but window budget is {budget}; "
                         "increase --slot-window-budget to retain every series")
    quotas = np.zeros_like(capacities)
    remaining = min(int(capacities.sum()), budget)
    # 按 SeriesInstanceUID 排序后的序列轮流领取一个窗口；短序列用完后自动跳过。
    while remaining:
        for index, capacity in enumerate(capacities):
            if quotas[index] < capacity:
                quotas[index] += 1
                remaining -= 1
                if not remaining:
                    break
    return quotas


def select_unique_anchors(positions, count):
    """物理等间隔目标→不同原始索引；为后续 anchor 留足切片，避免最近邻重复。"""
    positions = np.asarray(positions)
    if positions.ndim != 1 or not 1 <= count <= len(positions):
        raise ValueError("Anchor count must be between one and the number of original slices")
    if not np.isfinite(positions).all() or np.any(np.diff(positions) < 0):
        raise ValueError("Slice positions must be finite and sorted")
    if positions[-1] == positions[0]:
        # 物理位置无法区分时按原始排序索引均匀选，不重复索引。
        return np.rint(np.linspace(0, len(positions) - 1, count)).astype(np.int64) if count > 1 else np.array([len(positions) // 2])
    targets = (np.linspace(positions[0], positions[-1], count) if count > 1
               else np.array([(positions[0] + positions[-1]) / 2]))
    anchors, lower = [], 0
    for index, target in enumerate(targets):
        upper = len(positions) - (count - index)
        chosen = lower + int(np.abs(positions[lower:upper + 1] - target).argmin())
        anchors.append(chosen)
        lower = chosen + 1
    return np.asarray(anchors, dtype=np.int64)


def make_adjacent_windows(volume, positions, num_windows):
    """每个不同 anchor 使用原始 [i-1,i,i+1]；仅序列边界重复端点。"""
    if not len(volume) or len(positions) != len(volume) or num_windows < 1:
        raise ValueError("Expected non-empty slices/positions and a positive window count")
    anchors = select_unique_anchors(positions, min(num_windows, len(volume)))
    indices = np.clip(anchors[:, None] + np.array([-1, 0, 1]), 0, len(volume) - 1)
    indices = torch.as_tensor(indices, dtype=torch.long, device=volume.device)
    images = volume[indices, 0]  # [窗口数,3,H,W]，不同序列绝不拼接成邻居。
    coverage = float(positions[-1] - positions[0])
    normalized = 2 * (np.asarray(positions)[anchors] - positions[0]) / coverage - 1 if coverage > 0 else np.zeros(len(anchors))
    return images, torch.as_tensor(normalized, dtype=torch.float32)


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


def load_series(
    series_path,
    image_size=DEFAULT_IMAGE_SIZE,
    crop_mm=DEFAULT_CROP_MM,
    fluid_sensitive=0,
    fat_suppression=0,
    return_features=False,
    num_windows=DEFAULT_SLOT_WINDOW_BUDGET,
    dicom_info=None,
):

    """沿用 v5 整序列强度统计；按配额选不同 anchor，取原始相邻三张。"""

    # 获取当前MRI序列所有DICOM文件，并按照患者真实空间位置排序
    files, positions, headers = get_sorted_dicom_info(series_path) if dicom_info is None else dicom_info
    # 每张原始切片读取一次；坏文件仍沿用 v1 的最近可读切片替代策略。
    volume = _read_sampled_slices(files, np.arange(len(files)))

    # 获取该MRI序列的像素间距(mm/pixel)，用于物理尺度裁剪
    spacing = _get_pixel_spacing(headers, series_path)
    # 按真实毫米尺寸进行中心裁剪，例如裁剪140mm×140mm区域
    volume = physical_center_crop(volume, spacing, crop_mm)

    # 提取volume中有效的非零像素，用于计算MRI强度范围   # 排除背景0值，避免背景影响归一化
    foreground = volume[np.isfinite(volume) & (volume != 0)]
    # 如果裁剪后没有有效像素，说明该序列异常，无法继续处理
    if not foreground.size:
        raise ValueError(f"Series contains no finite non-zero pixels after crop: {series_path}")
    # 计算前景像素的0.5%和99.5%分位数作为强度截断范围 # 避免极端噪声影响归一化
    low, high = np.percentile(foreground, [0.5, 99.5])
    # 将MRI强度归一化到0~1范围，并限制异常值
    volume = np.clip((volume - low) / (high - low + 1e-6), 0, 1).astype(np.float32)
    # 将numpy数组转换为PyTorch tensor
    volume = torch.from_numpy(volume)[:, None]
    # 只 resize 选中的窗口；整序列分位数、crop 和归一化与 v5 一致。
    images, group_positions = make_adjacent_windows(volume, positions, num_windows)
    images = F.interpolate(
        images,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False
    )
    # 选择中间slice的header作为整个series的代表metadata
    header = headers[len(headers) // 2]
    # 获取Echo Time和Repetition Time
    # MRI序列的重要扫描参数
    te, tr = float(getattr(header, "EchoTime", 0) or 0), float(getattr(header, "RepetitionTime", 0) or 0)
    # 根据相邻slice空间位置差计算slice间距
    # 使用median减少异常slice影响
    slice_spacing = float(np.median(np.abs(np.diff(positions)))) if len(positions) > 1 else 0


    # 计算整个MRI序列覆盖的物理长度(mm) # 即第一张slice到最后一张slice之间的距离
    coverage = float(abs(positions[-1] - positions[0])) if len(positions) > 1 else 0
    # 根据TE/TR以及脂肪抑制信息，推断MRI对比类型
    # 输出T1、PD、T2、unknown等标记
    contrast = _contrast_features(te, tr, int(fat_suppression))
    # 将MRI序列相关信息组成11维feature向量
    # 顺序必须与SERIES_FEATURES保持一致
    features = torch.tensor(
        [
            float(fluid_sensitive),
            float(fat_suppression),
            *contrast,
            te / 100,
            tr / 5000,
            float(spacing.mean()),
            slice_spacing / 10,
            coverage / 200
        ],
        dtype=torch.float32
    )
    # 根据需求决定是否同时返回图像和metadata feature
    # 训练模型时可能需要两者
    return (images, features, group_positions) if return_features else images


class KneeDataset(Dataset):
    """每个 Study 保留六个 slot 的全部序列，图像按有效三张组合紧凑存放。"""

    def __init__(self, study_df, series_df, image_size=DEFAULT_IMAGE_SIZE, crop_mm=DEFAULT_CROP_MM,
                 slot_window_budget=DEFAULT_SLOT_WINDOW_BUDGET):
        if slot_window_budget < 1:
            raise ValueError("slot_window_budget must be positive")
        self.study_df = study_df.reset_index(drop=True)
        self.image_size, self.crop_mm = image_size, crop_mm
        self.slot_window_budget = slot_window_budget
        self.series_groups = {uid: group.reset_index(drop=True)
                              for uid, group in series_df.groupby("StudyInstanceUID")}

    def __len__(self):
        return len(self.study_df)

    def __getitem__(self, index):
        row = self.study_df.iloc[index]
        uid = row.StudyInstanceUID
        if uid not in self.series_groups:
            raise ValueError(f"No series metadata for study {uid}")
        series_df = self.series_groups[uid]
        images, metadata, positions, counts, slots = [], [], [], [], []
        slot_mask = torch.zeros(len(SLOTS), dtype=torch.bool)
        for slot_index, slot in enumerate(SLOTS):
            # 序列顺序只用于可复现；全部保留，不按 Fluid_Sensitive 或 NumSlices 择一。
            candidates = series_df[series_df.SeriesSlot == slot].sort_values("SeriesInstanceUID")
            selected_series = list(candidates.itertuples(index=False))
            # 先读取各序列排序后的头信息，分配配额；传给 loader 避免重复读取头信息。
            dicom_infos = [get_sorted_dicom_info(selected.SeriesPath) for selected in selected_series]
            quotas = allocate_slot_windows([len(info[0]) for info in dicom_infos], self.slot_window_budget)
            for selected, info, quota in zip(selected_series, dicom_infos, quotas):
                groups, features, group_positions = load_series(
                    selected.SeriesPath, image_size=self.image_size, crop_mm=self.crop_mm,
                    fluid_sensitive=selected.Fluid_Sensitive,
                    fat_suppression=selected.Fat_Suppression, return_features=True,
                    num_windows=int(quota), dicom_info=info,
                )
                images.append(groups)
                metadata.append(features)
                positions.append(group_positions)
                counts.append(len(groups))
                slots.append(slot_index)
                slot_mask[slot_index] = True
        if not images:
            raise ValueError(f"No supported MRI series for study {uid}")
        values = (pd.to_numeric(row[LABELS], errors="coerce").to_numpy(dtype=np.float32)
                  if set(LABELS).issubset(row.index) else np.full(len(LABELS), np.nan, dtype=np.float32))
        valid = ~np.isnan(values)
        return {
            "study_uid": uid,
            "images": torch.cat(images),                 # [所有序列组数之和,3,H,W]
            "group_positions": torch.cat(positions),    # 每个组在所属序列内的物理深度
            "group_counts": torch.tensor(counts, dtype=torch.long),
            "series_slot_indices": torch.tensor(slots, dtype=torch.long),
            "series_features": torch.stack(metadata),  # [序列数,11]
            "slot_mask": slot_mask,
            "targets": torch.from_numpy(np.nan_to_num(values, nan=0)),
            "label_mask": torch.from_numpy(valid.astype(np.float32)),
            "label_weight": torch.from_numpy(np.where(valid, 2 * np.abs(values - 0.5), 0).astype(np.float32)),
        }


def collate_studies(samples):
    """拼接有效图像/序列，保持每个序列的组边界以及所属 Study、slot。"""
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    batch = {"study_uid": [sample["study_uid"] for sample in samples]}
    for key in ("images", "group_positions", "group_counts", "series_slot_indices", "series_features"):
        batch[key] = torch.cat([sample[key] for sample in samples])
    batch["series_batch_indices"] = torch.cat([
        torch.full((len(sample["group_counts"]),), index, dtype=torch.long)
        for index, sample in enumerate(samples)
    ])
    for key in ("slot_mask", "targets", "label_mask", "label_weight"):
        batch[key] = torch.stack([sample[key] for sample in samples])
    return batch


def build_datasets(root=DEFAULT_ROOT, image_size=DEFAULT_IMAGE_SIZE, crop_mm=DEFAULT_CROP_MM,
                   slot_window_budget=DEFAULT_SLOT_WINDOW_BUDGET):
    metadata = load_metadata(root)
    train_df, train_series_df, test_df, test_series_df, sample_submission = metadata
    train_dataset = KneeDataset(train_df, train_series_df, image_size, crop_mm, slot_window_budget)
    test_dataset = KneeDataset(test_df, test_series_df, image_size, crop_mm, slot_window_budget)
    return train_dataset, test_dataset, sample_submission, metadata


def build_dataloaders(root=DEFAULT_ROOT, batch_size=2, num_workers=4,
                      image_size=DEFAULT_IMAGE_SIZE, crop_mm=DEFAULT_CROP_MM,
                      slot_window_budget=DEFAULT_SLOT_WINDOW_BUDGET):
    train_dataset, test_dataset, sample_submission, metadata = build_datasets(root, image_size, crop_mm, slot_window_budget)
    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True, collate_fn=collate_studies)
    return (DataLoader(train_dataset, shuffle=True, **kwargs),
            DataLoader(test_dataset, shuffle=False, **kwargs), sample_submission, metadata)


if __name__ == "__main__":
    train_dataset, test_dataset, _, _ = build_datasets()
    sample = train_dataset[0]
    print("train studies:", len(train_dataset), "test studies:", len(test_dataset))
    for key, value in sample.items():
        print(key, value.shape if isinstance(value, torch.Tensor) else value)


__all__ = [
    "DEFAULT_ROOT", "DEFAULT_IMAGE_SIZE", "DEFAULT_CROP_MM", "DEFAULT_SLOT_WINDOW_BUDGET", "LABELS", "SLOTS",
    "SERIES_FEATURES", "load_metadata", "get_sorted_dicom_info", "sort_dicom_files",
    "read_dicom", "physical_center_crop", "allocate_slot_windows", "select_unique_anchors", "make_adjacent_windows", "load_series",
    "KneeDataset", "collate_studies", "build_datasets", "build_dataloaders",
]
