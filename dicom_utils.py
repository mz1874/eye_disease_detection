import pydicom
import numpy as np
from PIL import Image
import os


def normalize_pixel(data):
    """将像素归一化到 0~255"""
    data = data.astype(np.float32)
    data -= data.min()
    data /= data.max()
    data *= 255.0
    return data.astype(np.uint8)


def load_dicom_image(dicom_path, output_jpg_path):
    """
    读取 DICOM 文件：
    ✓ 提取 metadata
    ✓ 转换为 JPG（供深度学习模型使用）
    """

    ds = pydicom.dcmread(dicom_path)

    # ==== 图像数据 ====
    pixel_array = ds.pixel_array
    pixel_array = normalize_pixel(pixel_array)

    # 保存为 JPG
    img = Image.fromarray(pixel_array)
    img = img.convert("RGB")
    img.save(output_jpg_path)

    # ==== DICOM Metadata ====
    info = {}

    def get(tag):
        return getattr(ds, tag, None)

    info["patient_id"] = get("PatientID")
    info["patient_name"] = str(get("PatientName"))
    info["patient_age"] = get("PatientAge")
    info["patient_sex"] = get("PatientSex")
    info["visit_id"] = get("VisitID")
    info["laterality"] = get("Laterality")  # 左/右眼 L/R
    info["image_orientation"] = get("ImageOrientationPatient")
    info["device_model"] = get("ManufacturerModelName")
    info["manufacturer"] = get("Manufacturer")
    info["capture_time"] = get("AcquisitionTime")
    info["capture_date"] = get("AcquisitionDate")
    info["field_of_view"] = get("FieldOfView")

    h, w = pixel_array.shape[:2]
    info["resolution"] = {"width": w, "height": h}

    spacing = get("PixelSpacing")
    if spacing:
        info["pixel_spacing"] = {"x_mm": float(spacing[0]), "y_mm": float(spacing[1])}
    else:
        info["pixel_spacing"] = None

    return info
