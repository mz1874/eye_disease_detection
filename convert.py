import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
from datetime import datetime
import numpy as np
import cv2
import os

def convert_image_to_dicom(
    img_path,
    dicom_path,
    patient_id="TEST123",
    study_id="STUDY1",
    eye_side="OD",
    pixel_spacing=(0.014, 0.014)  # ⭐ 默认加入真实 mm/pixel
):
    # 读取图片
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("无法读取图片: " + img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转为 RGB

    # 文件元信息
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    # 创建 DICOM
    ds = FileDataset(dicom_path, {}, file_meta=file_meta, preamble=b"\0" * 128)

    # 基本病人信息
    ds.PatientID = patient_id
    ds.PatientName = "Test^Patient"
    ds.PatientSex = "O"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.Modality = "OT"
    ds.SeriesDescription = f"Fundus {eye_side}"
    ds.ImageComments = "Converted from JPEG"

    # 时间
    dt = datetime.now()
    ds.StudyDate = dt.strftime('%Y%m%d')
    ds.StudyTime = dt.strftime('%H%M%S')
    ds.ContentDate = ds.StudyDate
    ds.ContentTime = ds.StudyTime

    # 图像字段
    ds.SamplesPerPixel = 3
    ds.PhotometricInterpretation = "RGB"
    ds.Rows, ds.Columns, _ = img.shape
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PlanarConfiguration = 0
    ds.PixelData = img.tobytes()

    # ⭐⭐⭐ 关键部分：加入 PixelSpacing（毫米/像素）
    # 让你的 predict_v3 可以得到真实 mm² 面积
    ds.PixelSpacing = [str(pixel_spacing[0]), str(pixel_spacing[1])]

    # 可选：加入 FOV（用于 debug）
    ds.FieldOfViewHorizontal = "45"
    ds.FieldOfViewVertical = "45"

    # 编码方式
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    # 保存
    ds.save_as(dicom_path)
    print("DICOM saved:", dicom_path)
    return dicom_path


# === 使用示例 ===
os.makedirs("dicom_outputs", exist_ok=True)
convert_image_to_dicom(
    "uploads/2.jpg",
    "dicom_outputs/fundus_example.dcm",
    pixel_spacing=(0.014, 0.014)  # ⭐ 这里随便你测试
)
