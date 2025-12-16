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
    patient_name="Test^Patient",
    patient_age="040Y",
    patient_sex="O",
    visit_id="VISIT001",
    eye_side="OD",
    pixel_spacing=(0.014, 0.014),
    device_model="Test Camera",
    manufacturer="Test Manufacturer",
    image_orientation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    field_of_view=(45, 45)  # 水平、垂直
):
    # === 读取图片 ===
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("无法读取图片: " + img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # === DICOM 文件元信息 ===
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    # === 创建 DICOM 数据集 ===
    ds = FileDataset(dicom_path, {}, file_meta=file_meta, preamble=b"\0" * 128)

    # ========= 病人信息 =========
    ds.PatientID = patient_id
    ds.PatientName = patient_name
    ds.PatientAge = patient_age
    ds.PatientSex = patient_sex
    # VisitID 存储在私有标签中，避免标准字典警告
    ds.add_new((0x0011, 0x1010), 'LO', visit_id)  # 私有标签存储 visit_id

    # ========= 图像方向 =========
    # load_dicom_image 解析的是 ImageOrientationPatient (0020,0037)
    ds.ImageOrientationPatient = image_orientation

    # ========= 检查方向 L/R =========
    ds.Laterality = eye_side  # L or R

    # ========= 设备信息 =========
    ds.Manufacturer = manufacturer
    ds.ManufacturerModelName = device_model  # 标准 DICOM 标签 (0008,1090)

    # ========= Study / Series =========
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.Modality = "OT"
    ds.SeriesDescription = f"Fundus {eye_side}"
    ds.ImageComments = "Converted from JPEG"

    # ========= 时间字段 =========
    dt = datetime.now()
    ds.StudyDate = dt.strftime('%Y%m%d')
    ds.StudyTime = dt.strftime('%H%M%S')
    ds.ContentDate = ds.StudyDate
    ds.ContentTime = ds.StudyTime
    ds.AcquisitionDate = ds.StudyDate
    ds.AcquisitionTime = ds.StudyTime

    # ========= 图像像素数据 =========
    ds.SamplesPerPixel = 3
    ds.PhotometricInterpretation = "RGB"
    ds.Rows, ds.Columns, _ = img.shape
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PlanarConfiguration = 0
    ds.PixelData = img.tobytes()

    # ========= Pixel Spacing（mm/pixel）=========
    ds.PixelSpacing = [str(pixel_spacing[0]), str(pixel_spacing[1])]

    # ========= Field Of View（存储为私有标签）=========
    ds.add_new((0x0019, 0x1008), 'DS', str(field_of_view[0]))  # FOV Horizontal
    ds.add_new((0x0019, 0x1009), 'DS', str(field_of_view[1]))  # FOV Vertical

    # ========= DICOM 编码 =========
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    # 保存
    ds.save_as(dicom_path)
    print("DICOM saved:", dicom_path)
    return dicom_path


# === 使用示例 ===
os.makedirs("dicom_outputs", exist_ok=True)
convert_image_to_dicom(
    "uploads/983131418_R_1_.png",
    "dicom_outputs/fundus_example.dcm",
    pixel_spacing=(0.014, 0.014)  # ⭐ 这里随便你测试
)
