from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf
import cv2
import os
import base64
import uuid
import pydicom
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.resnet50 import preprocess_input
from dotenv import load_dotenv
from dicom_utils import load_dicom_image 
import logging

load_dotenv()
model_version = os.environ.get("MODEL_VERSION", "unknown")

# ============ 初始化 Flask ============
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ============ 工具函数：清理非 JSON 序列化对象 ============
def make_json_safe(obj):
    """递归清理对象中的 MultiValue 等非 JSON 序列化类型，转为基础类型"""
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_safe(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        # 将其他类型转为字符串
        return str(obj)

# ============ 加载模型 ============
model = load_model("eye_disease_model.keras")
class_names = ['amd', 'cataract', 'diabetes', 'normal']
base_model = model.layers[0]

amd_stage_model = load_model("best_amd_resnet50_finetune.keras")
amd_stage_classes = ['normal', 'early', 'intermediate_late']

# ============ 工具函数 ============
def encode_image_base64(img_path):
    with open(img_path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()

def preprocess_image(img_path):
    input_shape = model.input_shape[1:3]
    img = load_img(img_path)
    img_array = img_to_array(img) / 255.0
    img_array = tf.image.resize_with_pad(img_array, *input_shape)
    return np.expand_dims(img_array, axis=0)

def compute_iqa_metrics(img_path):
    """
    轻量级图像质量评估（IQA）函数：
    - 依据清晰度（锐度）、曝光（过曝/欠曝）、对比度三个维度，计算一组指标和综合评分。
    - 返回内容包括：
      * iqa_score: 综合评分（0-100），分数越高质量越好。
      * quality_label: 质量等级（good / fair / poor）。
      * metrics: 详细的指标数值（sharpness、brightness、exposure、contrast）。
    注意：该方法为无参考轻量级评估，适合在线推理快速筛查；如需更高精度可引入 BRISQUE/NIQE 或训练专用模型。
    """
    img = cv2.imread(img_path)
    if img is None:
        return {
            "iqa_score": 0,
            "quality_label": "poor",
            "metrics": {
                "sharpness": 0,
                "brightness_mean": None,
                "brightness_std": None,
                "exposure_over_pct": None,
                "exposure_under_pct": None,
                "contrast_rms": None
            }
        }

    # 转灰度以便进行亮度、对比度与锐度计算
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 清晰度（锐度）：使用拉普拉斯算子方差衡量边缘强度（数值越大越清晰）
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(lap.var())

    # 亮度与曝光：统计平均亮度、亮度标准差，以及过曝/欠曝像素比例
    brightness_mean = float(gray.mean())
    brightness_std = float(gray.std())
    over_thresh = 240  # 过曝阈值（灰度≥240视为过曝）
    under_thresh = 15  # 欠曝阈值（灰度≤15视为欠曝）
    total = gray.size
    exposure_over_pct = float((gray >= over_thresh).sum()) / total
    exposure_under_pct = float((gray <= under_thresh).sum()) / total

    # 对比度：RMS 对比度（亮度相对均值的均方根）
    contrast_rms = float(np.sqrt(np.mean((gray - brightness_mean) ** 2)))

    # 综合评分启发式：
    # - 锐度占比 40 分（按 sharpness/300 折算，封顶 1.0）
    # - 对比度占比 30 分（按 contrast_rms/50 折算，封顶 1.0）
    # - 曝光占比 30 分（过曝+欠曝作为惩罚项，越少越好）
    score = 0.0
    score += min(sharpness / 300.0, 1.0) * 40.0
    score += min(contrast_rms / 50.0, 1.0) * 30.0
    exposure_penalty = min(exposure_over_pct + exposure_under_pct, 1.0)
    score += (1.0 - exposure_penalty) * 30.0
    iqa_score = int(round(max(0.0, min(100.0, score))))
    # 质量等级划分：≥70 为 good，≥40 为 fair，其余为 poor
    quality_label = "good" if iqa_score >= 70 else ("fair" if iqa_score >= 40 else "poor")

    return {
        "iqa_score": iqa_score,
        "quality_label": quality_label,
        "metrics": {
            "sharpness": round(sharpness, 4),
            "brightness_mean": round(brightness_mean, 4),
            "brightness_std": round(brightness_std, 4),
            "exposure_over_pct": round(exposure_over_pct, 6),
            "exposure_under_pct": round(exposure_under_pct, 6),
            "contrast_rms": round(contrast_rms, 4)
        }
    }

def make_gradcam_heatmap(img_array, base_model, last_conv_layer_name, pred_index=None):
    """
    img_array: (1,H,W,3) 经过 preprocess_input
    base_model: Functional API 模型
    last_conv_layer_name: 最后一层 Conv2D 名字
    pred_index: 目标类别索引
    """
    logger.info(f"[GradCAM] Using last_conv_layer_name={last_conv_layer_name}")
    last_conv_layer = base_model.get_layer(last_conv_layer_name)
    
    # 获取中间层输出
    grad_model = Model(
        inputs=base_model.inputs,
        outputs=[last_conv_layer.output, base_model.output]
    )
    grad_model.trainable = False

    img_array_tensor = tf.cast(tf.convert_to_tensor(img_array), tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(img_array_tensor)
        conv_outputs, predictions = grad_model(img_array_tensor)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        logger.warning("[GradCAM] grads is None — gradient could not be computed.")
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val == 0:
        logger.warning("[GradCAM] heatmap max is 0 — likely all-zero activation.")
        heatmap = tf.zeros_like(heatmap)
    else:
        heatmap = heatmap / max_val
    logger.info(f"[GradCAM] heatmap stats: max={float(tf.math.reduce_max(heatmap)):.6f}, min={float(tf.math.reduce_min(heatmap)):.6f}, mean={float(tf.math.reduce_mean(heatmap)):.6f}")
    return heatmap.numpy()

def extract_lesion_area_px(heatmap, threshold=0.6):
    logger.info(f"[Lesion] Threshold={threshold}")
    mask = (heatmap >= threshold).astype(np.uint8)
    area = int(np.sum(mask))
    logger.info(f"[Lesion] lesion_area_px={area}")
    return area

def analyze_lesion_regions(heatmap, threshold=0.6, mm2_per_px=None):
    """计算病灶连通域数量与最大单灶面积（像素/可选 mm²）。
    - threshold: 热力图阈值。
    - mm2_per_px: 每像素对应的 mm²（DICOM: spacing_x*spacing_y；非 DICOM: 近似换算）。
    返回: count, max_px, max_mm2
    """
    mask = (heatmap >= threshold).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # stats: [label, left, top, width, height, area]
    # label 0 是背景
    if num_labels <= 1:
        return 0, 0, 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]  # 排除背景
    max_px = int(areas.max())
    count = int(len(areas))
    max_mm2 = float(max_px * mm2_per_px) if mm2_per_px is not None else 0.0
    return count, max_px, max_mm2

def generate_heatmap_base64(img_path, heatmap, alpha=0.6, min_val=None, max_val=None):
    # 使用动态归一化，并在彩色原图上叠加，提升可视化
    img = cv2.imread(img_path)
    if img is None:
        logger.error(f"[Overlay] Failed to read image at {img_path}")
        return None

    h, w = img.shape[:2]
    heatmap = cv2.resize(heatmap, (w, h))

    # 动态确定归一化范围
    if min_val is None or max_val is None or max_val <= min_val:
        min_val = float(np.min(heatmap))
        max_val = float(np.max(heatmap))
    if max_val <= min_val + 1e-8:
        # 退化情况，直接使用 [0,1] 归一化的结果（或全零）
        norm = np.clip(heatmap, 0, 1)
    else:
        norm = np.clip((heatmap - min_val) / (max_val - min_val), 0, 1)

    logger.info(f"[Overlay] img_path={img_path}, alpha={alpha}, min_val={min_val:.6f}, max_val={max_val:.6f}, norm_stats max={float(np.max(norm)):.6f}, mean={float(np.mean(norm)):.6f}")

    heatmap_uint8 = np.uint8(255 * norm)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # 提高热图可见性：适度锐化/对比度可选（此处仅提高 alpha 默认值）
    alpha = max(0.3, min(alpha, 0.9))
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)

    _, buffer = cv2.imencode('.jpg', overlay)
    b64 = base64.b64encode(buffer).decode("utf-8")
    return "data:image/jpeg;base64," + b64

def predict_image_with_heatmap(img_path, min_val=None, max_val=None, fov=45.0, eye_d=12.0, mm2_per_px=None):
    """计算粗分模型的预测、热力图与病灶统计（可选 DICOM mm² 换算）。"""
    logger.info(f"[Predict] img_path={img_path}, fov={fov}, eye_d={eye_d}, min_val={min_val}, max_val={max_val}, mm2_per_px={mm2_per_px}")
    img_array = preprocess_image(img_path)

    preds = model.predict(img_array)
    predicted_idx = np.argmax(preds, axis=1)[0]
    predicted_label = class_names[predicted_idx]
    confidence = float(np.max(preds))
    logger.info(f"[Predict] coarse label={predicted_label}, confidence={confidence:.6f}")

    last_conv_layer_name = None
    for layer in reversed(base_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break
    logger.info(f"[Predict] selected last_conv_layer_name={last_conv_layer_name}")

    heatmap = make_gradcam_heatmap(img_array, base_model, last_conv_layer_name, predicted_idx)
    lesion_px = extract_lesion_area_px(heatmap, threshold=0.6)

    img = cv2.imread(img_path)
    if img is None:
        w = 1
    else:
        h, w = img.shape[:2]

    # 优先使用 DICOM 像素间距换算 mm²，否则退回基于 fov/眼径的近似计算
    if mm2_per_px is None:
        mm2_per_px = (eye_d * (fov / 45.0) / w) ** 2 if w != 0 else 0.0
    lesion_mm2 = lesion_px * mm2_per_px if mm2_per_px else 0.0
    lesion_count, max_lesion_px, max_lesion_mm2 = analyze_lesion_regions(
        heatmap, threshold=0.6, mm2_per_px=mm2_per_px
    )
    logger.info(
        f"[Predict] width={w}, lesion_mm2_estimate={lesion_mm2:.6f}, "
        f"regions={lesion_count}, max_px={max_lesion_px}, max_mm2={max_lesion_mm2:.6f}"
    )

    heatmap_b64 = generate_heatmap_base64(img_path, heatmap, alpha=0.6, min_val=min_val, max_val=max_val)
    original_b64 = encode_image_base64(img_path)

    return (
        predicted_label,
        confidence,
        original_b64,
        heatmap_b64,
        lesion_px,
        lesion_mm2,
        lesion_count,
        max_lesion_px,
        max_lesion_mm2,
    )

# ===========使用 PixelSpacing ==========
@app.route('/predict_v3', methods=['POST'])
def predict_v3():
    """DICOM 支持的病灶面积估计（PixelSpacing 必须存在），返回热力图和面积统计。"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    ext = filename.lower().split('.')[-1]

    if ext == "dcm":
        try:
            ds = pydicom.dcmread(file_path)
            logger.info(f"[v3] Received DICOM file {filename}")

            if "PixelSpacing" not in ds:
                return jsonify({
                    "error": "DICOM does not contain PixelSpacing (cannot compute real lesion area)"
                }), 400

            spacing_x, spacing_y = map(float, ds.PixelSpacing)
            logger.info(f"[v3] PixelSpacing x={spacing_x}, y={spacing_y}")

            jpg_name = str(uuid.uuid4()) + ".jpg"
            jpg_path = os.path.join(UPLOAD_FOLDER, jpg_name)

            dicom_info = load_dicom_image(file_path, jpg_path)
            logger.info(f"[v3] Converted DICOM to JPEG at {jpg_path}")

            (
                label,
                confidence,
                original_b64,
                heatmap_b64,
                area_px,
                lesion_area_mm2,
                lesion_count,
                max_lesion_px,
                max_lesion_mm2,
            ) = predict_image_with_heatmap(
                jpg_path,
                0.0,
                1.0,
                45.0,
                12.0,
                mm2_per_px=spacing_x * spacing_y,
            )
            logger.info(f"[v3] lesion_area_px={area_px}, lesion_area_mm2={lesion_area_mm2:.6f}")

            return jsonify({
                "mode": "dicom",
                "dicom_metadata": dicom_info,
                "prediction": label,
                "confidence": round(confidence, 4),
                "lesion_area_px": area_px,
                "lesion_area_mm2": round(lesion_area_mm2, 6),
                "lesion_region_count": lesion_count,
                "max_lesion_area_px": max_lesion_px,
                "max_lesion_area_mm2": round(max_lesion_mm2, 6),
                "pixel_spacing_x_mm": spacing_x,
                "pixel_spacing_y_mm": spacing_y,
                "original_base64": original_b64,
                "heatmap_base64": heatmap_b64
            })

        except Exception as e:
            logger.exception("[v3] DICOM processing failed")
            return jsonify({"error": f"Failed to process DICOM: {str(e)}"}), 500

    if ext in ["jpg", "jpeg", "png"]:
        try:
            (
                label,
                confidence,
                original_b64,
                heatmap_b64,
                area_px,
                lesion_area_mm2,
                lesion_count,
                max_lesion_px,
                max_lesion_mm2,
            ) = predict_image_with_heatmap(file_path, 0.6, 1.0, 45.0, 12.0)
            logger.info(f"[v3] image mode: lesion_area_px={area_px}, lesion_area_mm2={lesion_area_mm2:.6f}")

            return jsonify({
                "mode": "image",
                "prediction": label,
                "confidence": round(confidence, 4),
                "lesion_area_px": area_px,
                "lesion_area_mm2": round(lesion_area_mm2, 6),
                "lesion_region_count": lesion_count,
                "max_lesion_area_px": max_lesion_px,
                "max_lesion_area_mm2": round(max_lesion_mm2, 6),
                "original_base64": original_b64,
                "heatmap_base64": heatmap_b64
            })

        except Exception as e:
            logger.exception("[v3] Image processing failed")
            return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

    return jsonify({
        "error": "Unsupported file type. Please upload DICOM (.dcm) or image (.jpg/.png)."
    }), 400

@app.route('/predict', methods=['POST'])
def predict_simple():
    """Simple prediction endpoint: returns only label and confidence.
    Accepts image files (jpg/png/jpeg) and DICOM (.dcm). DICOM will be converted to JPEG for inference.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    orig_name = file.filename or ""
    ext = orig_name.rsplit('.', 1)[-1].lower() if "." in orig_name else "jpg"
    filename = f"{uuid.uuid4()}.{ext}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    logger.info(f"[predict] Saved upload to {file_path} (ext={ext})")

    img_path = file_path
    if ext == 'dcm':
        try:
            jpg_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.jpg")
            _ = load_dicom_image(file_path, jpg_path)
            img_path = jpg_path
            logger.info(f"[predict] DICOM converted to JPEG at {img_path}")
        except Exception as e:
            logger.exception("[predict] DICOM conversion error")
            return jsonify({"error": f"DICOM conversion error: {str(e)}"}), 500

    try:
        img_array = preprocess_image(img_path)
        preds = model.predict(img_array)
        predicted_idx = int(np.argmax(preds, axis=1)[0])
        label = class_names[predicted_idx]
        confidence = float(np.max(preds))
        logger.info(f"[predict] label={label}, confidence={confidence:.6f}")
        iqa = compute_iqa_metrics(img_path)
        return jsonify({
            "prediction": label,
            "confidence": round(confidence, 4),
            "iqa_result": iqa
        })
    except Exception as e:
        logger.exception("[predict] Inference error")
        return jsonify({"error": f"Inference error: {str(e)}"}), 500

def predict_single_model(img_path, model, class_names, min_val=None, max_val=None, fov=45.0, eye_d=12.0, mm2_per_px=None):
    img_array = preprocess_image(img_path)

    preds = model.predict(img_array)
    predicted_idx = np.argmax(preds, axis=1)[0]
    predicted_label = class_names[predicted_idx]
    confidence = float(np.max(preds))

    base_model = model.layers[0]
    last_conv_layer_name = None
    for layer in reversed(base_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break

    heatmap = make_gradcam_heatmap(img_array, base_model, last_conv_layer_name, predicted_idx)
    lesion_px = extract_lesion_area_px(heatmap, threshold=0.6)

    img = cv2.imread(img_path)
    if img is None:
        w = 1
    else:
        h, w = img.shape[:2]
    if mm2_per_px is None:
        mm2_per_px = (eye_d * (fov / 45.0) / w) ** 2 if w != 0 else 0.0
    lesion_mm2 = lesion_px * mm2_per_px if mm2_per_px else 0.0
    lesion_count, max_lesion_px, max_lesion_mm2 = analyze_lesion_regions(
        heatmap, threshold=0.6, mm2_per_px=mm2_per_px
    )

    heatmap_b64 = generate_heatmap_base64(img_path, heatmap, alpha=0.6, min_val=min_val, max_val=max_val)
    original_b64 = encode_image_base64(img_path)

    return {
        "label": predicted_label,
        "confidence": confidence,
        "heatmap_base64": heatmap_b64,
        "original_base64": original_b64,
        "lesion_px": lesion_px,
        "lesion_mm2": lesion_mm2,
        "lesion_region_count": lesion_count,
        "max_lesion_area_px": max_lesion_px,
        "max_lesion_area_mm2": max_lesion_mm2,
    }

@app.route('/predict_v4', methods=['POST'])
def predict_v4():
    """多模型推理端点：可调热力图归一化，支持 DICOM/mm² 换算与连通域统计。"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    orig_name = file.filename or ""
    ext = orig_name.rsplit('.', 1)[-1].lower() if "." in orig_name else "jpg"
    filename = f"{uuid.uuid4()}.{ext}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    logger.info(f"[v4] Saved upload to {file_path} (ext={ext})")

    is_dicom = ext == "dcm"

    # 读取可选可视化参数（表单或 JSON），用于控制归一化和叠加
    # 命名规则：coarse_* 作用于一级模型；amd_* 作用于分期模型
    def _get_float(name, default=None):
        val = request.form.get(name)
        if val is None and request.is_json:
            try:
                val = (request.get_json(silent=True) or {}).get(name)
            except Exception:
                val = None
        try:
            # 将值强制转为字符串，处理 MultiValue 等特殊类型
            val_str = str(val).strip() if val is not None else None
            return float(val_str) if val_str and val_str != "" else default
        except (ValueError, TypeError):
            return default

    # 支持通用 min_val/max_val/alpha，如果未提供具体 coarse/amd 参数，则回退使用通用参数
    global_min_val = _get_float("min_val", default=None)
    global_max_val = _get_float("max_val", default=None)
    global_alpha   = _get_float("alpha",   default=None)

    coarse_min_val = _get_float("coarse_min_val", default=(global_min_val if global_min_val is not None else 0.0))
    coarse_max_val = _get_float("coarse_max_val", default=(global_max_val if global_max_val is not None else 1.0))
    coarse_alpha   = _get_float("coarse_alpha",   default=(global_alpha   if global_alpha   is not None else 0.6))

    amd_min_val = _get_float("amd_min_val", default=(global_min_val if global_min_val is not None else 0.0))
    amd_max_val = _get_float("amd_max_val", default=(global_max_val if global_max_val is not None else 1.0))
    amd_alpha   = _get_float("amd_alpha",   default=(global_alpha   if global_alpha   is not None else 0.6))

    # 仅图片类型支持 fov（视场角）和眼球直径输入
    input_fov = _get_float("fov", default=45.0)
    eye_diameter_mm = _get_float("eye_diameter_mm", default=12.0)
    logger.info(f"[v4] Params coarse(min={coarse_min_val}, max={coarse_max_val}, alpha={coarse_alpha}), amd(min={amd_min_val}, max={amd_max_val}, alpha={amd_alpha}), fov={input_fov}, eye_diameter_mm={eye_diameter_mm}")

    dicom_meta = None
    pixel_spacing_x = None
    pixel_spacing_y = None
    img_path = file_path

    if is_dicom:
        try:
            ds = pydicom.dcmread(file_path)
            if "PixelSpacing" not in ds:
                return jsonify({"error": "DICOM missing PixelSpacing"}), 400
            pixel_spacing_x, pixel_spacing_y = map(float, ds.PixelSpacing)
            logger.info(f"[v4] DICOM PixelSpacing x={pixel_spacing_x}, y={pixel_spacing_y}")

            jpg_name = str(uuid.uuid4()) + ".jpg"
            jpg_path = os.path.join(UPLOAD_FOLDER, jpg_name)

            dicom_meta = load_dicom_image(file_path, jpg_path)
            img_path = jpg_path
            logger.info(f"[v4] DICOM converted to JPEG at {img_path}")

        except Exception as e:
            logger.exception("[v4] DICOM conversion error")
            return jsonify({"error": f"DICOM error: {str(e)}"}), 500

    # 一级大模型预测
    logger.info("[v4] Running coarse model prediction")
    coarse_result = predict_single_model(
        img_path,
        model,
        class_names,
        min_val=coarse_min_val,
        max_val=coarse_max_val,
        fov=input_fov,
        eye_d=eye_diameter_mm,
        mm2_per_px=(pixel_spacing_x * pixel_spacing_y) if is_dicom else None,
    )
    logger.info(f"[v4] Coarse model: label={coarse_result['label']}, conf={coarse_result['confidence']:.6f}, lesion_px={coarse_result['lesion_px']}, lesion_mm2={coarse_result['lesion_mm2']:.6f}")

    # AMD 分期模型预测
    img = load_img(img_path, target_size=(224,224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    logger.info("[v4] Running AMD stage model prediction")
    preds = amd_stage_model.predict(img_array)
    predicted_idx = np.argmax(preds, axis=1)[0]
    predicted_label = amd_stage_classes[predicted_idx]
    confidence = float(np.max(preds))
    logger.info(f"[v4] AMD stage: label={predicted_label}, conf={confidence:.6f}")

    base_model_layer = amd_stage_model.layers[0]
    last_conv_layer_name = None
    for layer in reversed(base_model_layer.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break
    logger.info(f"[v4] AMD stage last_conv_layer_name={last_conv_layer_name}")

    heatmap = make_gradcam_heatmap(img_array, base_model_layer, last_conv_layer_name, predicted_idx)
    max_val_local = np.max(heatmap)
    heatmap = np.zeros_like(heatmap) if max_val_local == 0 else heatmap / max_val_local
    logger.info(f"[v4] AMD heatmap stats: max={float(np.max(heatmap)):.6f}, min={float(np.min(heatmap)):.6f}, mean={float(np.mean(heatmap)):.6f}")
    lesion_px = extract_lesion_area_px(heatmap, threshold=0.6)
    heatmap_b64 = generate_heatmap_base64(img_path, heatmap, alpha=amd_alpha, min_val=amd_min_val, max_val=amd_max_val)

    # AMD 分期病灶面积与连通域统计
    if is_dicom:
        mm2_per_px_amd = pixel_spacing_x * pixel_spacing_y
        logger.info(f"[v4] AMD using DICOM mm2_per_px={mm2_per_px_amd:.6f}")
    else:
        img_cv = cv2.imread(img_path)
        w = 1 if img_cv is None else img_cv.shape[1]
        mm2_per_px_amd = (eye_diameter_mm * (input_fov / 45.0) / w) ** 2 if w != 0 else 0.0
        logger.info(f"[v4] AMD using image mm2_per_px={mm2_per_px_amd:.6f}")

    lesion_mm2 = lesion_px * mm2_per_px_amd if mm2_per_px_amd else 0.0
    lesion_count, max_lesion_px, max_lesion_mm2 = analyze_lesion_regions(
        heatmap, threshold=0.6, mm2_per_px=mm2_per_px_amd
    )
    logger.info(
        f"[v4] AMD lesions: px={lesion_px}, mm2={lesion_mm2:.6f}, "
        f"regions={lesion_count}, max_px={max_lesion_px}, max_mm2={max_lesion_mm2:.6f}"
    )

    amd_stage_result = {
        "label": predicted_label,
        "confidence": confidence,
        "lesion_px": lesion_px,
        "lesion_mm2": lesion_mm2,
        "lesion_region_count": lesion_count,
        "max_lesion_area_px": max_lesion_px,
        "max_lesion_area_mm2": max_lesion_mm2,
        "heatmap_base64": heatmap_b64
    }

    if is_dicom:
        coarse_result["lesion_mm2"] = coarse_result["lesion_px"] * pixel_spacing_x * pixel_spacing_y

    # IQA：对输入图像（DICOM 会使用转换后的 JPEG）进行质量评估
    iqa = compute_iqa_metrics(img_path)

    response_data = {
        "mode": "dicom" if is_dicom else "image",
        "dicom_metadata": dicom_meta,
        "coarse_model": coarse_result,
        "amd_stage_model": amd_stage_result,
        "pixel_spacing_x_mm": pixel_spacing_x,
        "pixel_spacing_y_mm": pixel_spacing_y,
        "model_version": model_version,
        "iqa_result": iqa
    }
    
    # 清理非 JSON 序列化对象
    response_data = make_json_safe(response_data)
    return jsonify(response_data)

# ============ 启动服务 ============
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
