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

load_dotenv()
model_version = os.environ.get("MODEL_VERSION", "unknown")

# ============ 初始化 Flask ============
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

def make_gradcam_heatmap(img_array, base_model, last_conv_layer_name, pred_index=None):
    """
    img_array: (1,H,W,3) 经过 preprocess_input
    base_model: Functional API 模型
    last_conv_layer_name: 最后一层 Conv2D 名字
    pred_index: 目标类别索引
    """
    last_conv_layer = base_model.get_layer(last_conv_layer_name)
    
    # 获取中间层输出
    grad_model = Model(
        inputs=base_model.inputs,
        outputs=[last_conv_layer.output, base_model.output]
    )
    grad_model.trainable = False

    img_array_tensor = tf.convert_to_tensor(img_array)
    with tf.GradientTape() as tape:
        tape.watch(img_array_tensor)
        conv_outputs, predictions = grad_model(img_array_tensor)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val == 0:
        heatmap = tf.zeros_like(heatmap)
    else:
        heatmap = heatmap / max_val

    return heatmap.numpy()

def extract_lesion_area_px(heatmap, threshold=0.6):
    mask = (heatmap >= threshold).astype(np.uint8)
    return int(np.sum(mask))

def generate_heatmap_base64(img_path, heatmap, alpha=0.4, min_val=0.0, max_val=1.0):
    img = cv2.imread(img_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    h, w = img_gray.shape[:2]
    heatmap = cv2.resize(heatmap, (w, h))
    heatmap = np.clip((heatmap - min_val) / max(max_val - min_val, 1e-8), 0, 1)
    heatmap = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_gray, 1 - alpha, heatmap_color, alpha, 0)

    _, buffer = cv2.imencode('.jpg', overlay)
    b64 = base64.b64encode(buffer).decode("utf-8")

    return "data:image/jpeg;base64," + b64

def predict_image_with_heatmap(img_path, min_val, max_val, fov, eye_d):
    img_array = preprocess_image(img_path)

    preds = model.predict(img_array)
    predicted_idx = np.argmax(preds, axis=1)[0]
    predicted_label = class_names[predicted_idx]
    confidence = float(np.max(preds))

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

    lesion_mm2 = 0 if w == 0 else (lesion_px * (eye_d * (fov / 45.0) / w) ** 2)

    heatmap_b64 = generate_heatmap_base64(img_path, heatmap, min_val, max_val)
    original_b64 = encode_image_base64(img_path)

    return predicted_label, confidence, original_b64, heatmap_b64, lesion_px, lesion_mm2

# ===========使用 PixelSpacing ==========
@app.route('/predict_v3', methods=['POST'])
def predict_v3():
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

            if "PixelSpacing" not in ds:
                return jsonify({
                    "error": "DICOM does not contain PixelSpacing (cannot compute real lesion area)"
                }), 400

            spacing_x, spacing_y = map(float, ds.PixelSpacing)

            jpg_name = str(uuid.uuid4()) + ".jpg"
            jpg_path = os.path.join(UPLOAD_FOLDER, jpg_name)

            dicom_info = load_dicom_image(file_path, jpg_path)

            label, confidence, original_b64, heatmap_b64, area_px, _ = \
                predict_image_with_heatmap(jpg_path, 0.0, 1.0, 45.0, 12.0)

            lesion_area_mm2 = area_px * spacing_x * spacing_y

            return jsonify({
                "mode": "dicom",
                "dicom_metadata": dicom_info,
                "prediction": label,
                "confidence": round(confidence, 4),
                "lesion_area_px": area_px,
                "lesion_area_mm2": round(lesion_area_mm2, 6),
                "pixel_spacing_x_mm": spacing_x,
                "pixel_spacing_y_mm": spacing_y,
                "original_base64": original_b64,
                "heatmap_base64": heatmap_b64
            })

        except Exception as e:
            return jsonify({"error": f"Failed to process DICOM: {str(e)}"}), 500

    if ext in ["jpg", "jpeg", "png"]:
        try:
            label, confidence, original_b64, heatmap_b64, area_px, lesion_area_mm2 = \
                predict_image_with_heatmap(file_path, 0.6, 1.0, 45.0, 12.0)

            return jsonify({
                "mode": "image",
                "prediction": label,
                "confidence": round(confidence, 4),
                "lesion_area_px": area_px,
                "lesion_area_mm2": round(lesion_area_mm2, 6),
                "original_base64": original_b64,
                "heatmap_base64": heatmap_b64
            })

        except Exception as e:
            return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

    return jsonify({
        "error": "Unsupported file type. Please upload DICOM (.dcm) or image (.jpg/.png)."
    }), 400

def predict_single_model(img_path, model, class_names, min_val=0.0, max_val=1.0, fov=45.0, eye_d=12.0):
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
    lesion_mm2 = 0 if w == 0 else (lesion_px * (eye_d * (fov / 45.0) / w) ** 2)

    heatmap_b64 = generate_heatmap_base64(img_path, heatmap, min_val, max_val)
    original_b64 = encode_image_base64(img_path)

    return {
        "label": predicted_label,
        "confidence": confidence,
        "heatmap_base64": heatmap_b64,
        "original_base64": original_b64,
        "lesion_px": lesion_px,
        "lesion_mm2": lesion_mm2
    }

@app.route('/predict_v4', methods=['POST'])
def predict_v4():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    orig_name = file.filename or ""
    ext = orig_name.rsplit('.', 1)[-1].lower() if "." in orig_name else "jpg"
    filename = f"{uuid.uuid4()}.{ext}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    is_dicom = ext == "dcm"

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

            jpg_name = str(uuid.uuid4()) + ".jpg"
            jpg_path = os.path.join(UPLOAD_FOLDER, jpg_name)

            dicom_meta = load_dicom_image(file_path, jpg_path)
            img_path = jpg_path

        except Exception as e:
            return jsonify({"error": f"DICOM error: {str(e)}"}), 500

    # 一级大模型预测
    coarse_result = predict_single_model(
        img_path,
        model,
        class_names,
        min_val=0.0,
        max_val=1.0
    )

    # AMD 分期模型预测
    img = load_img(img_path, target_size=(224,224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = amd_stage_model.predict(img_array)
    predicted_idx = np.argmax(preds, axis=1)[0]
    predicted_label = amd_stage_classes[predicted_idx]
    confidence = float(np.max(preds))

    base_model_layer = amd_stage_model.layers[0]
    last_conv_layer_name = None
    for layer in reversed(base_model_layer.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break

    heatmap = make_gradcam_heatmap(img_array, base_model_layer, last_conv_layer_name, predicted_idx)
    max_val = np.max(heatmap)
    heatmap = np.zeros_like(heatmap) if max_val == 0 else heatmap / max_val
    lesion_px = extract_lesion_area_px(heatmap, threshold=0.6)
    heatmap_b64 = generate_heatmap_base64(img_path, heatmap, min_val=0.0, max_val=1.0)

    lesion_mm2 = lesion_px
    if is_dicom:
        lesion_mm2 *= pixel_spacing_x * pixel_spacing_y
    else:
        img_cv = cv2.imread(img_path)
        w = 1 if img_cv is None else img_cv.shape[1]
        lesion_mm2 = 0 if w == 0 else lesion_px * (12.0 * (45.0 / 45.0) / w) ** 2

    amd_stage_result = {
        "label": predicted_label,
        "confidence": confidence,
        "lesion_px": lesion_px,
        "lesion_mm2": lesion_mm2,
        "heatmap_base64": heatmap_b64
    }

    if is_dicom:
        coarse_result["lesion_mm2"] = coarse_result["lesion_px"] * pixel_spacing_x * pixel_spacing_y

    return jsonify({
        "mode": "dicom" if is_dicom else "image",
        "dicom_metadata": dicom_meta,
        "coarse_model": coarse_result,
        "amd_stage_model": amd_stage_result,
        "pixel_spacing_x_mm": pixel_spacing_x,
        "pixel_spacing_y_mm": pixel_spacing_y,
        "model_version": model_version
    })

# ============ 启动服务 ============
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
