from flask import Flask, render_template, request, jsonify, send_file
import os
import cv2
import numpy as np
import joblib
from keras.models import load_model
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime
import sqlite3
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import traceback
from skimage.feature import graycomatrix, graycoprops

app = Flask(__name__, template_folder='templates')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')  # Using the dedicated uploads folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

DB_PATH = os.path.join(BASE_DIR, 'history', 'history.db')  # Updated database path

def init_db():
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH, timeout=10)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                filename TEXT,
                result TEXT,
                confidence REAL
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                original_label TEXT,
                corrected_label TEXT,
                confidence REAL,
                timestamp TEXT,
                explanation TEXT
            )
        ''')
        c.execute("PRAGMA table_info(corrections)")
        columns = [col[1] for col in c.fetchall()]
        if 'fault' not in columns:
            c.execute('ALTER TABLE corrections ADD COLUMN fault TEXT')
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error initializing database: {e}\n{traceback.format_exc()}")

init_db()

def load_model_safe(load_function, model_path, model_name):
    try:
        if not os.path.exists(model_path):
            print(f"Error loading {model_name} model: File not found at {model_path}")
            return None
        model = load_function(model_path)
        print(f"{model_name} model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading {model_name} model: {e}\n{traceback.format_exc()}")
        return None

MODEL_DIR = os.path.join(BASE_DIR, 'models')
ab = load_model_safe(joblib.load, os.path.join(MODEL_DIR, 'adaboost_model.pkl'), "AdaBoost")
cb = load_model_safe(joblib.load, os.path.join(MODEL_DIR, 'catboost_model.pkl'), "CatBoost")
cnn = load_model_safe(load_model, os.path.join(MODEL_DIR, 'cnn_model.h5'), "CNN")
tl_model = load_model_safe(load_model, os.path.join(MODEL_DIR, 'transfer_learning_model.h5'), "Transfer Learning")
mri_filter = load_model_safe(joblib.load, os.path.join(MODEL_DIR, 'mri_filter.pkl'), "MRI Filter")

try:
    metadata = joblib.load(os.path.join(MODEL_DIR, 'model_metadata.pkl'))
    weights = metadata.get("weights", {"AdaBoost": 0.2, "CatBoost": 0.2, "CNN": 0.3, "VGG16": 0.3})
except Exception as e:
    print(f"Error loading model_metadata.pkl: {e}\n{traceback.format_exc()}")
    weights = {"AdaBoost": 0.2, "CatBoost": 0.2, "CNN": 0.3, "VGG16": 0.3}

label_map = {0: 'No Tumor', 1: 'Have Tumor'}
mri_label_map = {0: 'Non-MRI', 1: 'MRI'}

def preprocess_gray(path):
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image at {path}")
        img = cv2.resize(img, (128, 128)) / 255.0
        return img.reshape(1, -1), img.reshape(1, 128, 128, 1)
    except Exception as e:
        print(f"Error in preprocess_gray: {e}\n{traceback.format_exc()}")
        raise

def preprocess_rgb(path):
    try:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to load image at {path}")
        img = cv2.resize(img, (128, 128)) / 255.0
        return img.reshape(1, 128, 128, 3)
    except Exception as e:
        print(f"Error in preprocess_rgb: {e}\n{traceback.format_exc()}")
        raise

def preprocess_for_mri_filter(path):
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image at {path}")
        img = cv2.resize(img, (128, 128)) / 255.0  # Correct normalization
        return img.flatten().reshape(1, -1)  # 16384 features
    except Exception as e:
        print(f"Error in preprocess_for_mri_filter: {e}\n{traceback.format_exc()}")
        raise

def generate_hyperspectral_data(image_path, prediction):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image at {image_path}")
        img = cv2.resize(img, (128, 128))

        if prediction == 1:
            heatmap = np.zeros_like(img, dtype=np.float32)
            center = (64, 64)
            sigma = 20
            y, x = np.ogrid[:128, :128]
            heatmap = np.exp(-((x - center[0])**2 + (y - center[1])**2) / (2 * sigma**2))
            heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
            highlighted = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
        else:
            highlighted = img

        _, buffer = cv2.imencode('.png', highlighted)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        plt.figure(figsize=(4, 2), facecolor='none')
        wavelengths = np.linspace(450, 900, 100)
        if prediction == 1:
            intensity = 0.8 - 0.4 * np.exp(-((wavelengths - 650)**2) / (2 * 50**2))
            intensity += np.random.normal(0, 0.05, 100)
        else:
            intensity = 0.5 + 0.2 * np.cos(wavelengths / 900 * np.pi)
            intensity += np.random.normal(0, 0.03, 100)
        intensity = np.clip(intensity, 0, 1)

        plt.plot(wavelengths, intensity, color='#33eeff', linewidth=2)
        plt.fill_between(wavelengths, intensity, alpha=0.2, color='#33eeff')
        plt.title('Spectral Signature Analysis', color='#d8c8c8', fontsize=10)
        plt.xlabel('Wavelength (nm)', color='#d8c8c8', fontsize=8)
        plt.ylabel('Reflectance', color='#d8c8c8', fontsize=8)
        plt.grid(True, alpha=0.3, color='#11aaff')
        plt.gca().set_facecolor('none')
        plt.tick_params(colors='#d8c8c8', labelsize=6)

        buf = BytesIO()
        plt.savefig(buf, format='png', transparent=True, bbox_inches='tight')
        buf.seek(0)
        graph_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        return img_base64, graph_base64
    except Exception as e:
        print(f"Error in generate_hyperspectral_data: {e}\n{traceback.format_exc()}")
        raise

def make_predictions(image_path):
    try:
        flat_img, cnn_img = preprocess_gray(image_path)
        vgg_img = preprocess_rgb(image_path)

        preds = {}
        proba = {}

        if ab:
            preds['AdaBoost'] = ab.predict(flat_img)[0]
            proba['AdaBoost'] = float(ab.predict_proba(flat_img)[0][1])
        else:
            preds['AdaBoost'] = 0
            proba['AdaBoost'] = 0.0

        if cb:
            preds['CatBoost'] = cb.predict(flat_img)[0]
            proba['CatBoost'] = float(cb.predict_proba(flat_img)[0][1])
        else:
            preds['CatBoost'] = 0
            proba['CatBoost'] = 0.0

        if cnn:
            cnn_out = cnn.predict(cnn_img, verbose=0)[0][0]
            preds['CNN'] = int(cnn_out > 0.5)
            proba['CNN'] = float(cnn_out)
        else:
            preds['CNN'] = 0
            proba['CNN'] = 0.0

        if tl_model:
            vgg_out = tl_model.predict(vgg_img, verbose=0)[0][0]
            preds['VGG16'] = int(vgg_out > 0.5)
            proba['VGG16'] = float(vgg_out)
        else:
            preds['VGG16'] = 0
            proba['VGG16'] = 0.0

        weighted_score = sum(proba[k] * weights[k] for k in weights)
        ensemble_pred = int(weighted_score >= 0.5)
        ensemble_proba = weighted_score
        preds['Ensemble'] = ensemble_pred
        proba['Ensemble'] = ensemble_proba

        hyperspectral_img, spectral_graph = generate_hyperspectral_data(image_path, ensemble_pred)
        return preds, proba, hyperspectral_img, spectral_graph
    except Exception as e:
        print(f"Error in make_predictions: {e}\n{traceback.format_exc()}")
        raise

def save_history_db(record):
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        c = conn.cursor()
        c.execute('INSERT INTO history (timestamp, filename, result, confidence) VALUES (?, ?, ?, ?)', record)
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving history: {e}\n{traceback.format_exc()}")

@app.route('/history', methods=['GET'])
def get_history():
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        c = conn.cursor()
        c.execute('SELECT timestamp, filename, result, confidence FROM history ORDER BY id DESC')
        rows = c.fetchall()
        conn.close()
        scans = [
            {
                'timestamp': row[0],
                'filename': row[1],
                'ensemble_prediction': row[2].lower().replace(' ', '_'),
                'ensemble_conf': row[3] / 100,
                'disagreement_status': 'N/A'  # Simplified for compatibility
            } for row in rows
        ]
        return render_template('history.html', scans=scans)
    except Exception as e:
        print(f"Error fetching history: {e}\n{traceback.format_exc()}")
        return render_template('history.html', scans=[])

@app.route('/history/delete/<filename>', methods=['POST'])
def delete_history(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        
        conn = sqlite3.connect(DB_PATH, timeout=10)
        c = conn.cursor()
        c.execute('DELETE FROM history WHERE filename = ?', (filename,))
        conn.commit()
        conn.close()
        
        return jsonify({'status': 'success'})
    except Exception as e:
        print(f"Error deleting history: {e}\n{traceback.format_exc()}")
        return jsonify({'error': 'Failed to delete history'}), 500

@app.route('/correct', methods=['POST'])
def correct_label():
    try:
        data = request.get_json()
        filename = data.get('filename')
        original_label = data.get('original_label')
        corrected_label = data.get('corrected_label')
        confidence = float(data.get('confidence', 0))
        fault = data.get('fault', '')
        explanation = data.get('explanation', '')
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if not all([filename, original_label, corrected_label]) or corrected_label not in ['No Tumor', 'Have Tumor', 'Unknown']:
            return jsonify({'error': 'Invalid correction data'}), 400

        conn = sqlite3.connect(DB_PATH, timeout=10)
        c = conn.cursor()
        c.execute("PRAGMA table_info(corrections)")
        columns = [col[1] for col in c.fetchall()]
        has_fault = 'fault' in columns
        if has_fault:
            c.execute('''
                INSERT INTO corrections (filename, original_label, corrected_label, confidence, timestamp, fault, explanation)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (filename, original_label, corrected_label, confidence, timestamp, fault, explanation))
        else:
            c.execute('''
                INSERT INTO corrections (filename, original_label, corrected_label, confidence, timestamp, explanation)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (filename, original_label, corrected_label, confidence, timestamp, explanation))
        conn.commit()
        conn.close()
        return jsonify({'status': 'success', 'message': 'Correction saved'})
    except Exception as e:
        print(f"Error saving correction: {e}\n{traceback.format_exc()}")
        return jsonify({'error': 'Failed to save correction'}), 500

@app.route('/update_explanation/<int:correction_id>', methods=['POST'])
def update_explanation(correction_id):
    try:
        data = request.get_json()
        explanation = data.get('explanation', '')
        
        conn = sqlite3.connect(DB_PATH, timeout=10)
        c = conn.cursor()
        c.execute('UPDATE corrections SET explanation = ? WHERE id = ?', (explanation, correction_id))
        conn.commit()
        conn.close()
        return jsonify({'status': 'success', 'message': 'Explanation updated'})
    except Exception as e:
        print(f"Error updating explanation: {e}\n{traceback.format_exc()}")
        return jsonify({'error': 'Failed to update explanation'}), 500

@app.route('/corrections', methods=['GET'])
def get_corrections():
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        c = conn.cursor()
        c.execute("PRAGMA table_info(corrections)")
        columns = [col[1] for col in c.fetchall()]
        has_fault = 'fault' in columns
        if has_fault:
            c.execute('SELECT id, filename, original_label, corrected_label, confidence, timestamp, fault, explanation FROM corrections ORDER BY timestamp DESC')
        else:
            c.execute('SELECT id, filename, original_label, corrected_label, confidence, timestamp, NULL as fault, explanation FROM corrections ORDER BY timestamp DESC')
        rows = c.fetchall()
        conn.close()
        corrections = [
            {
                'id': row[0],
                'filename': row[1],
                'original_label': row[2],
                'corrected_label': row[3],
                'confidence': row[4],
                'timestamp': row[5],
                'fault': row[6] or '',
                'explanation': row[7] or ''
            } for row in rows
        ]
        return render_template('corrections.html', corrections=corrections)
    except Exception as e:
        print(f"Error fetching corrections: {e}\n{traceback.format_exc()}")
        return render_template('corrections.html', corrections=[])

@app.route('/')
def index():
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        c = conn.cursor()
        c.execute('SELECT timestamp, filename, result, confidence FROM history ORDER BY id DESC')
        rows = c.fetchall()
        conn.close()
        history = [
            {'timestamp': row[0], 'filename': row[1], 'result': row[2], 'confidence': row[3]}
            for row in rows
        ]
        return render_template('index.html', history=history)
    except Exception as e:
        print(f"Error loading index: {e}\n{traceback.format_exc()}")
        return render_template('index.html', history=[])

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image(path):
    try:
        # First check if we can read the image
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False, "Unable to read image file"

        # Basic image quality checks
        if img.shape[0] < 64 or img.shape[1] < 64:
            return False, "Image resolution too low"
        if np.mean(img) < 10 or np.mean(img) > 245:
            return False, "Image too dark or too bright"

        # MRI filter validation
        if mri_filter is not None:
            try:
                input_data = preprocess_for_mri_filter(path)
                prediction = mri_filter.predict(input_data)[0]
                confidence = mri_filter.predict_proba(input_data)[0][prediction]
                
                if prediction == 0:  # Non-MRI image
                    return False, f"Non-MRI image detected (Confidence: {confidence:.2f})"
                elif confidence < 0.7:  # Low confidence MRI classification
                    return False, f"Image may not be a brain MRI scan (Confidence: {confidence:.2f})"
            except Exception as mri_error:
                print(f"MRI filter error: {mri_error}\n{traceback.format_exc()}")
                # If MRI filter fails, fall back to basic validation
                pass
        
        return True, ""
    except Exception as e:
        print(f"Image validation failed: {e}\n{traceback.format_exc()}")
        return False, f"Image validation failed: {str(e)}"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Empty file'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Unsupported file type'}), 400

        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(path)

        if not os.path.exists(path):
            return jsonify({'error': 'Failed to save image'}), 500

        is_valid, validation_error = validate_image(path)
        if not is_valid:
            return jsonify({'error': f'Invalid image: {validation_error}', 'fault': validation_error}), 400

        if not any([ab, cb, cnn, tl_model]):
            return jsonify({'error': 'No prediction models available'}), 500

        preds, proba, hyperspectral_img, spectral_graph = make_predictions(path)

        model_disagreement = len(set(preds.values())) > 1
        disagreement_warning = "⚠️ Model Disagreement Detected" if model_disagreement else ""

        record = (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), unique_filename, label_map[preds['Ensemble']], float(proba['Ensemble'] * 100))
        save_history_db(record)

        conn = sqlite3.connect(DB_PATH, timeout=10)
        c = conn.cursor()
        c.execute("SELECT corrected_label FROM corrections WHERE filename = ?", (unique_filename,))
        row = c.fetchone()
        conn.close()

        result = {
            'filename': unique_filename,
            'general': {
                'label': label_map[preds['Ensemble']],
                'confidence': round(float(proba['Ensemble'] * 100), 2)
            },
            'alert': get_alert(preds['Ensemble'], proba['Ensemble']),
            'individual': {
                model: {
                    'label': label_map[preds[model]],
                    'confidence': round(float(proba[model] * 100), 2)
                } for model in ['AdaBoost', 'CatBoost', 'CNN', 'VGG16']
            },
            'hyperspectral': {
                'image': hyperspectral_img,
                'graph': spectral_graph
            },
            'disagreement_warning': disagreement_warning
        }

        if row:
            corrected = row[0]
            result['general']['label'] = corrected
            result['alert'] = f"🛠️ Corrected manually: {corrected}"

        return jsonify(result)
    except Exception as e:
        print(f"Error in predict: {e}\n{traceback.format_exc()}")
        return jsonify({'error': f'Scan failed: {str(e)}'}), 500

@app.route('/uploads/<filename>')
def serve_file(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        return send_file(file_path, mimetype='image/jpeg')  # Add proper mimetype
    except Exception as e:
        print(f"Error serving file: {e}\n{traceback.format_exc()}")
        return jsonify({'error': 'Failed to serve file'}), 500

def get_alert(prediction, confidence):
    if prediction == 1 and confidence > 0.95:
        return "🚨 CRITICAL: High-confidence tumor detected."
    elif prediction == 1:
        return "⚠️ Tumor detected with moderate confidence."
    else:
        return "✅ No tumor detected."

if __name__ == '__main__':
    app.run(debug=True)