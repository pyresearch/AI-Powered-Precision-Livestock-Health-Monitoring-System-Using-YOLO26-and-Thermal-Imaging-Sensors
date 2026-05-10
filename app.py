"""
Livestock Thermography Detection — Flask + YOLO (best.pt)
Returns thermal temperature estimate per detected body part.
"""
import os, time, random
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import cv2

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

ALLOWED = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}

# ── Load YOLO model ────────────────────────────────────────────────────────────
MODEL_PATH = 'best.pt'
try:
    from ultralytics import YOLO
    model = YOLO(MODEL_PATH)
    CLASSES = list(model.names.values())
    MODEL_LOADED = True
    print(f"Model loaded: {MODEL_PATH}  |  Classes: {CLASSES}")
except Exception as e:
    model = None
    CLASSES = ['ear', 'eyes', 'eyess', 'hump', 'neck', 'udder']
    MODEL_LOADED = False
    print(f"Model not loaded ({e}) — simulation mode")

# BGR colours per class
COLORS = [
    (0,212,170),(59,130,246),(249,115,22),
    (168,85,247),(239,68,68),(34,197,94),(236,72,153),(245,158,11),
]

# ── Normal thermal range per body part (°C) ───────────────────────────────────
# Based on published veterinary thermography references for cattle / camels
THERMAL_NORMS = {
    'ear':   {'normal_low': 37.0, 'normal_high': 38.5},
    'eyes':  {'normal_low': 38.0, 'normal_high': 39.5},
    'eyess': {'normal_low': 38.0, 'normal_high': 39.5},
    'hump':  {'normal_low': 34.5, 'normal_high': 36.5},
    'neck':  {'normal_low': 36.0, 'normal_high': 38.0},
    'udder': {'normal_low': 37.5, 'normal_high': 39.5},
}
DEFAULT_NORM = {'normal_low': 36.5, 'normal_high': 39.5}


def allowed(f):
    return '.' in f and f.rsplit('.', 1)[1].lower() in ALLOWED


# ── Thermal temperature estimation ────────────────────────────────────────────
def estimate_thermal_temp(img, bbox, cls_name):
    """
    Estimate temperature for a detected region.

    Strategy (when real thermal camera data is absent):
      1. Extract mean pixel intensity of the bounding-box ROI from the
         image (grayscale proxy for heat emittance).
      2. Map pixel intensity (0-255) → temperature within the anatomical
         normal range ± a physiological spread of ±2 °C.
      3. Add small deterministic jitter based on bbox coordinates so
         each detection has a unique value.

    When a real thermal camera produces 16-bit TIFF / radiometric JPEG,
    replace this function body with direct pixel→temperature calibration
    using the camera's emissivity and offset constants.
    """
    norm = THERMAL_NORMS.get(cls_name.lower(), DEFAULT_NORM)
    t_low  = norm['normal_low']  - 1.5   # allow slightly below normal
    t_high = norm['normal_high'] + 2.0   # allow slightly above normal

    x1, y1, x2, y2 = bbox
    if img is not None:
        roi = img[max(0,y1):max(1,y2), max(0,x1):max(1,x2)]
        if roi.size > 0:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape)==3 else roi
            mean_intensity = float(gray.mean())           # 0–255
            # Map intensity to temperature range
            t = t_low + (mean_intensity / 255.0) * (t_high - t_low)
            # Deterministic jitter (±0.3°C) from bbox position
            jitter = ((x1 * 7 + y1 * 13 + x2 * 3 + y2 * 5) % 60 - 30) / 100.0
            t = round(t + jitter, 1)
            return max(t_low, min(t_high, t))

    # Fallback: simple deterministic value within range
    seed = ((x1 * 7 + y1 * 13 + x2 * 3 + y2 * 5) % 1000) / 1000.0
    t = t_low + seed * (t_high - t_low)
    return round(t, 1)


# ── Draw annotated boxes with temperature ────────────────────────────────────
def draw_boxes(img_path, detections):
    img = cv2.imread(img_path)
    if img is None:
        return img_path
    for d in detections:
        x1, y1, x2, y2 = d['bbox']
        color = COLORS[d['class_id'] % len(COLORS)]
        temp_str = f"{d['thermal_temp']:.1f}°C" if 'thermal_temp' in d else ''
        label = f"{d['class']}  {d['confidence']*100:.1f}%  {temp_str}"
        ov = img.copy()
        cv2.rectangle(ov, (x1,y1), (x2,y2), color, -1)
        cv2.addWeighted(ov, 0.18, img, 0.82, 0, img)
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        ly = max(y1 - 4, th + 6)
        cv2.rectangle(img, (x1, ly-th-6), (x1+tw+8, ly+2), color, -1)
        cv2.putText(img, label, (x1+4, ly-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,0,0), 1, cv2.LINE_AA)
    base, ext = os.path.splitext(img_path)
    out = base + '_det' + (ext or '.jpg')
    cv2.imwrite(out, img)
    return out


# ── Inference ─────────────────────────────────────────────────────────────────
def run_detection(img_path, conf=0.25):
    img = cv2.imread(img_path)

    if not MODEL_LOADED:
        return _simulate(img)

    results = model(img_path, conf=conf, verbose=False)[0]
    dets = []
    for box in results.boxes:
        cid  = int(box.cls[0])
        cn   = model.names.get(cid, f'class_{cid}')
        cv_  = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        temp = estimate_thermal_temp(img, [x1,y1,x2,y2], cn)
        dets.append({
            'class': cn, 'class_id': cid,
            'confidence': round(cv_, 3),
            'bbox': [x1, y1, x2, y2],
            'area_px': (x2-x1)*(y2-y1),
            'thermal_temp': temp,
        })
    return dets


def _simulate(img=None):
    dets = []
    for cls in random.sample(CLASSES, k=random.randint(2, min(4, len(CLASSES)))):
        x1, y1 = random.randint(50,350), random.randint(50,250)
        w, h   = random.randint(80,180), random.randint(60,140)
        bbox   = [x1, y1, x1+w, y1+h]
        temp   = estimate_thermal_temp(img, bbox, cls)
        dets.append({
            'class': cls, 'class_id': CLASSES.index(cls),
            'confidence': round(random.uniform(0.60, 0.97), 3),
            'bbox': bbox,
            'area_px': w * h,
            'thermal_temp': temp,
        })
    return dets


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/static/<path:f>')
def static_f(f):
    return send_from_directory('static', f)

@app.route('/api/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    if not file.filename or not allowed(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    conf  = float(request.form.get('conf', 0.25))
    fname = f"{int(time.time())}_{secure_filename(file.filename)}"
    fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    file.save(fpath)

    t0   = time.time()
    dets = run_detection(fpath, conf=conf)
    ms   = round((time.time()-t0)*1000, 1)

    ann_path = draw_boxes(fpath, dets)
    ann_name = os.path.basename(ann_path)

    total    = len(dets)
    avg_conf = round(sum(d['confidence'] for d in dets)/total, 3) if total else 0
    classes  = {}
    for d in dets:
        classes[d['class']] = classes.get(d['class'], 0) + 1

    # Thermal summary
    temps = [d['thermal_temp'] for d in dets]
    thermal_summary = {
        'max_temp':  round(max(temps), 1) if temps else None,
        'min_temp':  round(min(temps), 1) if temps else None,
        'avg_temp':  round(sum(temps)/len(temps), 1) if temps else None,
    }

    return jsonify({
        'ok': True, 'model_loaded': MODEL_LOADED, 'classes': CLASSES,
        'inference_ms': ms, 'conf_used': conf,
        'original_url':  f'/static/uploads/{fname}',
        'annotated_url': f'/static/uploads/{ann_name}',
        'detections': dets, 'total': total,
        'avg_conf': avg_conf, 'class_dist': classes,
        'thermal_summary': thermal_summary,
        'timestamp': datetime.now().strftime('%H:%M:%S'),
    })


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    print("http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
