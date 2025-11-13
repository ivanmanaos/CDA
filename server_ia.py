import os
import json
import tempfile
import subprocess
from flask import Flask, request, jsonify, make_response
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Simple CORS (no flask_cors dependency required)
ALLOWED_ORIGIN = os.environ.get("CORS_ORIGIN", "*")

DEVICE = os.environ.get("DEVICE", "cpu")
WEIGHTS_PATH = os.environ.get("WEIGHTS", "./models/cda_b7_best_mae.pth")
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", 384))

@app.after_request
def add_cors_headers(resp):
    resp.headers['Access-Control-Allow-Origin'] = ALLOWED_ORIGIN
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    resp.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return resp

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"ok": True})

@app.route('/infer', methods=['POST'])
def infer():
    if 'image' not in request.files:
        return jsonify({"error": "missing form file 'image'"}), 400
    img_file = request.files['image']
    if img_file.filename == '':
        return jsonify({"error": "empty filename"}), 400
    filename = secure_filename(img_file.filename)

    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, filename)
        out_json_path = os.path.join(tmpdir, 'out.json')
        img_file.save(img_path)

        cmd = [
            'python', 'scripts/infer_single.py',
            '--weights', WEIGHTS_PATH,
            '--image', img_path,
            '--image_size', str(IMAGE_SIZE),
            '--device', DEVICE,
            '--out_json', out_json_path,
            '--print_json'
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.abspath('.'), timeout=300)
        except subprocess.TimeoutExpired:
            return jsonify({"error": "inference timeout"}), 500

        if proc.returncode != 0:
            return jsonify({"error": "inference failed", "stderr": proc.stderr[-500:]}), 500

        # The script prints JSON; but we trust the file out_json_path for structured data.
        if not os.path.exists(out_json_path):
            return jsonify({"error": "output json not found"}), 500
        try:
            with open(out_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            return jsonify({"error": "failed to parse output json", "detail": str(e)}), 500

        return jsonify(data)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    host = '0.0.0.0'
    # Use threaded=True to allow concurrent requests.
    app.run(host=host, port=port, threaded=True)
