import os
import sys
import io
import json
import redis
from datetime import datetime, timezone
from dotenv import load_dotenv
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from load_champion import predict_image

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev_secret_key')
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_CONTENT_LENGTH', 5 * 1024 * 1024)) # 5MB limit

redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
r = redis.Redis.from_url(redis_url, decode_responses=True)

@app.route('/health')
def health_check():
    try:
        r.ping() # Verify Redis is connected
        return jsonify({"status": "healthy", "model_loaded": True, "db_connected": True})
    except redis.ConnectionError:
        return jsonify({"status": "unhealthy", "db_connected": False}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    flash('File is too large. Maximum allowed size is 5MB.')
    return redirect(url_for('upload'))

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files.get('file')
        print(f"File: {file}")
        
        if not file or file.filename == '':
            flash('No file selected. Please choose an image.')
            return redirect(request.url)
        
        allowed_extensions = {'png', 'jpg', 'jpeg'}
        if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions: # type: ignore
            flash('Invalid file type. Only PNG, JPG, or JPEG images are allowed.')
            return redirect(request.url)
        
        filename = secure_filename(file.filename) # type: ignore
        
        file_bytes = file.read()
        
        try:
            img = Image.open(io.BytesIO(file_bytes))
            img.verify() # Checks if it's a valid image structure
        except Exception:
            flash('The file is corrupted or not a valid decodable image.')
            return redirect(request.url)
        
        try:
            label, confidence = predict_image(file_bytes)
            
            record = {
                "id": r.llen("prediction_history") + 1,
                "filename": filename,
                "verdict": label,
                "confidence": confidence,
                "model_version": "champion_v1",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            r.lpush("prediction_history", json.dumps(record))
            
            return render_template('result.html', filename=filename, verdict=label, confidence=confidence)
            
        except Exception as e:
            flash(f'An error occurred during prediction: {str(e)}')
            return redirect(request.url)
            
    return render_template('index.html')

@app.route('/history')
def history():
    # Get all records from the Redis list
    raw_records = r.lrange("prediction_history", 0, -1)
    
    predictions = []
    for rec in raw_records:
        p = json.loads(rec)
        p['timestamp'] = datetime.fromisoformat(p['timestamp'])
        predictions.append(p)
        
    return render_template('history.html', predictions=predictions)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)