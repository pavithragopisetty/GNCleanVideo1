import os
from flask import Flask, request, render_template, jsonify, send_file
from werkzeug.utils import secure_filename
import basketball_analysis
import uuid
import shutil
import logging
from logging.handlers import RotatingFileHandler

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi'}

# Set up logging
if not os.path.exists('logs'):
    os.mkdir('logs')
file_handler = RotatingFileHandler('logs/basketball-analysis.log', maxBytes=10240, backupCount=10)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(logging.INFO)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)
app.logger.info('Basketball Analysis startup')

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        app.logger.warning('No video file provided in request')
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        app.logger.warning('Empty filename provided')
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        app.logger.warning(f'Invalid file type: {file.filename}')
        return jsonify({'error': 'Invalid file type'}), 400

    # Create a unique session ID for this analysis
    session_id = str(uuid.uuid4())
    session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    os.makedirs(session_dir, exist_ok=True)
    app.logger.info(f'Created session {session_id}')

    # Save the uploaded video
    video_path = os.path.join(session_dir, secure_filename(file.filename))
    file.save(video_path)
    app.logger.info(f'Saved video to {video_path}')

    try:
        # Run the analysis
        app.logger.info('Starting frame extraction')
        frames_dir = basketball_analysis.extract_frames(video_path, output_dir=os.path.join(session_dir, "frames"))
        
        app.logger.info('Starting frame analysis')
        points, total_passes, rebounds = basketball_analysis.analyze_frames(frames_dir, output_dir=os.path.join(session_dir, "output"))
        
        # Prepare the results
        results = {
            'points': dict(points),
            'total_passes': total_passes,
            'rebounds': dict(rebounds),
            'session_id': session_id
        }
        
        app.logger.info(f'Analysis complete for session {session_id}')
        return jsonify(results)

    except Exception as e:
        app.logger.error(f'Error during analysis: {str(e)}', exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/download/<session_id>/<filename>')
def download_file(session_id, filename):
    try:
        return send_file(
            os.path.join(app.config['UPLOAD_FOLDER'], session_id, filename),
            as_attachment=True
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/cleanup/<session_id>', methods=['POST'])
def cleanup_session(session_id):
    try:
        session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        if os.path.exists(session_dir):
            shutil.rmtree(session_dir)
        return jsonify({'message': 'Cleanup successful'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 