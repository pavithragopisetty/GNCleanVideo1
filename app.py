import os
from flask import Flask, request, render_template, jsonify, send_file, redirect, url_for, session
from werkzeug.utils import secure_filename
import basketball_analysis
import uuid
import shutil
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
from datetime import datetime
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable static file caching
app.jinja_env.auto_reload = True  # Enable Jinja2 auto-reload
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi'}

# Flask-Mail config (set these in your .env or here directly)
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER', app.config['MAIL_USERNAME'])
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')
app.config['SESSION_TYPE'] = 'filesystem'

# SQLAlchemy config
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

mail = Mail(app)
serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])

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

@app.template_filter('fromjson')
def fromjson_filter(s):
    import json
    return json.loads(s)

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
    video_filename = secure_filename(file.filename)
    video_path = os.path.join(session_dir, video_filename)
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
            'session_id': session_id,
            'video_filename': video_filename
        }
        
        # Save video info to DB if user is logged in
        user_email = session.get('user_email')
        if user_email:
            try:
                video = Video(
                    user_email=user_email,
                    filename=video_filename,
                    session_id=session_id,
                    points=json.dumps(dict(points)),
                    total_passes=total_passes,
                    rebounds=json.dumps(dict(rebounds))
                )
                db.session.add(video)
                db.session.commit()
                app.logger.info(f'Saved video analysis to database for user {user_email}')
            except Exception as db_error:
                app.logger.error(f'Error saving to database: {str(db_error)}')
                # Continue even if DB save fails
        
        app.logger.info(f'Analysis complete for session {session_id}')
        return jsonify(results)

    except Exception as e:
        app.logger.error(f'Error during analysis: {str(e)}', exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/video/<session_id>/<filename>')
def serve_video(session_id, filename):
    try:
        return send_file(
            os.path.join(app.config['UPLOAD_FOLDER'], session_id, filename),
            mimetype='video/mp4'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 404

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

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/blogs')
def blogs():
    return render_template('blogs.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/send-login-link', methods=['POST'])
def send_login_link():
    email = request.form.get('email')
    if not email:
        return render_template('login.html', error='Please enter your email.')
    # Check if user exists in the database
    user = User.query.filter_by(email=email).first()
    if not user:
        return render_template('login.html', error='No account found with that email. Please sign up first.')
    # Generate token
    token = serializer.dumps(email, salt='login-salt')
    login_url = 'https://girlsnav.com/magic-login/' + token
    # Send email
    try:
        msg = Message('Your GirlsNav Login Link', recipients=[email])
        msg.html = f'''
            <html>
                <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                    <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                        <h2 style="color: #111111;">Welcome to GirlsNav!</h2>
                        <p>Click the button below to log in to your account:</p>
                        <div style="text-align: center; margin: 30px 0;">
                            <a href="{login_url}" style="background-color: #FFC72C; color: #000; padding: 12px 24px; text-decoration: none; border-radius: 6px; font-weight: bold; display: inline-block;">Log In to GirlsNav</a>
                        </div>
                        <p style="color: #666; font-size: 14px;">This link will expire in 15 minutes.</p>
                        <hr style="border: 1px solid #eee; margin: 20px 0;">
                        <p style="color: #666; font-size: 14px;">If the button above doesn't work, copy and paste this URL into your browser:</p>
                        <p style="background: #f5f5f5; padding: 10px; border-radius: 4px; word-break: break-all; font-size: 14px;">{login_url}</p>
                    </div>
                </body>
            </html>
        '''
        msg.body = f'Click the link to log in: {login_url}\n\nThis link will expire in 15 minutes.'
        mail.send(msg)
        return render_template('login.html', message='A login link has been sent to your email.')
    except Exception as e:
        app.logger.error(f'Error sending login email: {e}')
        return render_template('login.html', error='Failed to send email. Please try again later.')

@app.route('/magic-login/<token>')
def magic_login(token):
    try:
        email = serializer.loads(token, salt='login-salt', max_age=900)  # 15 min expiry
        session.clear()  # Clear any existing session data
        session['user_email'] = email
        session.permanent = True  # Make the session persistent
        user = User.query.filter_by(email=email).first()
        first_name = user.first_name if user else None
        welcome_message = f'Welcome back to GirlsNav AI!'
        videos = Video.query.filter_by(user_email=email).order_by(Video.upload_time.desc()).all()
        return render_template('upload_stats.html', email=email, first_name=first_name, welcome_message=welcome_message, videos=videos)
    except SignatureExpired:
        return 'This login link has expired.'
    except BadSignature:
        return 'Invalid login link.'

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if session.get('user_email'):
        return render_template('upload_stats.html', email=session['user_email'], welcome_message='Welcome back to GirlsNav AI!')
    if request.method == 'POST':
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        email = request.form.get('email')
        state = request.form.get('state')
        phone = request.form.get('phone')
        terms = request.form.get('terms')
        # Check if user already exists
        if User.query.filter_by(email=email).first():
            return render_template('signup.html', message='Email already registered. Please log in.')
        user = User(first_name=first_name, last_name=last_name, email=email, state=state, phone=phone)
        db.session.add(user)
        db.session.commit()
        return render_template('signup_success.html')
    return render_template('signup.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        # You can add logic to handle form submission here (e.g., send email, store in DB)
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        message = request.form.get('message')
        # For now, just show a thank you message
        return render_template('contact.html', success=True, name=name)
    return render_template('contact.html', success=False)

@app.route('/dashboard')
def dashboard():
    if not session.get('user_email'):
        return redirect(url_for('login'))
    user_email = session.get('user_email')
    user = User.query.filter_by(email=user_email).first()
    first_name = user.first_name if user else None
    welcome_message = f'Welcome back to GirlsNav AI!'
    videos = Video.query.filter_by(user_email=user_email).order_by(Video.upload_time.desc()).all()
    return render_template('upload_stats.html', email=user_email, first_name=first_name, welcome_message=welcome_message, videos=videos)

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Resource not found (404)'}), 404

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum allowed size is 50MB.'}), 413

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error (500)'}), 500

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(50))
    last_name = db.Column(db.String(50))
    email = db.Column(db.String(120), unique=True)
    state = db.Column(db.String(2))
    phone = db.Column(db.String(20))

class Video(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String(120), db.ForeignKey('user.email'))
    filename = db.Column(db.String(256))
    session_id = db.Column(db.String(64))
    upload_time = db.Column(db.DateTime, default=datetime.utcnow)
    points = db.Column(db.Text)  # Store as JSON string
    total_passes = db.Column(db.Integer)
    rebounds = db.Column(db.Text)  # Store as JSON string

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
