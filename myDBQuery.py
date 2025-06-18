from app import User, Video, app, db, LoginRequest, Feedback

with app.app_context():
    users = User.query.all()
    for user in users:
        print(f"User: {user.first_name} {user.last_name} ({user.email})")
        videos = Video.query.filter_by(user_email=user.email).all()
        if videos:
            for video in videos:
                print(f"  - Video: {video.filename}, Session ID: {video.session_id}, Uploaded: {video.upload_time}")
        else:
            print("  - No videos uploaded.")
        print()  # Blank line between users

def query_login_requests():
    with app.app_context():
        db.create_all()  # Create tables if they don't exist
        login_requests = LoginRequest.query.order_by(LoginRequest.timestamp.desc()).all()
        print("Emails that requested login links:")
        for request in login_requests:
            print(f"Email: {request.email}, Timestamp: {request.timestamp}")

def query_feedback():
    with app.app_context():
        db.create_all()  # Ensure tables exist
        feedbacks = Feedback.query.order_by(Feedback.submitted_at.desc()).all()
        print("\nFeedback Entries:")
        for fb in feedbacks:
            print(f"User: {fb.user_email}, Session: {fb.video_session_id}, Rating: {fb.rating}, Text: {fb.feedback_text}, Submitted: {fb.submitted_at}")

if __name__ == '__main__':
    query_login_requests()
    query_feedback()