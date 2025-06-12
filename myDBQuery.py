from app import User, Video, app

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