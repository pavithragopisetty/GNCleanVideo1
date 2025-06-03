import cv2
import os

def extract_frames(video_path, output_dir="frames", step=30):
    """
    Extracts frames from a video at a regular interval and saves them to output_dir.

    Args:
        video_path (str): Path to the input video.
        output_dir (str): Directory where frames will be saved.
        step (int): Frame interval to extract. (e.g., every 30 frames)
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    if not cap.isOpened():
        print("❌ Error: Cannot open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % step == 0:
            frame_filename = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Saved: {frame_filename}")
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"✅ Done. Extracted {saved_count} frames to folder: {output_dir}")

if __name__ == "__main__":
    # For testing locally
    test_video = "GirlsNav.mp4"
    extract_frames(test_video, output_dir="GirlsNav_frames", step=30)
