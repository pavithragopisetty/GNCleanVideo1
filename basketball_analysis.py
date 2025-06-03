import os
import base64
import openai
import json
import csv
import cv2
from collections import defaultdict
import logging
import argparse

# --- Initialize OpenAI client ---
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def extract_frames(video_path, output_dir="frames", step=30):
    """
    Extracts frames from a video at a regular interval and saves them to output_dir.
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
    return output_dir

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def analyze_frame(image_path):
    base64_image = encode_image(image_path)

    prompt = """
You are analyzing a youth basketball game frame. Identify:
1. Points scored in this frame, and jersey number of the player (if visible).
2. Passes that are clearly happening (ball in motion between teammates).
3. Rebound attempts or successful rebounds, and jersey number if visible.

Output JSON like:
{
  "points": { "23": 2 },
  "passes": 1,
  "rebounds": { "11": 1 }
}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You analyze basketball frames for player stats."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        max_tokens=500
    )

    return response.choices[0].message.content

def annotate_frame(image_path, annotations, output_path):
    image = cv2.imread(image_path)
    if image is None:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (0, 255, 0)
    thickness = 2
    y_offset = 30

    for key, val in annotations.items():
        label = f"{key}: {val}"
        cv2.putText(image, label, (10, y_offset), font, font_scale, color, thickness)
        y_offset += 25

    cv2.imwrite(output_path, image)

def analyze_frames(folder_path, output_dir="output"):
    logger = logging.getLogger("basketball_analysis")
    os.makedirs(output_dir, exist_ok=True)
    annotated_frames_dir = os.path.join(output_dir, "annotated_frames")
    os.makedirs(annotated_frames_dir, exist_ok=True)

    points = defaultdict(int)
    passes = 0
    rebounds = defaultdict(int)
    frame_data_list = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".jpg"):
            image_path = os.path.join(folder_path, filename)
            logger.info(f"Analyzing {filename}...")

            try:
                result = analyze_frame(image_path)
                logger.info(f"Result for {filename}: {result}")

                # Robustly extract JSON from the response
                cleaned = result.strip()
                if "```json" in cleaned:
                    cleaned = cleaned.split("```json")[1]
                if "```" in cleaned:
                    cleaned = cleaned.split("```", 1)[0]
                cleaned = cleaned.strip()
                frame_data = json.loads(cleaned)
                frame_data["frame"] = filename
                frame_data_list.append(frame_data)

                # Safely handle points
                points_data = frame_data.get("points", {})
                if not isinstance(points_data, dict):
                    points_data = {}
                for jersey, score in points_data.items():
                    points[jersey] += score

                # Safely handle passes
                passes_val = frame_data.get("passes", 0)
                if not isinstance(passes_val, int):
                    try:
                        passes_val = int(passes_val)
                    except Exception:
                        passes_val = 0
                passes += passes_val

                # Safely handle rebounds
                rebounds_data = frame_data.get("rebounds", {})
                if not isinstance(rebounds_data, dict):
                    rebounds_data = {}
                for jersey, count in rebounds_data.items():
                    rebounds[jersey] += count

                # Annotate and save
                out_img_path = os.path.join(annotated_frames_dir, f"annotated_{filename}")
                annotate_frame(image_path, {
                    "Points": points_data,
                    "Passes": passes_val,
                    "Rebounds": rebounds_data
                }, out_img_path)

            except Exception as e:
                logger.error(f"Parsing error for {filename}: {e}", exc_info=True)

    # Save .json
    try:
        summary_json_path = os.path.join(output_dir, "summary.json")
        logger.info(f"Writing summary.json to {summary_json_path} with data: {frame_data_list}")
        with open(summary_json_path, "w") as jf:
            json.dump(frame_data_list, jf, indent=2)
        logger.info("summary.json written successfully")
    except Exception as e:
        logger.error(f"Error writing summary.json: {e}", exc_info=True)

    # Save .csv
    try:
        summary_csv_path = os.path.join(output_dir, "summary.csv")
        logger.info(f"Writing summary.csv to {summary_csv_path}")
        with open(summary_csv_path, "w", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=["frame", "points", "passes", "rebounds"])
            writer.writeheader()
            for row in frame_data_list:
                writer.writerow({
                    "frame": row["frame"],
                    "points": json.dumps(row.get("points", {})),
                    "passes": row.get("passes", 0),
                    "rebounds": json.dumps(row.get("rebounds", {})),
                })
        logger.info("summary.csv written successfully")
    except Exception as e:
        logger.error(f"Error writing summary.csv: {e}", exc_info=True)

    return points, passes, rebounds

def print_final_stats(csv_path):
    points = defaultdict(int)
    rebounds = defaultdict(int)
    total_passes = 0

    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                # Parse points
                frame_points = json.loads(row["points"] or "{}")
                if isinstance(frame_points, dict):
                    for jersey, score in frame_points.items():
                        points[jersey] += score

                # Parse rebounds
                frame_rebounds = json.loads(row["rebounds"] or "{}")
                if isinstance(frame_rebounds, dict):
                    for jersey, count in frame_rebounds.items():
                        rebounds[jersey] += count

                # Parse passes
                passes_val = row["passes"].strip()
                if passes_val.isdigit():
                    total_passes += int(passes_val)

            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error parsing row {row.get('frame', '?')}: {e}")

    # --- Final Output ---
    print("\n🏀 Final Game Stats from Video Analysis")
    
    print("\n1️⃣ Total Points Scored by Jersey:")
    if points:
        for jersey, score in points.items():
            print(f"   - Jersey #{jersey}: {score} point(s)")
    else:
        print("   - No points detected.")

    print("\n2️⃣ Total Passes:")
    print(f"   - {total_passes} pass(es) detected.")

    print("\n3️⃣ Total Rebounds by Jersey:")
    if rebounds:
        for jersey, count in rebounds.items():
            print(f"   - Jersey #{jersey}: {count} rebound(s)")
    else:
        print("   - No rebounds detected.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='output', help='Directory to save outputs')
    parser.add_argument('--frames_dir', default='GirlsNav_frames', help='Directory with extracted frames')
    args = parser.parse_args()

    os.makedirs(args.frames_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Extract frames from video
    video_path = "GirlsNav.mp4"
    extract_frames(video_path, output_dir=args.frames_dir, step=30)

    # Step 2: Analyze frames
    points, total_passes, rebounds = analyze_frames(args.frames_dir, output_dir=args.output_dir)

    # Step 3: Print final stats
    print_final_stats(os.path.join(args.output_dir, "summary.csv"))

if __name__ == "__main__":
    main() 