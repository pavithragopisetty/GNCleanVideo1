import os
import base64
import openai
import json
import csv
import cv2
from collections import defaultdict
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor
from skimage.metrics import structural_similarity as ssim

# --- Initialize OpenAI client ---
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("girlsnav_analysis")

def extract_frames(video_path, output_dir="frames", step=30):
    """
    Extracts frames from a video at a regular interval and saves them to output_dir.
    """
    logger.info(f"📽 Extracting frames from: {video_path}")
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    if not cap.isOpened():
        logger.error("❌ Error: Cannot open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % step == 0:
            frame_filename = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
        frame_count += 1

    cap.release()
    logger.info(f"✅ Done. Extracted {saved_count} frames to folder: {output_dir}")
    return output_dir

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def is_significant_change(curr_path, prev_path, threshold=0.90):
    curr = cv2.imread(curr_path)
    prev = cv2.imread(prev_path)
    if curr is None or prev is None:
        return True
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(curr_gray, prev_gray, full=True)
    return score < threshold

def analyze_frame_batch(image_paths):
    logger.info(f"🔍 Analyzing batch of {len(image_paths)} frame(s)")
    images_b64 = [encode_image(p) for p in image_paths]

    content_blocks = []
    for img_b64 in images_b64:
        content_blocks.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
        })

    # Now append the instruction text block with more detailed examples
    content_blocks.append({
        "type": "text",
        "text": (
            "Analyze each basketball frame above and return stats in the following JSON format list (one object per frame).\n\n"
            "IMPORTANT: Return ONLY the raw JSON array without any markdown formatting or code blocks.\n\n"
            "For each frame, carefully identify:\n"
            "1. Points scored: Include the jersey number and points (2 or 3) in the points array\n"
            "2. Passes: Count any completed passes\n"
            "3. Rebounds: Note the jersey number of any player who gets a rebound\n\n"
            "Example format (return exactly like this, without markdown):\n"
            "[\n"
            "  {\n"
            "    \"frame\": \"frame_0001.jpg\",\n"
            "    \"points\": [\n"
            "      {\"jersey\": 23, \"points\": 2, \"confidence\": 0.95},\n"
            "      {\"jersey\": 15, \"points\": 3, \"confidence\": 0.90}\n"
            "    ],\n"
            "    \"passes\": 1,\n"
            "    \"rebounds\": {\"23\": 1, \"15\": 0}\n"
            "  },\n"
            "  {\n"
            "    \"frame\": \"frame_0002.jpg\",\n"
            "    \"points\": [],\n"
            "    \"passes\": 0,\n"
            "    \"rebounds\": {}\n"
            "  }\n"
            "]\n\n"
            "Important:\n"
            "- Always include jersey numbers for points and rebounds\n"
            "- Set confidence scores between 0.0 and 1.0\n"
            "- If no points/passes/rebounds occur, use empty arrays/objects\n"
            "- Be precise with jersey numbers and point values\n"
            "- DO NOT wrap the response in markdown code blocks or ```json tags"
        )
    })

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a precise basketball game analyzer. You carefully identify jersey numbers, points scored, passes, and rebounds in each frame. You always include jersey numbers in your analysis and maintain high accuracy in your observations. You return only raw JSON without any markdown formatting."},
            {"role": "user", "content": content_blocks}
        ],
        max_tokens=2000
    )

    try:
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"❌ Failed to parse response JSON: {e}\nRaw content: {response.choices[0].message.content}")
        return []

def analyze_frames(folder_path, output_dir="output"):
    logger.info(f"📂 Analyzing frames in folder: {folder_path}")
    os.makedirs(output_dir, exist_ok=True)
    annotated_frames_dir = os.path.join(output_dir, "annotated_frames")
    os.makedirs(annotated_frames_dir, exist_ok=True)

    frame_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".jpg")])
    logger.info(f"🔎 Found {len(frame_files)} total frame(s)")

    # Filter out similar frames
    filtered_frames = []
    for i in range(1, len(frame_files)):
        curr_path = os.path.join(folder_path, frame_files[i])
        prev_path = os.path.join(folder_path, frame_files[i-1])
        if is_significant_change(curr_path, prev_path):
            filtered_frames.append(curr_path)

    logger.info(f"⚡ {len(filtered_frames)} frame(s) retained after filtering")

    # Process frames in parallel batches
    batch_size = 5
    batches = [filtered_frames[i:i+batch_size] for i in range(0, len(filtered_frames), batch_size)]

    def process_batch(batch):
        try:
            return analyze_frame_batch(batch)
        except Exception as e:
            logger.error(f"Error analyzing batch {batch}: {e}")
            return []

    frame_data_list = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(process_batch, batches)
        for batch_result in results:
            frame_data_list.extend(batch_result)

    logger.info("✅ Frame analysis complete. Aggregating stats...")

    # Aggregate stats
    points = defaultdict(int)
    passes = 0
    rebounds = defaultdict(int)

    # Track scoring events to prevent duplicates
    scoring_events = set()  # (jersey, frame_number)
    last_scoring_frame = None
    last_scoring_jersey = None

    for frame_data in frame_data_list:
        frame_number = int(frame_data.get("frame", "0").split("_")[1].split(".")[0])
        
        # Process points with duplicate detection
        for point_event in frame_data.get("points", []):
            jersey = str(point_event.get("jersey"))
            points_value = point_event.get("points", 0)
            confidence = point_event.get("confidence", 0)
            
            # Skip low confidence events
            if confidence < 0.7:
                continue
                
            # Create unique event identifier
            event_id = (jersey, frame_number)
            
            # Check for duplicate scoring events
            if event_id in scoring_events:
                continue
                
            # Check for scoring events in consecutive frames
            if (last_scoring_frame == frame_number - 1 and 
                last_scoring_jersey == jersey):
                continue
            
            # Add to scoring events
            scoring_events.add(event_id)
            points[jersey] += points_value
            
            # Update last scoring event
            last_scoring_frame = frame_number
            last_scoring_jersey = jersey

        # Process passes
        passes += frame_data.get("passes", 0)

        # Process rebounds
        rebounds_dict = frame_data.get("rebounds") or {}
        for jersey, count in rebounds_dict.items():
            rebounds[jersey] += count

    # Save detailed results
    with open(os.path.join(output_dir, "summary.json"), "w") as jf:
        json.dump(frame_data_list, jf, indent=2)
    logger.info("📝 Saved summary.json")

    with open(os.path.join(output_dir, "summary.csv"), "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=["frame", "points", "passes", "rebounds"])
        writer.writeheader()
        for row in frame_data_list:
            writer.writerow({
                "frame": row.get("frame"),
                "points": json.dumps(row.get("points", {})),
                "passes": row.get("passes", 0),
                "rebounds": json.dumps(row.get("rebounds", {})),
            })
    logger.info("📝 Saved summary.csv")

    return points, passes, rebounds 

def main():
    logger.info("🚀 GirlsNav Analysis Started")
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', required=True, help='Path to the input video file')
    parser.add_argument('--frames_dir', default='GirlsNav_frames', help='Directory to save extracted frames')
    parser.add_argument('--output_dir', default='output', help='Directory to save analysis results')
    args = parser.parse_args()

    os.makedirs(args.frames_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    extract_frames(args.video_path, output_dir=args.frames_dir, step=30)
    points, total_passes, rebounds = analyze_frames(args.frames_dir, output_dir=args.output_dir)

    print("\n🏀 Final Game Stats:")
    print("\n1️⃣ Total Points by Jersey:")
    for jersey, pts in points.items():
        print(f"   - Jersey #{jersey}: {pts} point(s)")

    print(f"\n2️⃣ Total Passes: {total_passes} pass(es)")

    print("\n3️⃣ Total Rebounds by Jersey:")
    for jersey, rb in rebounds.items():
        print(f"   - Jersey #{jersey}: {rb} rebound(s)")

if __name__ == "__main__":
    main() 