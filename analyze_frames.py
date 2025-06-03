import os
import base64
import openai
import json
import csv
import cv2
from collections import defaultdict

# --- Initialize OpenAI client (new SDK style) ---
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# --- Encode image to base64 ---
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# --- Ask GPT to analyze a single frame ---
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

IMPORTANT: Only output the JSON object, no additional text or explanation.
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You analyze basketball frames for player stats. Only output valid JSON."},
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

# --- Annotate a frame with overlays ---
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

# --- Main loop over all frames ---
def aggregate_stats(folder_path, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotated_frames"), exist_ok=True)

    points = defaultdict(int)
    passes = 0
    rebounds = defaultdict(int)
    frame_data_list = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".jpg"):
            image_path = os.path.join(folder_path, filename)
            print(f"Analyzing {filename}...")

            try:
                result = analyze_frame(image_path)
                print("Result:", result)

                # Extract JSON from the response
                cleaned = result.strip()
                if "```json" in cleaned:
                    cleaned = cleaned.split("```json")[1]
                if "```" in cleaned:
                    cleaned = cleaned.split("```")[0]
                cleaned = cleaned.strip()
                
                frame_data = json.loads(cleaned)
                frame_data["frame"] = filename
                frame_data_list.append(frame_data)

                for jersey, score in frame_data.get("points", {}).items():
                    points[jersey] += score
                passes += frame_data.get("passes", 0)
                for jersey, count in frame_data.get("rebounds", {}).items():
                    rebounds[jersey] += count

                # Annotate and save
                out_img_path = os.path.join(output_dir, "annotated_frames", f"annotated_{filename}")
                annotate_frame(image_path, {
                    "Points": frame_data.get("points", {}),
                    "Passes": frame_data.get("passes", 0),
                    "Rebounds": frame_data.get("rebounds", {})
                }, out_img_path)

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                # Add empty frame data for failed frames
                frame_data_list.append({
                    "frame": filename,
                    "points": {},
                    "passes": 0,
                    "rebounds": {}
                })

    # Save .json
    with open(os.path.join(output_dir, "summary.json"), "w") as jf:
        json.dump(frame_data_list, jf, indent=2)

    # Save .csv
    with open(os.path.join(output_dir, "summary.csv"), "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=["frame", "points", "passes", "rebounds"])
        writer.writeheader()
        for row in frame_data_list:
            writer.writerow({
                "frame": row["frame"],
                "points": json.dumps(row.get("points", {})),
                "passes": row.get("passes", 0),
                "rebounds": json.dumps(row.get("rebounds", {})),
            })

    return points, passes, rebounds

# --- Run it ---
if __name__ == "__main__":
    input_folder = os.path.join(os.path.dirname(__file__), "GirlsNav_frames")
    points, total_passes, rebounds = aggregate_stats(input_folder)

    print("\n--- Final Stats ---")
    print("Points Per Player:", dict(points))
    print("Total Passes:", total_passes)
    print("Rebounds Per Player:", dict(rebounds))
