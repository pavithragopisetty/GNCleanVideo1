import csv
import json
from collections import defaultdict

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

# --- Run it ---
if __name__ == "__main__":
    csv_file_path = "output/summary.csv"  # Update if needed
    print_final_stats(csv_file_path)
