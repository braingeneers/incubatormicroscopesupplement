import cv2
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
import csv
import os

"""
Single‑organoid tracker for bright (white/grey) objects on a dark background
--------------------------------------------------------------------------
* Enhanced contrast and relaxed edge detection to better capture faint boundaries.
* Saves one metrics CSV, one optional contours CSV, and a visual overlay video.
"""

# -------------------- USER CONFIG --------------------
video_path = "/Coding/DrewOrganoidTrackingandGraphing/input/santhosh4x_longorganoid_COMPLETE.mp4"
output_folder = "/Coding/DrewOrganoidTrackingandGraphing/output"
frame_skip = 1               # analyse every N‑th frame
min_area = 2000                   # further reduced to capture smaller/fragmented shapes

total_duration = timedelta(days=14)
start_time = datetime(2025, 5, 5, 16, 0, 0)
# -----------------------------------------------------

contour_data = []   # frame‑wise quantitative metrics
all_contours = []   # optional raw contour points for debugging


def process_frame(frame, timestamp):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(6, 6))  # stronger CLAHE
    enhanced = clahe.apply(gray)

    # Slight smoothing to boost faint gradients
    blurred = cv2.GaussianBlur(enhanced, (9, 9), sigmaX=2.0)  # broader blur

    # Otsu threshold
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphology with smaller kernel, fewer iterations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return frame

    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < min_area:
        return frame

    all_contours.append((cnt, timestamp))

    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    M = cv2.moments(cnt)
    cx = int(M["m10"] / M["m00"]) if M["m00"] else 0
    cy = int(M["m01"] / M["m00"]) if M["m00"] else 0
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / h
    shape_factor = (4 * np.pi * area) / (perimeter ** 2) if perimeter else 0
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area else 0
    extent = area / (w * h) if w * h else 0
    eccentricity = np.sqrt(1 - (min(w, h) / max(w, h)) ** 2) if max(w, h) else 0

    contour_data.append({
        "timestamp": timestamp,
        "area": area,
        "perimeter": perimeter,
        "centroid_x": cx,
        "centroid_y": cy,
        "aspect_ratio": aspect_ratio,
        "shape_factor": shape_factor,
        "solidity": solidity,
        "extent": extent,
        "eccentricity": eccentricity,
    })

    # Overlay
    cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
    label = f"Area: {int(area)} | Solidity: {solidity:.2f}"
    cv2.putText(frame, label, (x, y - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


def main():
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    frame_period = total_duration / total_frames if total_duration else timedelta(seconds=1 / fps)
    overlay_path = os.path.join(output_folder, "tracked_overlay.avi")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(overlay_path, fourcc, fps // frame_skip, (width, height))

    frame_idx = 0
    with tqdm(total=total_frames // frame_skip, desc="Processing", unit="frame") as bar:
        while cap.isOpened():
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                cap.grab()
                continue

            ret, frame = cap.read()
            if not ret:
                break

            ts = start_time + frame_idx * frame_period
            result_frame = process_frame(frame, ts)
            out.write(result_frame)

            frame_idx += 1
            bar.update(1)

    cap.release()
    out.release()

    metrics_file = os.path.join(output_folder, "white_organoid_metrics.csv")
    if contour_data:
        with open(metrics_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=contour_data[0].keys())
            writer.writeheader()
            writer.writerows(contour_data)
        print(f"Saved contour metrics → {metrics_file}")

    contours_file = os.path.join(output_folder, "white_organoid_all_contours.csv")
    if all_contours:
        with open(contours_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "contour_points"])
            for cnt, ts in all_contours:
                pts = [[p[0][0], p[0][1]] for p in cnt]
                writer.writerow([ts, pts])
        print(f"Saved raw contours     → {contours_file}")

    print(f"Overlay video saved    → {overlay_path}")


if __name__ == "__main__":
    main()


# # ------------------ Save Figure -----------------
# # Export the composite figure as a PDF in the same output folder  
# do you figure a figure's fig