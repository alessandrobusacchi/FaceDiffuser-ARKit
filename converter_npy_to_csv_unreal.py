import numpy as np
import csv
import os
from pathlib import Path

# ---- SETTINGS ----
FPS = 30
blendshape_names = [
    "BrowDownLeft","BrowDownRight","BrowInnerUp","BrowOuterUpLeft","BrowOuterUpRight",
    "CheekPuff","CheekSquintLeft","CheekSquintRight","EyeBlinkLeft","EyeBlinkRight",
    "EyeLookDownLeft","EyeLookDownRight","EyeLookInLeft","EyeLookInRight",
    "EyeLookOutLeft","EyeLookOutRight","EyeLookUpLeft","EyeLookUpRight",
    "EyeSquintLeft","EyeSquintRight","EyeWideLeft","EyeWideRight",
    "JawForward","JawLeft","JawOpen","JawRight","MouthClose",
    "MouthDimpleLeft","MouthDimpleRight","MouthFrownLeft","MouthFrownRight",
    "MouthFunnel","MouthLeft","MouthLowerDownLeft","MouthLowerDownRight",
    "MouthPressLeft","MouthPressRight","MouthPucker","MouthRight",
    "MouthRollLower","MouthRollUpper","MouthShrugLower","MouthShrugUpper",
    "MouthSmileLeft","MouthSmileRight","MouthStretchLeft","MouthStretchRight",
    "MouthUpperUpLeft","MouthUpperUpRight","NoseSneerLeft","NoseSneerRight"
]
# ------------------

def frame_to_timecode(idx, fps):
    h = idx // (3600 * fps)
    m = (idx // (60 * fps)) % 60
    s = (idx // fps) % 60
    f = idx % fps
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"

# Directories
result_dir = Path("result")
conv_dir = result_dir / "conversions"
conv_dir.mkdir(exist_ok=True)


# List all .npy files in result/
npy_files = sorted(result_dir.glob("*.npy"))

print("Found", len(npy_files), "npy files")

for npy_path in npy_files:
    filename = npy_path.stem                     # e.g. W036_028_7_1
    csv_path = conv_dir / f"{filename}_unreal.csv"
    print("Converting:", filename)

    arr = np.load(npy_path)

    frames, n_shapes = arr.shape
    if n_shapes != len(blendshape_names):
        print(f"  ERROR: {filename}: expected {len(blendshape_names)} blendshapes, got {n_shapes}")
        continue

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Timecode", "BlendshapeCount"] + blendshape_names)

        for i, weights in enumerate(arr):
            tc = frame_to_timecode(i, FPS)
            writer.writerow([tc, n_shapes] + weights.tolist())

    print("  → Saved:", csv_path)

print("All conversions complete.")