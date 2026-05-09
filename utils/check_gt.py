import os
import shutil
import random

SOURCE_DIR = "../data/mead_arkit/arkit"
TARGET_DIR = "../result"

os.makedirs(TARGET_DIR, exist_ok=True)

# expressive emotions
expressive_emotions = {"1", "2", "3", "4", "5", "6"}  # happy, surprised, fear, angry

# intensities: 1 -> level_2, 2 -> level_3
expressive_intensities = {"1", "2"}

# dictionary: subject -> list of candidate files
subject_files = {}

for fname in os.listdir(SOURCE_DIR):
    if not fname.endswith(".npy"):
        continue

    parts = fname.replace(".npy", "").split("_")
    if len(parts) != 4:
        continue

    subject, sentence, emo, intensity = parts

    # filter by emotion and intensity
    if emo in expressive_emotions and intensity in expressive_intensities:
        subject_files.setdefault(subject, []).append(fname)

# sample and copy
for subject, files in subject_files.items():
    sample = random.sample(files, min(50, len(files)))

    # subject_dir = os.path.join(TARGET_DIR, subject)
    # os.makedirs(subject_dir, exist_ok=True)

    for fname in sample:
        src = os.path.join(SOURCE_DIR, fname)
        dst = os.path.join(TARGET_DIR, fname)
        shutil.copy(src, dst)

    print(f"{subject}: selected {len(sample)} expressive sequences")
