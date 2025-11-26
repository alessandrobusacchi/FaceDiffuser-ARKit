import subprocess
import os

# Folder containing your PNG frames
frames_folder = "result/unreal/M009_029_5_2_unreal"

audioFile = "data/mead_arkit/wav/M009_029_5_2.wav"

# Output video filename
output_video = "output_with_audio.mp4"

# Build full input path pattern
input_pattern = f"{frames_folder}.%04d.png"

# FFmpeg command
cmd = [
    "ffmpeg",
    "-framerate", "30",
    "-i", input_pattern,     # PNG sequence
    "-i", audioFile,         # Audio track
    "-pix_fmt", "yuv420p",
    "-crf", "18",
    "-c:a", "aac",           # Encode audio for MP4
    output_video
]

print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True)

print("Done! MP4 saved as", output_video)
