import bpy
import numpy as np
import os
import sys

# ----------------------------
# --- Command-line args ------
# ----------------------------
filename = str(sys.argv[-1])
root_dir = str(sys.argv[-2])

seq_pickle = np.load(root_dir + filename + '.npy')

# ----------------------------
# --- Render settings --------
# ----------------------------
scene = bpy.context.scene
scene.render.engine = 'BLENDER_WORKBENCH'
scene.display.shading.light = 'MATCAP'
scene.display.render_aa = 'FXAA'
scene.render.resolution_x = int(1280)
scene.render.resolution_y = int(720)
scene.render.fps = 30
scene.render.image_settings.file_format = 'PNG'

cam = bpy.data.objects['Camera']
scene.camera = cam

face_obj = scene.objects["Face"]
face_obj.select_set(True)
if face_obj.animation_data:
    face_obj.animation_data_clear()

sk_blocks = face_obj.data.shape_keys.key_blocks

# ---------------------------------------------------
# ARKit blendshape names
# ---------------------------------------------------
arkit_names = [
    'browDownLeft', 'browDownRight', 'browInnerUp',
    'browOuterUpLeft', 'browOuterUpRight', 'cheekPuff',
    'cheekSquintLeft', 'cheekSquintRight', 'eyeBlinkLeft',
    'eyeBlinkRight', 'eyeLookDownLeft', 'eyeLookDownRight',
    'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft',
    'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight',
    'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft',
    'eyeWideRight', 'jawForward', 'jawLeft', 'jawOpen',
    'jawRight', 'mouthClose', 'mouthDimpleLeft',
    'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight',
    'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft',
    'mouthLowerDownRight', 'mouthPressLeft', 'mouthPressRight',
    'mouthPucker', 'mouthRight', 'mouthRollLower',
    'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper',
    'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft',
    'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight',
    'noseSneerLeft', 'noseSneerRight', 'tongueOut'
]

# ---------------------------------------------------
# Match ARKit names to Blender shape keys (case-insensitive)
# ---------------------------------------------------
def find_matching_key(arkit_name):
    arkit_norm = arkit_name.lower().replace("_", "")
    for sk_name in sk_blocks.keys():
        if arkit_norm in sk_name.lower().replace("_", ""):
            return sk_name
    return None

name_map = {}
for n in arkit_names:
    match = find_matching_key(n)
    if match:
        name_map[n] = match
    else:
        print(f"No matching shape key for {n}")

print(f"Found {len(name_map)} mapped blendshapes")

# ---------------------------------------------------
# Animation loop
# ---------------------------------------------------

output_dir = root_dir + filename
os.makedirs(output_dir, exist_ok=True)


for frame, frame_values in enumerate(seq_pickle):
    bpy.context.scene.frame_set(frame)

    for idx, value in enumerate(frame_values):
        arkit = arkit_names[idx]
        if arkit not in name_map:
            continue
        sk_name = name_map[arkit]
        sk_blocks[sk_name].value = float(value)

    bpy.context.view_layer.update()

    # Render current frame
    scene.render.filepath = os.path.join(output_dir, f"{frame:04d}.png")
    bpy.ops.render.render(write_still=True)

print(f"Finished rendering frames to: {output_dir}")