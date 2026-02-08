import numpy as np
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str, default="")
    parser.add_argument("--gt_path", type=str, default="")
    args = parser.parse_args()


    cnt = 0
    motion_std_difference = []
    abs_motion_std_difference = []

    mve = 0
    lve = 0
    num_seq = 0

    mouth_mask = list(range(94, 114)) + list(range(146, 178)) + list(range(183, 192))
    upper_mask = [x for x in range(192) if x not in mouth_mask]
    for file in os.listdir(args.pred_path):
        if file.endswith('.npy'):
            seq_name = "_".join(os.path.basename(file).split('.')[0].split('_')[:2])

            gt_seq = np.load(os.path.join(args.gt_path, seq_name + ".npy"))

            pred_seq = np.load(os.path.join(args.pred_path, file))

            pred_seq = pred_seq[:gt_seq.shape[0], :]
            gt_seq = gt_seq[:pred_seq.shape[0], :]

            mve += np.linalg.norm(pred_seq - gt_seq, axis = 1).mean()
            lve += np.linalg.norm(pred_seq[:, mouth_mask] - gt_seq[:, mouth_mask], axis=1).mean()

            cnt += pred_seq.shape[0]

            L2_dis_upper = np.array([np.square(gt_seq[:, v]) for v in upper_mask])
            L2_dis_upper = np.transpose(L2_dis_upper, (1, 0))
            L2_dis_upper = np.sum(L2_dis_upper, axis=1)
            L2_dis_upper = np.std(L2_dis_upper, axis=0)
            gt_motion_std = np.mean(L2_dis_upper)

            L2_dis_upper = np.array([np.square(pred_seq[:, v]) for v in upper_mask])
            L2_dis_upper = np.transpose(L2_dis_upper, (1, 0))
            L2_dis_upper = np.sum(L2_dis_upper, axis=1)
            L2_dis_upper = np.std(L2_dis_upper, axis=0)
            pred_motion_std = np.mean(L2_dis_upper)

            motion_std_difference.append(gt_motion_std - pred_motion_std)
            abs_motion_std_difference.append(np.abs(gt_motion_std - pred_motion_std))
            num_seq += 1

    print('Frame Number: {}'.format(cnt))

    print('Mean Vertex Error: {:.4e}'.format(mve / num_seq))
    print('Lip Vertex Error: {:.4e}'.format(lve / num_seq))
    print('FDD: {:.4e}'.format(sum(motion_std_difference) / len(motion_std_difference)))
    print('ABS FDD: {:.4e}'.format(sum(abs_motion_std_difference) / len(motion_std_difference)))


def main_beat():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str, default="")
    parser.add_argument("--gt_path", type=str, default="")
    args = parser.parse_args()


    cnt = 0
    motion_std_difference = []
    abs_motion_std_difference = []

    mve = 0
    lve = 0
    num_seq = 0

    #upper_mask = [
    #    8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 49, 50
    #]

    #mouth_mask = [
    #    23, 25, 27, 28, 31, 37, 39, 40, 41, 42, 47, 48
    #]

    upper_mask = [
        8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 49, 50
    ]

    mouth_mask = [
        23, 25, 27, 28, 31, 37, 39, 40, 41, 42, 47, 48
    ]

    for file in os.listdir(args.pred_path):
        if file.endswith('.npy'):
            seq_name = "_".join(os.path.basename(file).split('.')[0].split('_')[:3])

            gt_seq = np.load(os.path.join(args.gt_path, seq_name + ".npy"))
            pred_seq = np.load(os.path.join(args.pred_path, file))

            pred_seq = pred_seq[:gt_seq.shape[0], :]
            gt_seq = gt_seq[:pred_seq.shape[0], :]

            mve += np.linalg.norm(pred_seq - gt_seq, axis = 1).mean()
            lve += np.linalg.norm(pred_seq[:, mouth_mask] - gt_seq[:, mouth_mask], axis=1).mean()

            cnt += pred_seq.shape[0]

            L2_dis_upper = np.array([np.square(gt_seq[:, v]) for v in upper_mask])
            L2_dis_upper = np.transpose(L2_dis_upper, (1, 0))
            L2_dis_upper = np.sum(L2_dis_upper, axis=1)
            L2_dis_upper = np.std(L2_dis_upper, axis=0)
            gt_motion_std = np.mean(L2_dis_upper)

            L2_dis_upper = np.array([np.square(pred_seq[:, v]) for v in upper_mask])
            L2_dis_upper = np.transpose(L2_dis_upper, (1, 0))
            L2_dis_upper = np.sum(L2_dis_upper, axis=1)
            L2_dis_upper = np.std(L2_dis_upper, axis=0)
            pred_motion_std = np.mean(L2_dis_upper)

            motion_std_difference.append(gt_motion_std - pred_motion_std)
            abs_motion_std_difference.append(np.abs(gt_motion_std - pred_motion_std))
            num_seq += 1

    print('Frame Number: {}'.format(cnt))

    print('Mean Vertex Error: {:.4e}'.format(mve / num_seq))
    print('Lip Vertex Error: {:.4e}'.format(lve / num_seq))
    print('FDD: {:.4e}'.format(sum(motion_std_difference) / len(motion_std_difference)))
    print('ABS FDD: {:.4e}'.format(sum(abs_motion_std_difference) / len(motion_std_difference)))


def main_mead_arkit():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str, default="result")
    parser.add_argument("--gt_path", type=str, default="data/mead_arkit/arkit")
    args = parser.parse_args()


    cnt = 0
    motion_std_difference = []
    abs_motion_std_difference = []

    mve = 0
    lve = 0
    num_seq = 0
    total_mee = 0

    mouth_mask = [
        22,  # jawForward
        23,  # jawLeft
        24,  # jawOpen
        25,  # jawRight
        26,  # mouthClose
        27,  # mouthDimpleLeft
        28,  # mouthDimpleRight
        29,  # mouthFrownLeft
        30,  # mouthFrownRight
        31,  # mouthFunnel
        32,  # mouthLeft
        33,  # mouthLowerDownLeft
        34,  # mouthLowerDownRight
        35,  # mouthPressLeft
        36,  # mouthPressRight
        37,  # mouthPucker
        38,  # mouthRight
        39,  # mouthRollLower
        40,  # mouthRollUpper
        41,  # mouthShrugLower
        42,  # mouthShrugUpper
        43,  # mouthSmileLeft
        44,  # mouthSmileRight
        45,  # mouthStretchLeft
        46,  # mouthStretchRight
        47,  # mouthUpperUpLeft
        48,  # mouthUpperUpRight
    ]

    upper_mask = [
        0,  # browDownLeft
        1,  # browDownRight
        2,  # browInnerUp
        3,  # browOuterUpLeft
        4,  # browOuterUpRight
        8,  # eyeBlinkLeft
        9,  # eyeBlinkRight
        10,  # eyeLookDownLeft
        11,  # eyeLookDownRight
        12,  # eyeLookInLeft
        13,  # eyeLookInRight
        14,  # eyeLookOutLeft
        15,  # eyeLookOutRight
        16,  # eyeLookUpLeft
        17,  # eyeLookUpRight
        18,  # eyeSquintLeft
        19,  # eyeSquintRight
        20,  # eyeWideLeft
        21,  # eyeWideRight
    ]

    # are old wrong?? to ask

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    pred_path = (project_root / args.pred_path).resolve()
    gt_path = (project_root / args.gt_path).resolve()

    print("Pred path:", pred_path)
    print("GT path:", gt_path)

    print(os.listdir(pred_path))
    for file in os.listdir(pred_path):
        if file.endswith('.npy'):

            gt_seq = np.load(os.path.join(gt_path, file))
            pred_seq = np.load(os.path.join(pred_path, file))

            pred_seqs_MEE = [np.load(os.path.join(pred_path, f"{file[:-4]}_sample{i}.npy")) for i in range(10)]
            pred_seqs_MEE = [p[:gt_seq.shape[0], :] for p in pred_seqs_MEE]
            mean_pred = np.mean(pred_seqs_MEE, axis=0)
            mee_seq = np.linalg.norm(mean_pred[:, mouth_mask] - gt_seq[:, mouth_mask], axis=1).mean()

            total_mee += mee_seq

            pred_seq = pred_seq[:gt_seq.shape[0], :]
            gt_seq = gt_seq[:pred_seq.shape[0], :]

            mve += np.linalg.norm(pred_seq - gt_seq, axis = 1).mean()
            lve += np.linalg.norm(pred_seq[:, mouth_mask] - gt_seq[:, mouth_mask], axis=1).mean()

            cnt += pred_seq.shape[0]

            L2_dis_upper = np.array([np.square(gt_seq[:, v]) for v in upper_mask])
            L2_dis_upper = np.transpose(L2_dis_upper, (1, 0))
            L2_dis_upper = np.sum(L2_dis_upper, axis=1)
            L2_dis_upper = np.std(L2_dis_upper, axis=0)
            gt_motion_std = np.mean(L2_dis_upper)

            L2_dis_upper = np.array([np.square(pred_seq[:, v]) for v in upper_mask])
            L2_dis_upper = np.transpose(L2_dis_upper, (1, 0))
            L2_dis_upper = np.sum(L2_dis_upper, axis=1)
            L2_dis_upper = np.std(L2_dis_upper, axis=0)
            pred_motion_std = np.mean(L2_dis_upper)

            motion_std_difference.append(gt_motion_std - pred_motion_std)
            abs_motion_std_difference.append(np.abs(gt_motion_std - pred_motion_std))
            num_seq += 1

    print('Frame Number: {}'.format(cnt))

    print('Mean Vertex Error: {:.4e}'.format(mve / num_seq))
    print('Lip Vertex Error: {:.4e}'.format(lve / num_seq))
    print('MEE: {:.4e}'.format(sum(total_mee) / len(num_seq)))
    print('FDD: {:.4e}'.format(sum(motion_std_difference) / len(motion_std_difference)))
    print('ABS FDD: {:.4e}'.format(sum(abs_motion_std_difference) / len(motion_std_difference)))


if __name__ == "__main__":
    main_mead_arkit()
