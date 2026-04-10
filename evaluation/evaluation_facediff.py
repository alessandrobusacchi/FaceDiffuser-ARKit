import logging
import os
from pathlib import Path
import pickle
import torch
import numpy as np
import argparse
import datetime
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from models import FaceDiffMeadARKit
from data_loader import get_dataloaders
from utils import *

logger = logging.getLogger(__name__)


emo_dict = {
       "0": "neutral",   # only have one intensity level
       "1": "happy",
       "2": "sad",
       "3": "surprised",
       "4": "fear",
       "5": "disgusted",
       "6": "angry",
       "7": "contempt"
   }

int_dict = {
       "0": "low",
       "1": "medium",
       "2": "high",
   }

upper_mask = [
    0, 1, 2, 3, 4,              # Brows
    5, 6, 7,                       # Cheek
    8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,   # eyes
    49, 50                      # Nose
]

mouth_mask =[
    22, 23, 24, 25,     # Jaw
    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48  # Mouth
]

def lve_compute(vertices_gt, vertices_pred, mouth_map):
    vertices_gt = np.array(vertices_gt)
    vertices_pred = np.array(vertices_pred)
    diff = vertices_gt[:, mouth_map] - vertices_pred[:, mouth_map]
    return np.linalg.norm(diff, axis=1)


def seq_std_compute(motion, map_indices):
    # motion shape: (T, 51)
    L2_dis = np.array([np.square(motion[:, v]) for v in map_indices])
    L2_dis = np.transpose(L2_dis, (1, 0))
    L2_dis = np.sum(L2_dis, axis=1)
    L2_dis = np.std(L2_dis, axis=0)
    std = np.mean(L2_dis)
    return std


@torch.no_grad()
def test_diff(args, model, test_loader, diffusion, device):
    result_path = os.path.join(args.result_path)
    os.makedirs(result_path, exist_ok=True)

    train_subjects_list = [i for i in args.train_subjects.split(" ")]
    model_path = f'pretrained_models/{args.model_name}.pth'
    print("load model from...", model_path)
    sys.stdout.flush()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print("Device checking:", device)

    template_file = os.path.join(args.data_path, args.dataset, args.template_file)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin, encoding='latin1')

    sr = 16000
    # count frame numbers
    seq_count = 0
    frame_count = 0
    vertices_all_gt = []        # mve, lve: gt
    vertices_all_pred = []      # mve, lve: first prediction
    mee_all = []                # mean value
    ce_all = []                 # closest value
    motion_std_difference = []  # fdd: first prediction
    abs_motion_std_difference =[] # absolute fdd
    diversity = 0               # 2 subsets
    for idx, (audio, vertice, template, one_hot_all, file_name) in enumerate(test_loader):

        print(f"Processing {file_name[0]}...")
        vertice = vertice_path = str(vertice[0])
        vertice = np.load(vertice, allow_pickle=True)
        vertices_npy_gt = vertice.copy()                    # (T, 5023, 3)

        vertice_path = os.path.split(vertice_path)[-1][:-4]
        emo = emo_dict[str(vertice_path.split("_")[2])]     # retrieve emotion
        ints = int_dict[str(vertice_path.split("_")[3])]    # retrieve intensity
        gt_path = os.path.join(args.result_path, f"{vertice_path}_{emo}_{ints}_gt.npy")

        if (idx + 1) % 100 == 0:
            print(f"Saving gt: {gt_path}")
            sys.stdout.flush()
            np.save(gt_path, vertices_npy_gt)

        vertice = vertice.astype(np.float32)
        vertice = torch.from_numpy(vertice)

        if len(vertice.shape) == 2:
            vertice = torch.unsqueeze(vertice, 0)

        audio, vertice = audio.to(device=device), vertice.to(device=device)
        template, one_hot_all = template.to(device=device), one_hot_all.to(device=device)

        num_frames = int(audio.shape[-1] / sr * args.output_fps)
        shape = (1, num_frames - 1, args.vertice_dim) if num_frames < vertice.shape[1] else vertice.shape

        name = os.path.splitext(file_name[0])[0]  # removes .wav
        parts = name.split("_")

        train_subject = parts[0]
        emotion = int(parts[2])
        intensity = int(parts[3])

        sys.stdout.flush()
        assert train_subject in train_subjects_list, "train_subject not in train_subjects_list"

        subj_vec = np.eye(len(train_subjects_list))[train_subjects_list.index(train_subject)]
        emo_vec = np.eye(8)[emotion]
        int_vec = np.eye(3)[intensity]

        # Concatenate into the 35-dim vector
        one_hot = np.concatenate([subj_vec, emo_vec, int_vec])

        # Reshape for the model (Batch size 1, 35 dimensions)
        one_hot = np.reshape(one_hot, (1, -1))
        one_hot = torch.FloatTensor(one_hot).to(device=args.device)

        ce_lve_set = []  # save 10 lve values
        motion_set = []  # save 10 samples
        for sample_idx in range(1, args.num_samples + 1):
            # use ddim
            sample = diffusion.p_sample_loop(
                model,
                shape,
                clip_denoised=False,
                model_kwargs={
                    "cond_embed": audio,
                    "one_hot": one_hot,
                    "template": template,
                },
                skip_timesteps=args.skip_steps,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=None,
                dump_steps=None,
                noise=None,
                const_noise=False,
                device=device
            )
            sample = sample.squeeze()
            sample = sample.detach().cpu().numpy()

            min_len = min(sample.shape[0], vertices_npy_gt.shape[0])
            pred_seq = sample[:min_len, :]
            gt_seq = vertices_npy_gt[:min_len, :]

            """CE: compute lve for each samples"""
            ce_lve = lve_compute(vertices_gt=list(gt_seq),
                                 vertices_pred=list(pred_seq),
                                 mouth_map=mouth_mask)
            ce_lve_set.append(ce_lve)               # (T,)

            # save 10 samples
            motion_set.append(pred_seq)    # (T, 5023, 3)
            torch.cuda.empty_cache()

            if args.num_samples != 1:
                out_path = f"{vertice_path}_{emo}_{ints}_{sample_idx}.npy"
            else:
                out_path = f"{vertice_path}_{emo}_{ints}_one.npy"

            # save vertices (a small part)
            if (idx + 1) % 100 == 0:                # save every 100 keyids for checking
                if (sample_idx + 1) % 3 == 1:       # save 3 samples in 10
                    print(f"Saving pred: {out_path}")
                    np.save(os.path.join(args.result_path, out_path), sample)

        aligned_len = motion_set[0].shape[0]

        """MVE, LVE: save prediction of all audio samples"""
        vertices_all_gt.extend(list(vertices_npy_gt[:aligned_len]))
        vertices_all_pred.extend(list(motion_set[0]))

        """MEE: mean over 10 samples"""
        motion_set_stack = np.stack(motion_set, axis=0)
        vertices_npy_pred_mean = np.mean(motion_set_stack, axis=0)  # (T, 5023, 3)
        mee_lve = lve_compute(vertices_gt=list(vertices_npy_gt[:aligned_len]),
                              vertices_pred=list(vertices_npy_pred_mean),
                              mouth_map=mouth_mask)
        mee_all.extend(list(mee_lve))

        """CE: closest lve in 10 samples"""
        smallest_lve = None
        smallest_lve_value = float('inf')                           # start with an infinitely large value
        for lve_of_one_seq in ce_lve_set:
            lve_value = np.sum(lve_of_one_seq)
            if lve_value < smallest_lve_value:
                smallest_lve_value = lve_value
                smallest_lve = lve_of_one_seq                       # (T,)
        assert smallest_lve is not None, "No smallest distance found"
        ce_all.extend(list(smallest_lve))

        # count sequence, frame numbers
        frame_count += vertices_npy_gt.shape[0]
        seq_count += 1

        """FDD computation: use the first sample"""
        upper_std_gt = seq_std_compute(motion=vertices_npy_gt[:aligned_len], map_indices=upper_mask)
        upper_std_pred = seq_std_compute(motion=motion_set[0], map_indices=upper_mask)
        motion_std_difference.append(upper_std_gt - upper_std_pred)
        abs_motion_std_difference.append(np.abs(upper_std_gt - upper_std_pred))

        """Diversity computation"""
        np.random.shuffle(motion_set)                               # list of (T, 5023, 3) number=10
        subset1 = motion_set[:5]
        subset2 = motion_set[5:]
        motion_diversity = 0
        for sample1, sample2 in zip(subset1, subset2):
            motion_diversity += np.linalg.norm(sample1 - sample2, axis=1).mean()
        if len(subset1) == 5 and len(subset2) == 5:
            motion_diversity /= len(subset1)
            diversity += motion_diversity
        else:
            raise ValueError("Subset length mismatch 5")

        part_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{part_time} Done sampling: {vertice_path} ")
        sys.stdout.flush()
        torch.cuda.empty_cache()

    print('Total sequence number: {}'.format(seq_count))
    print('Total frame number: {}'.format(frame_count))
    sys.stdout.flush()
    if seq_count == 0:
        print("No sequences were processed. Unable to compute metrics.")
    else:
        """MVE computation"""
        vertices_all_gt = np.array(vertices_all_gt)  # (frame_cunt, 5023, 3)
        vertices_all_pred = np.array(vertices_all_pred)
        vertices_dis = np.linalg.norm(vertices_all_gt - vertices_all_pred, axis=1)
        print('MVE: {:.4e}'.format(np.mean(vertices_dis)))

        """LVE computation"""
        L2_dis_mouth_max = lve_compute(vertices_gt=vertices_all_gt,
                                       vertices_pred=vertices_all_pred,
                                       mouth_map=mouth_mask)
        print('LVE: {:.4e}'.format(np.mean(L2_dis_mouth_max)))

        """MEE computation"""
        print('MEE: {:.4e}'.format(np.mean(mee_all)))

        """CE computation"""
        print('CE: {:.4e}'.format(np.mean(ce_all)))

        """FDD computation"""
        print('FDD: {:.4e}'.format(sum(motion_std_difference) / len(motion_std_difference)))
        print('ABS FDD: {:.4e}'.format(sum(abs_motion_std_difference) / len(abs_motion_std_difference)))

        """Divertiy computation"""
        print('Diversity: {:.4e}'.format(diversity / seq_count))

        print(f"All the sampling are done. ")
        sys.stdout.flush()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--model_name", type=str, default="pretrained_mead_arkit")
    parser.add_argument("--dataset", type=str, default="mead_arkit", help='Name of the dataset folder. eg: BIWI')
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--vertice_dim", type=int, default=51, help='number of vertices - 23370*3 for BIWI dataset')
    parser.add_argument("--feature_dim", type=int, default=512, help='Latent Dimension to encode the inputs to')
    parser.add_argument("--gru_dim", type=int, default=512, help='GRU Vertex decoder hidden size')
    parser.add_argument("--gru_layers", type=int, default=2, help='GRU Vertex decoder hidden size')
    parser.add_argument("--wav_path", type=str, default="wav", help='path of the audio signals')
    parser.add_argument("--vertices_path", type=str, default="arkit", help='path of the ground truth')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation')
    parser.add_argument("--max_epoch", type=int, default=50, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="face_diffuser_arkit", help='name of the trained model')
    parser.add_argument("--template_file", type=str, default="templates.pkl", help='path of the train subject templates')
    parser.add_argument("--save_path", type=str, default="save", help='path of the trained models')
    parser.add_argument("--result_path", type=str, default="result", help='path to the predictions')
    parser.add_argument("--train_subjects", type=str, default="M003 M005 M007 M009 M012 M019 M022 M023 M024 M026 M027 M029 M030 M031 M032 M035 W009 W011 W015 W019 W036 W037 W038 W040")
    parser.add_argument("--val_subjects", type=str, default="M003 M005 M007 M009 M012 M019 M022 M023 M024 M026 M027 M029 M030 M031 M032 M035 W009 W011 W015 W019 W036 W037 W038 W040")
    parser.add_argument("--test_subjects", type=str, default="M003 M005 M007 M009 M012 M019 M022 M023 M024 M026 M027 M029 M030 M031 M032 M035 W009 W011 W015 W019 W036 W037 W038 W040")
    parser.add_argument("--input_fps", type=int, default=50, help='HuBERT last hidden state produces 50 fps audio representation')
    parser.add_argument("--output_fps", type=int, default=30, help='fps of the visual data, BIWI was captured in 25 fps')
    parser.add_argument("--diff_steps", type=int, default=1000, help='number of diffusion steps')
    parser.add_argument("--skip_steps", type=int, default=0, help='number of diffusion steps to skip during inference')
    parser.add_argument("--num_samples", type=int, default=10, help='number of samples to generate per audio')
    args = parser.parse_args()

    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get the start time
    print(f"Start time: {start_time}")
    sys.stdout.flush()
    assert torch.cuda.is_available()
    diffusion = create_gaussian_diffusion(args)

    model = FaceDiffMeadARKit(
        args,
        vertice_dim=args.vertice_dim,
        latent_dim=args.feature_dim,
        diffusion_steps=args.diff_steps,
        gru_latent_dim=args.gru_dim,
        num_layers=args.gru_layers,
    )
    print("model parameters: ", count_parameters(model))
    sys.stdout.flush()
    cuda = torch.device(args.device)
    dataset = get_dataloaders(args)

    test_diff(args, model, dataset["test"], diffusion, device=args.device)

    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"End time: {end_time}")