import argparse
import os
import pickle
import shutil

import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from data_loader import get_dataloaders
from diffusion.resample import create_named_schedule_sampler
from models import FaceDiff, FaceDiffBeat, FaceDiffDamm, FaceDiffMeadARKit
from utils import plot_losses, create_gaussian_diffusion

def velocity_loss(x_pred, x_gt, reduction='mean'):
    vel_pred = x_pred[:, 1:, :] - x_pred[:, :-1, :]
    vel_gt = x_gt[:, 1:, :] - x_gt[:, :-1, :]
    return F.mse_loss(vel_pred, vel_gt, reduction=reduction)

def acceleration_loss(x_pred, x_gt, reduction='mean'):
    acc_pred = x_pred[:, 2:, :] - 2 * x_pred[:, 1:-1, :] + x_pred[:, :-2, :]
    acc_gt = x_gt[:, 2:, :] - 2 * x_gt[:, 1:-1, :] + x_gt[:, :-2, :]
    return F.mse_loss(acc_pred, acc_gt, reduction=reduction)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def trainer_diff(args, train_loader, dev_loader, model, diffusion, optimizer, start_epoch=0, device="cuda"):
    train_losses = []
    val_losses = []

    save_path = os.path.join(args.save_path)
    os.makedirs(save_path, exist_ok=True)
    schedule_sampler = create_named_schedule_sampler('uniform', diffusion)
    train_subjects_list = [i for i in args.train_subjects.split(" ")]

    iteration = 0

    # Check if previous loss logs exist
    train_losses_file = os.path.join(save_path, f"{args.model}_{args.dataset}_train_losses.npy")
    val_losses_file = os.path.join(save_path, f"{args.model}_{args.dataset}_val_losses.npy")
    if os.path.exists(train_losses_file):
        train_losses = list(np.load(train_losses_file))
    if os.path.exists(val_losses_file):
        val_losses = list(np.load(val_losses_file))

    for e in range(start_epoch + 1, args.max_epoch + 1):
        loss_log = []
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        optimizer.zero_grad()

        for i, (audio, vertice, template, one_hot, file_name) in pbar:
            iteration += 1
            vertice = torch.from_numpy(np.load(str(vertice[0]), allow_pickle=True).astype(np.float32)).unsqueeze(0).to(device)
            audio, template, one_hot = audio.to(device), template.to(device), one_hot.to(device)

            if args.dataset == 'vocaset':
                vertice = vertice[:, ::2, :]

            t, _ = schedule_sampler.sample(1, device)
            result = diffusion.training_losses(
                model,
                x_start=vertice,
                t=t,
                model_kwargs={
                    "cond_embed": audio,
                    "one_hot": one_hot,
                    "template": template,
                }
            )
            loss = torch.mean(result['loss'])
            loss.backward()
            loss_log.append(loss.item())

            if i % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

            pbar.set_description(
                "(Epoch {}, iteration {}) TRAIN LOSS:{:.8f}".format((e + 1), iteration, np.mean(loss_log))
            )

        train_losses.append(np.mean(loss_log))

        # Validation
        valid_loss_log = []
        model.eval()
        with torch.no_grad():
            for audio, vertice, template, one_hot_all, file_name in dev_loader:
                vertice = torch.from_numpy(np.load(str(vertice[0]), allow_pickle=True).astype(np.float32)).unsqueeze(0).to(device)
                audio, template, one_hot_all = audio.to(device), template.to(device), one_hot_all.to(device)

                if args.dataset == 'vocaset':
                    vertice = vertice[:, ::2, :]

                t, _ = schedule_sampler.sample(1, device)
                train_subject = file_name[0].split("_")[0]

                if train_subject in train_subjects_list:
                    iter_idx = train_subjects_list.index(train_subject)
                    one_hot = one_hot_all[:, iter_idx, :]
                    loss = torch.mean(diffusion.training_losses(
                        model, x_start=vertice, t=t,
                        model_kwargs={"cond_embed": audio, "one_hot": one_hot, "template": template}
                    )['loss'])
                    valid_loss_log.append(loss.item())
                else:
                    for iter_idx in range(one_hot_all.shape[-1]):
                        one_hot = one_hot_all[:, iter_idx, :]
                        loss = torch.mean(diffusion.training_losses(
                            model, x_start=vertice, t=t,
                            model_kwargs={"cond_embed": audio, "one_hot": one_hot, "template": template}
                        )['loss'])
                        valid_loss_log.append(loss.item())

        current_loss = np.mean(valid_loss_log)
        val_losses.append(current_loss)
        print(f"Epoch {e}, Current validation loss: {current_loss:.8f}")

        # Save checkpoint
        checkpoint = {
            "epoch": e,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "args": vars(args)
        }
        torch.save(checkpoint, os.path.join(save_path, f"{args.model}_{args.dataset}_latest.pth"))
        np.save(train_losses_file, np.array(train_losses))
        np.save(val_losses_file, np.array(val_losses))
        plot_losses(train_losses, val_losses, os.path.join(save_path, f"losses_{args.model}_{args.dataset}"))

    return model

@torch.no_grad()
def test_diff(args, model, test_loader, epoch, diffusion, device="cuda"):
    result_path = os.path.join(args.result_path)
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path)

    save_path = os.path.join(args.save_path)
    train_subjects_list = [i for i in args.train_subjects.split(" ")]

    model.load_state_dict(torch.load(os.path.join(save_path, f'{args.model}_{args.dataset}_{epoch}.pth')))
    model = model.to(torch.device(device))
    model.eval()

    sr = 16000
    for audio, vertice, template, one_hot_all, file_name in test_loader:
        vertice = vertice_path = str(vertice[0])
        vertice = np.load(vertice, allow_pickle=True)
        vertice = vertice.astype(np.float32)
        vertice = torch.from_numpy(vertice)

        if args.dataset == 'vocaset':
            vertice = vertice[::2, :]
        vertice = torch.unsqueeze(vertice, 0)


        audio, vertice =  audio.to(device=device), vertice.to(device=device)
        template, one_hot_all = template.to(device=device), one_hot_all.to(device=device)

        num_frames = int(audio.shape[-1] / sr * args.output_fps) # it calculates the number of frames based on audio length and output fps. number of samples (16000) / sampling rate * frames per second
        shape = (1, num_frames - 1, args.vertice_dim) if num_frames < vertice.shape[1] else vertice.shape # if the audio-derived frame count (num_frames) is smaller than vertice.shape[1] (the number of frames in the ground-truth vertice sequence), then shape is set to a 3-tuple: (1 (batch), num_frames - 1, vertices/blendshapes nr)
        # else, shape is set to vertice.shape (the shape of the ground-truth vertice sequence)
        train_subject = file_name[0].split("_")[0]
        vertice_path = os.path.split(vertice_path)[-1][:-4]
        print(vertice_path)

        if train_subject in train_subjects_list or args.dataset == 'beat' or args.dataset == 'mead_arkit':
            condition_subject = train_subject
            iter = train_subjects_list.index(condition_subject)
            one_hot = one_hot_all[:, iter, :]
            one_hot = one_hot.to(device=device)

            for sample_idx in range(1, args.num_samples + 1):
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
                    progress=True,
                    dump_steps=None,
                    noise=None,
                    const_noise=False,
                    device=device
                )
                sample = sample.squeeze()
                sample = sample.detach().cpu().numpy()

                if args.dataset == 'beat' or args.dataset == 'mead_arkit':
                    out_path = f"{vertice_path}.npy"
                else:
                    if args.num_samples != 1:
                        out_path = f"{vertice_path}_condition_{condition_subject}_{sample_idx}.npy"
                    else:
                        out_path = f"{vertice_path}_condition_{condition_subject}.npy"
                if 'damm' in args.dataset:
                    # sample = RIG_SCALER.inverse_transform(sample)
                    np.save(os.path.join(args.result_path, out_path), sample)
                    df = pd.DataFrame(sample)
                    df.to_csv(os.path.join(args.result_path, f"{vertice_path}.csv"), header=None, index=None)
                else:
                    np.save(os.path.join(args.result_path, out_path), sample)

        else:
            for iter in range(one_hot_all.shape[-1]):
                condition_subject = train_subjects_list[iter]
                one_hot = one_hot_all[:, iter, :]
                one_hot = one_hot.to(device=device)

                # sample conditioned
                sample_cond = diffusion.p_sample_loop(
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
                    progress=True,
                    dump_steps=None,
                    noise=None,
                    const_noise=False,
                    device=device
                )
                prediction_cond = sample_cond.squeeze()
                prediction_cond = prediction_cond.detach().cpu().numpy()

                prediction = prediction_cond
                if 'damm' in args.dataset:
                    #prediction = RIG_SCALER.inverse_transform(prediction)
                    df = pd.DataFrame(prediction)
                    df.to_csv(os.path.join(args.result_path, f"{vertice_path}.csv"), header=None, index=None)
                else:
                    np.save(os.path.join(args.result_path, f"{vertice_path}_condition_{condition_subject}.npy"), prediction)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--dataset", type=str, default="mead_arkit")
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--vertice_dim", type=int, default=51)
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--gru_dim", type=int, default=512)
    parser.add_argument("--gru_layers", type=int, default=2)
    parser.add_argument("--wav_path", type=str, default="wav")
    parser.add_argument("--vertices_path", type=str, default="arkit")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="face_diffuser_arkit")
    parser.add_argument("--template_file", type=str, default="templates.pkl")
    parser.add_argument("--save_path", type=str, default="save")
    parser.add_argument("--result_path", type=str, default="result")
    parser.add_argument("--train_subjects", type=str, default="M040 M041")
    parser.add_argument("--val_subjects", type=str, default="M040 M041")
    parser.add_argument("--test_subjects", type=str, default="M040 M041")
    parser.add_argument("--input_fps", type=int, default=50)
    parser.add_argument("--output_fps", type=int, default=30)
    parser.add_argument("--diff_steps", type=int, default=1000)
    parser.add_argument("--skip_steps", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=1)
    args = parser.parse_args()

    assert torch.cuda.is_available()
    cuda = torch.device(args.device)
    diffusion = create_gaussian_diffusion(args)

    # Initialize model
    if 'damm' in args.dataset:
        model = FaceDiffDamm(args)
    elif 'beat' in args.dataset:
        model = FaceDiffBeat(
            args, vertice_dim=args.vertice_dim, latent_dim=args.feature_dim,
            diffusion_steps=args.diff_steps, gru_latent_dim=args.gru_dim, num_layers=args.gru_layers
        )
    elif 'mead_arkit' in args.dataset:
        model = FaceDiffMeadARKit(
            args, vertice_dim=args.vertice_dim, latent_dim=args.feature_dim,
            diffusion_steps=args.diff_steps, gru_latent_dim=args.gru_dim, num_layers=args.gru_layers
        )
    else:
        model = FaceDiff(
            args, vertice_dim=args.vertice_dim, latent_dim=args.feature_dim,
            diffusion_steps=args.diff_steps, gru_latent_dim=args.gru_dim, num_layers=args.gru_layers
        )
    print("Model parameters:", count_parameters(model))
    model = model.to(cuda)

    dataset = get_dataloaders(args)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # Resume training if checkpoint exists
    checkpoint_path = os.path.join(args.save_path, f"{args.model}_{args.dataset}_latest.pth")
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=cuda)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming from epoch {start_epoch}")

    model = trainer_diff(args, dataset["train"], dataset["valid"], model, diffusion, optimizer,
                         start_epoch=start_epoch, device=args.device)

    test_diff(args, model, dataset["test"], args.max_epoch, diffusion, device=args.device)

if __name__ == "__main__":
    main()