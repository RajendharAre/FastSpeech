import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# MODIFIED(both): read the single per-language config.yaml (preprocess/model/train/inference).
from utils.config import load_config, split_config
from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample
from utils.train_utils import (
    append_loss_csv,
    checkpoint_path,
    find_latest_checkpoint,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
    set_seed,
)
from model import FastSpeech2Loss
from dataset import Dataset

from evaluate import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_restore_step(args, train_config):
    """Turn ``--resume`` into a concrete step; return ``--restore_step`` otherwise."""
    if not getattr(args, "resume", False):
        return args.restore_step
    latest = find_latest_checkpoint(train_config["path"]["ckpt_path"])
    if latest is None:
        raise SystemExit(
            "--resume found no <step>.pth.tar checkpoint in {}".format(
                train_config["path"]["ckpt_path"]
            )
        )
    print("Resuming from checkpoint step {}".format(latest))
    return latest


def main(args, configs):
    print("Prepare training ...")

    preprocess_config, model_config, train_config = configs

    set_seed(train_config["seed"])
    args.restore_step = resolve_restore_step(args, train_config)
    total_step = train_config["step"]["total_step"]
    if args.restore_step >= total_step:
        raise SystemExit(
            "restore step {} is not below total_step {}".format(
                args.restore_step, total_step
            )
        )
    if args.restore_step:
        restored = load_checkpoint(
            checkpoint_path(train_config["path"]["ckpt_path"], args.restore_step),
            device,
        )
        restore_rng_state(restored.get("rng_state"))

    # Get dataset
    dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = train_config["optimizer"]["group_size"]
    # MODIFIED(both): the reference asserted batch_size * group_size < len(dataset),
    # which made a 50-utterance overfit test impossible. The collate function
    # still needs at least one full microbatch per DataLoader batch.
    assert batch_size <= len(dataset), (
        "batch_size ({}) must not exceed the training set size ({}); lower "
        "train.optimizer.batch_size for the overfit test".format(
            batch_size, len(dataset)
        )
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    # Prepare model
    model, optimizer = get_model(args, configs, device, train=True)
    model = nn.DataParallel(model)
    num_param = get_param_num(model)
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    print("Number of FastSpeech2 Parameters:", num_param)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)
    train_csv = os.path.join(train_log_path, "train_loss.csv")
    val_csv = os.path.join(val_log_path, "val_loss.csv")

    def save(step_to_save):
        path = checkpoint_path(train_config["path"]["ckpt_path"], step_to_save)
        save_checkpoint(
            path,
            model.module.state_dict(),
            optimizer._optimizer.state_dict(),
            step_to_save,
        )
        print("Saved checkpoint: {}".format(path))

    # Training
    step = args.restore_step + 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)

                # Forward
                output = model(*(batch[2:]))

                # Cal Loss
                losses = Loss(batch, output)
                total_loss = losses[0]

                # Backward
                total_loss = total_loss / grad_acc_step
                total_loss.backward()
                if step % grad_acc_step == 0:
                    # Clipping gradients to avoid gradient explosion
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                    # Update weights
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                if step % log_step == 0:
                    loss_values = [l.item() for l in losses]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
                        *loss_values
                    )

                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")

                    append_loss_csv(train_csv, step, loss_values)
                    outer_bar.write(message1 + message2)

                    log(train_logger, step, losses=loss_values)

                if step % synth_step == 0:
                    fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                        batch,
                        output,
                        vocoder,
                        model_config,
                        preprocess_config,
                    )
                    log(
                        train_logger,
                        fig=fig,
                        tag="Training/step_{}_{}".format(step, tag),
                    )
                    sampling_rate = preprocess_config["preprocessing"]["audio"][
                        "sampling_rate"
                    ]
                    log(
                        train_logger,
                        audio=wav_reconstruction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_reconstructed".format(step, tag),
                    )
                    log(
                        train_logger,
                        audio=wav_prediction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_synthesized".format(step, tag),
                    )

                if step % val_step == 0:
                    model.eval()
                    message, val_losses = evaluate(
                        model, step, configs, val_logger, vocoder
                    )
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    append_loss_csv(val_csv, step, val_losses)
                    outer_bar.write(message)

                    model.train()

                if step % save_step == 0:
                    save(step)

                if step == total_step:
                    # MODIFIED(both): always leave a final checkpoint, even if
                    # total_step is not a multiple of save_step.
                    if step % save_step != 0:
                        save(step)
                    outer_bar.close()
                    inner_bar.close()
                    return
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    restore = parser.add_mutually_exclusive_group()
    restore.add_argument("--restore_step", type=int, default=0)
    restore.add_argument(
        "--resume",
        action="store_true",
        help="resume from the highest numbered checkpoint in train.path.ckpt_path",
    )
    # MODIFIED(both): one config.yaml replaces the reference's three YAML arguments.
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.yaml",
        help="path to the language config.yaml",
    )
    args = parser.parse_args()

    # Read Config
    configs = split_config(load_config(args.config))

    main(args, configs)
