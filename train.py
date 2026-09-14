import os
import yaml
import random
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from easydict import EasyDict as edict
from torch.amp import autocast, GradScaler

import model as gaze_model
from reader.reader_adap import loader as data_loader
from ctools import TimeCounter, GetLR
from model import Gelossop, Delossop


def load_config(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        return edict(yaml.safe_load(f))


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(cfg, test_subject=None):
    if test_subject is None:
        raise ValueError(
            "For strict LOPO training, TEST_SUBJECT must be p00~p14. "
            "Use None only for an explicitly non-LOPO experiment."
        )

    set_seed()
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{cfg.device}" if use_cuda else "cpu")
    print(f"==> device: {device}")
    print(f"==> LOPO held-out subject: {test_subject}")

    if use_cuda:
        torch.cuda.set_device(cfg.device)

    save_dir = os.path.join(cfg.save.metapath, cfg.save.folder, "checkpoint")
    os.makedirs(save_dir, exist_ok=True)

    train_loader = data_loader(
        source=cfg.data,
        batch_size=cfg.params.batch_size,
        shuffle=True,
        num_workers=4 if use_cuda else 0,
        is_sampler=False,
        persistent_workers=False,
        prefetch_factor=2,
        pin_memory=use_cuda,
        test_subject=test_subject,
        mode="train",
        drop_last=True,
    )

    # 严格检查：训练集绝不能包含 held-out subject
    train_subjects = set(train_loader.dataset.subject_indices.keys())
    if test_subject in train_subjects:
        raise RuntimeError(
            f"LOPO training split error: {test_subject} is present in training subjects."
        )

    print(
        f"==> train samples: {len(train_loader.dataset)}, "
        f"steps/epoch: {len(train_loader)}"
    )
    print(f"==> train subjects: {sorted(train_subjects)}")

    net = gaze_model.Model().to(device)

    gaze_criterion = Gelossop(w1=1.0, w2=1.0, w3=1.0).to(device)
    de_criterion = Delossop(w_ssim=0.8, w_l1=0.2).to(device)

    optimizer = optim.Adam(
        net.parameters(),
        lr=cfg.params.lr,
        weight_decay=1e-4,
    )
    scheduler = StepLR(
        optimizer,
        step_size=cfg.params.decay_step,
        gamma=cfg.params.decay,
    )

    scaler = GradScaler("cuda", enabled=use_cuda)

    for epoch in range(1, cfg.params.epoch + 1):
        train_loader.dataset.reset_paired_indices()
        net.train()
        timer = TimeCounter(len(train_loader))
        epoch_loss = 0.0

        for step, (face_img, paired_face_img, gaze_label, paired_gaze_label) in enumerate(train_loader, 1):
            face_img = face_img.to(device, non_blocking=use_cuda)
            paired_face_img = paired_face_img.to(device, non_blocking=use_cuda)
            gaze_label = gaze_label.to(device, non_blocking=use_cuda)
            paired_gaze_label = paired_gaze_label.to(device, non_blocking=use_cuda)

            x_in = {
                "face": face_img,
                "pairedface": paired_face_img,
            }

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=use_cuda):
                gaze_pred, paired_gaze_pred, delta_gaze_pred, img1, img2, feat_loss = net(
                    x_in,
                    require_img=True,
                )

                loss_gaze = gaze_criterion(
                    gaze_pred,
                    paired_gaze_pred,
                    delta_gaze_pred,
                    gaze_label,
                    paired_gaze_label,
                )

                loss_de = de_criterion(
                    img1,
                    img2,
                    face_img,
                    paired_face_img,
                )
                total_loss = 10.0 * loss_gaze + loss_de + feat_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += total_loss.item()

            if step % 20 == 0 or step == len(train_loader):
                eta = timer.step()
                current_lr = GetLR(optimizer)
                print(
                    f"Epoch [{epoch}/{cfg.params.epoch}] "
                    f"Step [{step}/{len(train_loader)}] "
                    f"G:{loss_gaze.item():.4f} "
                    f"D:{loss_de.item():.4f} "
                    f"F:{feat_loss.item():.4f} "
                    f"Total:{total_loss.item():.4f} "
                    f"LR:{current_lr:.6f} "

                )

        scheduler.step()

        name = f"{cfg.save.name}_{test_subject}"
        if epoch % cfg.save.step == 0 or epoch == cfg.params.epoch:
            save_path = os.path.join(save_dir, f"Iter_{epoch}_{name}.pt")
            torch.save(net.state_dict(), save_path)
            print(f"saved: {save_path}")

        print(f"Epoch {epoch} avg loss: {epoch_loss / len(train_loader):.6f}")


if __name__ == "__main__":
    config_path = "config/train/config_mpii.yaml"
    cfg = load_config(config_path)

    # PyCharm 直接修改：p00 ~ p14
    TEST_SUBJECT = "p00"
    train(cfg, test_subject=TEST_SUBJECT)
