import os
import sys
sys.path.insert(0, os.getcwd())

import math
import numpy as np
import torch
import model
import yaml
from importlib import import_module
from easydict import EasyDict as edict



TEST_SUBJECT = "Subject01"


def gazeto3d(gaze):
    """与训练标签反解严格对应的 2D -> 3D gaze conversion。"""
    gaze = np.asarray(gaze, dtype=np.float64)
    if gaze.size != 2:
        raise ValueError(f"gaze must have 2 values, got shape {gaze.shape}")

    a0, a1 = gaze
    v = np.array([
        -np.cos(a0) * np.sin(a1),
        -np.sin(a0),
        -np.cos(a0) * np.cos(a1),
    ], dtype=np.float64)
    norm = np.linalg.norm(v)
    return v / max(norm, 1e-12)


def angular(v1, v2):
    """返回两条 3D gaze 向量的夹角，单位 degree。"""
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)
    v1 /= max(np.linalg.norm(v1), 1e-12)
    v2 /= max(np.linalg.norm(v2), 1e-12)
    cosang = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def main(train, test, test_subject=None):
    if test_subject is None:
        raise ValueError("TEST_SUBJECT must be subject01~xx for strict LOPO evaluation.")

    reader = import_module(f"reader.{test.reader}")

    if torch.cuda.is_available():
        torch.cuda.set_device(test.device)
        device = torch.device(f"cuda:{test.device}")
    else:
        device = torch.device("cpu")

    data = test.data
    load = test.load

    modelpath = os.path.join(
        train.save.metapath,
        train.save.folder,
        "checkpoint",
    )

    subject_name = test_subject
    logpath = os.path.join(
        train.save.metapath,
        train.save.folder,
        f"{test.savename}_{subject_name}",
    )
    os.makedirs(logpath, exist_ok=True)

    dataset = reader.loader(
        data,
        batch_size=16,
        num_workers=0,
        shuffle=False,
        is_sampler=False,
        persistent_workers=False,
        prefetch_factor=2,
        pin_memory=torch.cuda.is_available(),
        test_subject=test_subject,
        mode="test",
        drop_last=False,
    )

    # 严格 LOPO 检查
    test_subjects = set(dataset.dataset.subject_indices.keys())
    if test_subjects != {test_subject}:
        raise RuntimeError(
            f"LOPO test split error: expected {{{test_subject}}}, "
            f"got {sorted(test_subjects)}"
        )

    print(
        f"==> LOPO test subject: {test_subject}, "
        f"samples: {len(dataset.dataset)}"
    )

    begin = int(load.begin)
    requested_end = int(load.end)
    # 当前训练配置为 epoch=30；避免测试误加载超出本次训练范围的旧 checkpoint。
    end = min(requested_end, int(train.params.epoch))
    step = int(load.steps)
    if requested_end != end:
        print(f"==> test end clipped from {requested_end} to {end} to match train epoch={train.params.epoch}")

    for saveiter in range(begin, end + step, step):
        print(f"Test {subject_name}: Iter {saveiter}")

        net = model.Model().to(device)
        model_file = os.path.join(
            modelpath,
            f"Iter_{saveiter}_{train.save.name}_{subject_name}.pt",
        )

        if not os.path.exists(model_file):
            raise FileNotFoundError(
                f"Missing LOPO checkpoint for {test_subject}:\n{model_file}"
            )

        state = torch.load(model_file, map_location=device)
        net.load_state_dict(state)
        net.eval()

        logname = os.path.join(logpath, f"{saveiter}.log")
        accs = 0.0
        count = 0

        with open(logname, "w", encoding="utf-8") as outfile:
            outfile.write("name results gts error_deg\n")

            with torch.no_grad():
                for j, (face_img, _, gaze_gt, _) in enumerate(dataset):
                    face_img = face_img.to(device, non_blocking=torch.cuda.is_available())
                    gaze_gt = gaze_gt.to(device, non_blocking=torch.cuda.is_available())

                    # 主 gaze 输出只依赖 x_in["face"]；保持接口完整即可。
                    # pairedface 这里使用自身图像，不参与主 gaze 的计算。
                    input_dict = {
                        "face": face_img,
                        "pairedface": face_img,
                    }

                    gaze, _, _, _, _, _ = net(
                        input_dict,
                        require_img=False,
                    )

                    pred_batch = gaze.detach().cpu().numpy()
                    gt_batch = gaze_gt.detach().cpu().numpy()

                    for k, (pred, gt) in enumerate(zip(pred_batch, gt_batch)):
                        err = angular(gazeto3d(gt), gazeto3d(pred))
                        accs += err
                        count += 1

                        sample_id = j * pred_batch.shape[0] + k
                        outfile.write(
                            f"sample_{sample_id} "
                            f"{pred[0]:.9f} {pred[1]:.9f} "
                            f"{gt[0]:.9f} {gt[1]:.9f} "
                            f"{err:.9f}\n"
                        )

            avg = accs / max(count, 1)
            outfile.write(f"[{saveiter}] Total Num: {count}, avg:{avg:.9f}\n")

        print(f"[{subject_name}] Iter {saveiter} avg error: {avg:.6f} deg")


if __name__ == "__main__":
    train_path = "config/train/config_lbw.yaml"
    test_path = "config/test/config_lbw.yaml"

    with open(train_path, "r", encoding="utf-8") as f:
        train_conf = edict(yaml.safe_load(f))
    with open(test_path, "r", encoding="utf-8") as f:
        test_conf = edict(yaml.safe_load(f))

    main(train_conf, test_conf, TEST_SUBJECT)
