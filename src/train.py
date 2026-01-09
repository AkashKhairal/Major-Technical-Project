import argparse
import logging
import os
import warnings
from copy import deepcopy

import torch
from torch.optim import lr_scheduler
from tqdm import tqdm

from src.models import EAST
from src.data.dataset import create_dataloader
from src.losses.loss import Loss
from src.utils.misc import strip_optimizer

warnings.simplefilter("ignore")


def train(opt, model, device):
    start_epoch = 0
    epoch_geo_losses = []
    epoch_cls_losses = []

    os.makedirs(opt.save_dir, exist_ok=True)

    # ---- Resume logic (SAFE) ----
    pretrained = opt.resume
    if pretrained:
        assert os.path.exists(opt.checkpoint), "Checkpoint not found!"
        ckpt = torch.load(opt.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"].float().state_dict())
    else:
        ckpt = None

    logging.info("Creating Dataloader")
    train_loader = create_dataloader(
        opt.data_path,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers
    )

    criterion = Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
    scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=[opt.epochs // 2], gamma=0.1
    )

    # ---- Resume optimizer & epoch ----
    if pretrained and ckpt is not None:
        if ckpt.get("optimizer") is not None:
            start_epoch = ckpt["epoch"] + 1
            optimizer.load_state_dict(ckpt["optimizer"])
            logging.info(f"Optimizer loaded from {opt.checkpoint}")
        del ckpt

    # ---- Training loop ----
    for epoch in range(start_epoch, opt.epochs):
        model.train()

        epoch_geo_loss = 0.0
        epoch_cls_loss = 0.0

        logging.info(("\n" + "%12s" * 4) % ("Epoch", "GPU Mem", "Geo Loss", "Cls Loss"))
        progress_bar = tqdm(train_loader, total=len(train_loader))

        for image, gt_score, gt_geo, ignored_map in progress_bar:
            image = image.to(device)
            gt_score = gt_score.to(device)
            gt_geo = gt_geo.to(device)
            ignored_map = ignored_map.to(device)

            pred_score, pred_geo = model(image)
            loss_out = criterion(
            gt_score, pred_score, gt_geo, pred_geo, ignored_map
            )

            # handle both dict and scalar loss
            if isinstance(loss_out, dict):
                geo_l = loss_out["geo_loss"]
                cls_l = loss_out["cls_loss"]
                loss = geo_l + cls_l
            else:
                geo_l = loss_out
                cls_l = torch.tensor(0.0, device=device)
                loss = loss_out


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # accumulate epoch loss
            epoch_geo_loss += geo_l.item()
            epoch_cls_loss += cls_l.item()


            mem = (
                f"{torch.cuda.memory_reserved() / 1E9:.3g}G"
                if torch.cuda.is_available()
                else "0G"
            )

            progress_bar.set_description(
                ("%12s" * 2 + "%12.4g" * 2)
                % (
                    f"{epoch + 1}/{opt.epochs}",
                    mem,
                    geo_l,
                    cls_l
                )
            )

        # epoch average
        epoch_geo_losses.append(epoch_geo_loss / len(train_loader))
        epoch_cls_losses.append(epoch_cls_loss / len(train_loader))

        scheduler.step()

        # ---- Save checkpoint ----
        ckpt = {
            "epoch": epoch,
            "model": deepcopy(model).half(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(ckpt, f"{opt.save_dir}/model.ckpt")

    #strip_optimizer(f"{opt.save_dir}/model.ckpt")

    return epoch_geo_losses, epoch_cls_losses


def main(opt):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info("train: " + ", ".join(f"{k}={v}" for k, v in vars(opt).items()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    # ---- SCRATCH MODEL ----
    model = EAST(cfg=opt.cfg, weights=None)
    model.to(device)

    train(opt, model, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EAST: An Efficient and Accurate Scene Text Detector"
    )

    parser.add_argument("--cfg", default="D", type=str, help="VGG backbone config")
    parser.add_argument(
        "--data-path", default="data/ch4_train_images", help="Path to training images"
    )
    parser.add_argument(
        "--checkpoint", default="./weights/model.ckpt", help="Checkpoint path"
    )
    parser.add_argument("--save-dir", default="./weights", help="Save directory")
    parser.add_argument("--batch-size", default=20, type=int)
    parser.add_argument("--learning-rate", default=1e-3, type=float)
    parser.add_argument("--num-workers", default=8, type=int)
    parser.add_argument("--epochs", default=600, type=int)
    parser.add_argument("--resume", action="store_true", help="Resume training")

    args = parser.parse_args()
    main(args)
