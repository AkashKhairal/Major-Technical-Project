# %% [markdown]
# # EAST Scene Text Detection (Scratch Training) – ICDAR2015
# 
# **Objective**  
# To build an end-to-end scene text detection pipeline using EAST, train it from scratch,
# and establish a baseline for further lightweight detector research.
# 
# **Key Focus**
# - End-to-end pipeline correctness
# - Proper evaluation using ICDAR2015 protocol
# - Recording accuracy + efficiency metrics
# 

# %% [markdown]
# ## Project setup

# %%
import os
import sys
import torch
import numpy as np
import random
import time
from torch.cuda.amp import autocast, GradScaler

PROJECT_ROOT = "/DATA/akash/akash_cnn/lightweight-text-detector"
os.chdir(PROJECT_ROOT)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("CWD:", os.getcwd())


# %% [markdown]
# ## Experiment CONFIG

# %%
# ===============================
# EXPERIMENT CONTROL
# ===============================

EXPERIMENT_NAME = "exp3.1_imagenet_vgg16_long_600eph"

USE_PRETRAINED = True
PRETRAINED_TYPE = "imagenet"

INPUT_SIZE = 512
EPOCHS = 600
BATCH_SIZE = 20
LEARNING_RATE = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# %% [markdown]
# ## Experiment Folder Setup

# %%
import os

EXP_ROOT = f"experiments/{EXPERIMENT_NAME}"

WEIGHTS_DIR = f"{EXP_ROOT}/weights"
LOG_DIR = f"{EXP_ROOT}/logs"
RESULTS_DIR = f"{EXP_ROOT}/results"
PRED_DIR = f"{RESULTS_DIR}/predictions"

for d in [WEIGHTS_DIR, LOG_DIR, PRED_DIR]:
    os.makedirs(d, exist_ok=True)

print("Experiment directories ready")


# %% [markdown]
# ## Dataset Paths

# %%
TRAIN_IMG_DIR = "data/icdar2015/ch4_train_images"
TEST_IMG_DIR  = "data/icdar2015/ch4_test_images"


# %% [markdown]
# ## Model Initialization

# %%
from src.models.east import EAST
import torch

model = EAST(
    cfg="D",
    weights="imagenet" if USE_PRETRAINED else None
)

model = torch.nn.DataParallel(model)   # 🔥 MAIN FIX
model = model.to(DEVICE)

print("EAST initialized with DataParallel")


# %% [markdown]
# ## Loss & Optimizer

# %%
from src.losses.loss import Loss

criterion = Loss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# %% [markdown]
# ## DATASET + DATALOADER CELL

# %%
from torch.utils.data import DataLoader
from src.data.dataset import Dataset

# -------------------------
# Dataset
# -------------------------
train_dataset = Dataset(
    data_path=TRAIN_IMG_DIR,
    scale=0.25,
    length=INPUT_SIZE
)

# -------------------------
# DataLoader
# -------------------------
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    drop_last=True
)

print("Train loader ready")
print("Total training samples:", len(train_dataset))
print("Batches per epoch:", len(train_loader))


# %%
from torch.cuda.amp import GradScaler

scaler = GradScaler()
print("AMP GradScaler initialized")


# %%
start_epoch = 0
print("Starting fresh training from epoch 0")


# %% [markdown]
# ## Training Loop

# %%
from tqdm import tqdm
import json   # 🔥 ADD (if not already imported)

model.train()
loss_log = []

for epoch in range(start_epoch, EPOCHS):

    epoch_geo_loss = 0.0
    epoch_cls_loss = 0.0

    pbar = tqdm(
        train_loader,
        desc=f"Epoch [{epoch+1}/{EPOCHS}]",
        dynamic_ncols=True
    )

    for imgs, gt_score, gt_geo, ignored_map in pbar:

        # -------------------------
        # Move to device
        # -------------------------
        imgs = imgs.to(DEVICE, non_blocking=True)
        gt_score = gt_score.to(DEVICE, non_blocking=True)
        gt_geo = gt_geo.to(DEVICE, non_blocking=True)
        ignored_map = ignored_map.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # -------------------------
        # Forward + EAST loss (AMP)
        # -------------------------
        with autocast():
            pred_score, pred_geo = model(imgs)

            # 🔥 geometry loss FP32
            loss_dict = criterion(
                gt_score.float(), pred_score.float(),
                gt_geo.float(), pred_geo.float(),
                ignored_map.float()
            )

            geo_loss = loss_dict["geo_loss"]
            cls_loss = loss_dict["cls_loss"]
            total_loss = geo_loss + cls_loss

        # -------------------------
        # Backward
        # -------------------------
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # -------------------------
        # Logging
        # -------------------------
        epoch_geo_loss += geo_loss.item()
        epoch_cls_loss += cls_loss.item()

        pbar.set_postfix(
            geo=f"{geo_loss.item():.3f}",
            cls=f"{cls_loss.item():.3f}"
        )

    # -------------------------
    # Epoch summary
    # -------------------------
    avg_geo = epoch_geo_loss / len(train_loader)
    avg_cls = epoch_cls_loss / len(train_loader)

    loss_log.append({
        "epoch": epoch + 1,
        "geo_loss": avg_geo,
        "cls_loss": avg_cls
    })

    # 🔥 NEW: SAVE LOSS LOG EVERY EPOCH (CRASH-SAFE)
    with open(f"{LOG_DIR}/loss_log.json", "w") as f:
        json.dump(loss_log, f, indent=2)

    print(
        f"\nEpoch {epoch+1} Summary | "
        f"Geo Loss: {avg_geo:.4f} | "
        f"Cls Loss: {avg_cls:.4f}"
    )

    # -------------------------
    # Save checkpoint (FULL STATE)
    # -------------------------
    if (epoch + 1) % 10 == 0:
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
            },
            f"{WEIGHTS_DIR}/epoch_{epoch+1}.pth"
        )

print("\nTraining finished successfully")


# %%


# %%


# %%


# %%


# %%


# %%



