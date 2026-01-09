import argparse
import os

import lanms
import numpy as np
import torch

from src.models import EAST
from src.utils.misc import get_rotate_mat
from PIL import Image, ImageDraw
from torchvision import transforms


# -------------------------
# Utility functions
# -------------------------

def resize(image):
    """Resize image to be divisible by 32"""
    old_w, old_h = image.size
    new_h = old_h if old_h % 32 == 0 else (old_h // 32) * 32
    new_w = old_w if old_w % 32 == 0 else (old_w // 32) * 32

    image = image.resize((new_w, new_h), Image.BILINEAR)
    ratio_h = new_h / old_h
    ratio_w = new_w / old_w

    return image, ratio_h, ratio_w


def is_valid_poly(res, score_shape, scale):
    cnt = 0
    for i in range(res.shape[1]):
        if (
            res[0, i] < 0
            or res[0, i] >= score_shape[1] * scale
            or res[1, i] < 0
            or res[1, i] >= score_shape[0] * scale
        ):
            cnt += 1
    return cnt <= 1


def restore_polys(valid_pos, valid_geo, score_shape, scale=4):
    polys = []
    index = []

    valid_pos *= scale
    d = valid_geo[:4, :]
    angle = valid_geo[4, :]

    for i in range(valid_pos.shape[0]):
        x, y = valid_pos[i, 0], valid_pos[i, 1]

        y_min, y_max = y - d[0, i], y + d[1, i]
        x_min, x_max = x - d[2, i], x + d[3, i]

        rotate_mat = get_rotate_mat(-angle[i])

        temp_x = np.array([[x_min, x_max, x_max, x_min]]) - x
        temp_y = np.array([[y_min, y_min, y_max, y_max]]) - y
        coordinates = np.concatenate((temp_x, temp_y), axis=0)

        res = np.dot(rotate_mat, coordinates)
        res[0, :] += x
        res[1, :] += y

        if is_valid_poly(res, score_shape, scale):
            index.append(i)
            polys.append(
                [res[0, 0], res[1, 0],
                 res[0, 1], res[1, 1],
                 res[0, 2], res[1, 2],
                 res[0, 3], res[1, 3]]
            )

    return np.array(polys), index


def get_boxes(confidence, geometries, confidence_thresh=0.9, nms_thresh=0.2):
    confidence = confidence[0, :, :]
    xy_text = np.argwhere(confidence > confidence_thresh)

    if xy_text.size == 0:
        return None

    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    valid_pos = xy_text[:, ::-1].copy()
    valid_geo = geometries[:, xy_text[:, 0], xy_text[:, 1]]

    polys, index = restore_polys(valid_pos, valid_geo, confidence.shape)
    if polys.size == 0:
        return None

    boxes = np.zeros((polys.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = polys
    boxes[:, 8] = confidence[xy_text[index, 0], xy_text[index, 1]]

    boxes = lanms.merge_quadrangle_n9(boxes.astype("float32"), nms_thresh)
    return boxes


def adjust_ratio(boxes, ratio_w, ratio_h):
    if boxes is None or boxes.size == 0:
        return None

    boxes[:, [0, 2, 4, 6]] /= ratio_w
    boxes[:, [1, 3, 5, 7]] /= ratio_h
    return np.around(boxes)


# -------------------------
# ICDAR helper (IMPORTANT)
# -------------------------

def order_points_clockwise(pts):
    """
    pts: numpy array (4,2)
    returns points ordered clockwise:
    top-left, top-right, bottom-right, bottom-left
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]       # top-left
    rect[2] = pts[np.argmax(s)]       # bottom-right
    rect[1] = pts[np.argmin(diff)]    # top-right
    rect[3] = pts[np.argmax(diff)]    # bottom-left

    return rect


# -------------------------
# Detection logic
# -------------------------

def detect(image, model, device):
    model.eval()
    image, ratio_h, ratio_w = resize(image)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        confidence, geometries = model(image)

    confidence = confidence.squeeze(0).cpu().numpy()
    geometries = geometries.squeeze(0).cpu().numpy()

    boxes = get_boxes(confidence, geometries)
    return adjust_ratio(boxes, ratio_w, ratio_h)


def plot_boxes(image, boxes):
    if boxes is None:
        return image

    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.polygon(
            [box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]],
            outline=(0, 255, 0),
        )
    return image


# -------------------------
# Argument parser
# -------------------------

def parse_opt():
    parser = argparse.ArgumentParser("EAST inference")
    parser.add_argument("--cfg", default="D", help="Backbone config [A, B, D, E]")
    parser.add_argument("--weights", required=True, help="Path to model state_dict (.pth)")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", required=True, help="Output image OR output directory")
    return parser.parse_args()


# -------------------------
# Main
# -------------------------

if __name__ == "__main__":
    opt = parse_opt()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = EAST(cfg=opt.cfg).to(device)
    state_dict = torch.load(opt.weights, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Load image
    image = Image.open(opt.input).convert("RGB")

    # Detect
    boxes = detect(image, model, device)
    plot_img = plot_boxes(image, boxes)

    # Prepare output paths
    if os.path.isdir(opt.output):
        os.makedirs(opt.output, exist_ok=True)
        img_name = os.path.basename(opt.input)
        save_img_path = os.path.join(opt.output, img_name)
        save_txt_path = os.path.join(
            opt.output, "res_" + os.path.splitext(img_name)[0] + ".txt"
        )
    else:
        save_img_path = opt.output
        save_txt_path = os.path.splitext(opt.output)[0] + ".txt"

    # Save image
    plot_img.save(save_img_path)

    # Save ICDAR-format txt (CLOCKWISE FIX APPLIED)
    if boxes is not None:
        with open(save_txt_path, "w") as f:
            for box in boxes:
                pts = np.array([
                    [box[0], box[1]],
                    [box[2], box[3]],
                    [box[4], box[5]],
                    [box[6], box[7]],
                ])

                pts = order_points_clockwise(pts)
                line = ",".join(str(int(v)) for v in pts.flatten())
                f.write(line + "\n")

    print(f"[OK] Saved image : {save_img_path}")
    print(f"[OK] Saved boxes : {save_txt_path}")
