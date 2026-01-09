import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, inputs, targets):
        intersection = torch.sum(inputs * targets)
        union = torch.sum(inputs) + torch.sum(targets)
        dice = (2.0 * intersection + self.eps) / (union + self.eps)
        return 1 - dice


class GeoLoss:
    def __call__(self, gt_geo, pred_geo):
        d1_gt, d2_gt, d3_gt, d4_gt, angle_gt = torch.split(gt_geo, 1, 1)
        d1_pr, d2_pr, d3_pr, d4_pr, angle_pr = torch.split(pred_geo, 1, 1)

        area_gt = (d1_gt + d2_gt) * (d3_gt + d4_gt)
        area_pr = (d1_pr + d2_pr) * (d3_pr + d4_pr)

        w_int = torch.min(d3_gt, d3_pr) + torch.min(d4_gt, d4_pr)
        h_int = torch.min(d1_gt, d1_pr) + torch.min(d2_gt, d2_pr)

        area_int = w_int * h_int
        area_union = area_gt + area_pr - area_int

        iou_loss_map = -torch.log((area_int + 1.0) / (area_union + 1.0))
        angle_loss_map = 1 - torch.cos(angle_pr - angle_gt)

        return iou_loss_map, angle_loss_map


class Loss(nn.Module):
    def __init__(self, weight_angle=10):
        super().__init__()
        self.weight_angle = weight_angle
        self.dice_loss = DiceLoss(eps=1e-5)
        self.geo_loss = GeoLoss()

    def forward(self, gt_score, pred_score, gt_geo, pred_geo, ignored_map):
        # -----------------------------
        # No positive samples case
        # -----------------------------
        if torch.sum(gt_score) < 1:
            zero = torch.sum(pred_score + pred_geo) * 0
            return {
                "geo_loss": zero,
                "cls_loss": zero
            }

        # -----------------------------
        # Classification loss
        # -----------------------------
        cls_loss = self.dice_loss(
            pred_score * (1 - ignored_map),
            gt_score
        )

        # -----------------------------
        # Geometry loss
        # -----------------------------
        iou_loss_map, angle_loss_map = self.geo_loss(gt_geo, pred_geo)

        angle_loss = torch.sum(angle_loss_map * gt_score) / torch.sum(gt_score)
        iou_loss = torch.sum(iou_loss_map * gt_score) / torch.sum(gt_score)

        geo_loss = self.weight_angle * angle_loss + iou_loss

        return {
            "geo_loss": geo_loss,
            "cls_loss": cls_loss
        }
