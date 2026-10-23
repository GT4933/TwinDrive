import torch


class BinarySegMetric:
    """
    Accumulates a 2x2 confusion matrix over a whole dataset for one binary task.
    Rows = ground truth (0 = background, 1 = foreground), columns = prediction.
    Metric definitions match base_paper/iou_eval.py SegmentationMetric, but counts are summed
    over every image before dividing (dataset-level), so empty images cannot fake a perfect score.
    """
    def __init__(self):
        self.cm = torch.zeros(2, 2, dtype=torch.long)

    def update(self, pred, target):
        # pred, target: tensors of 0/1 (any shape, same shape)
        idx = target.long().flatten() * 2 + pred.long().flatten()
        self.cm += torch.bincount(idx, minlength=4).reshape(2, 2).cpu()

    def _iou_per_class(self):
        cm = self.cm.double()
        inter = cm.diag()
        union = cm.sum(0) + cm.sum(1) - inter
        return inter / union.clamp(min=1)

    def iou(self):
        """Foreground IoU = TP / (TP + FP + FN)."""
        return self._iou_per_class()[1].item()

    def miou(self):
        """Mean of background and foreground IoU (how drivable area is usually reported)."""
        return self._iou_per_class().mean().item()

    def recall(self):
        """TP / (TP + FN). Base paper calls this lane 'accuracy' (lineAccuracy)."""
        tp, fn = self.cm[1, 1].item(), self.cm[1, 0].item()
        return tp / max(tp + fn, 1)

    def pixel_accuracy(self):
        return (self.cm.diag().sum() / self.cm.sum().clamp(min=1)).item()

    def num_pixels(self):
        return self.cm.sum().item()
