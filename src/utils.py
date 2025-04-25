import numpy as np
import torch
from torch import nn
import evaluate

metric = evaluate.load("mean_iou")

def add_batch(eval_pred):
    # with torch.no_grad():
    logits, labels = eval_pred
    labels_tensor = torch.from_numpy(labels)
    logits_tensor = torch.from_numpy(logits)
    logits_tensor = nn.functional.interpolate(
        logits_tensor,
        size=labels.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).argmax(dim=1)

    pred = logits_tensor.detach().cpu().numpy()
    labels = labels_tensor.detach().cpu().numpy()
    # metric.add_batch(pred_labels, labels)
    return pred, labels


def compute(all_preds, num_labels:int=2):
    pred, labels = all_preds[0], all_preds[1]
    print(pred, labels)
    metrics = metric.compute(
        predictions=pred.cpu(),
        references=labels.cpu(),
        num_labels=num_labels,
        ignore_index=255,
        reduce_labels=False,
    )
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            metrics[key] = value.tolist()
    return metrics


def apalette():
    return [[0, 0, 0], [255, 255, 255]]