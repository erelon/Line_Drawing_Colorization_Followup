from datetime import datetime

import numpy as np
import torch


def gatherClassImbalanceInfo(dataloader, outName="imbalance_vector"):
    lamb = 0.5
    Q = 512

    p = torch.zeros((Q), dtype=torch.float64)

    counter = 0
    for i, (labels, _, _) in enumerate(dataloader):
        for label in labels:
            # print(".", sep="")
            p += label.reshape((-1, 512)).sum(axis=0)
        counter += labels.shape[0]
        print(f"Done {counter} images.")
        if i == 10:
            break

    tmpP = p / counter
    w = 1 / ((1 - lamb) * tmpP + lamb / Q)
    scale = (tmpP * w).sum()
    w /= scale

    w = w.numpy()
    np.save(outName, w.astype(np.float16))


def gatherLiveClassImbalanceInfo(labels_batch):
    lamb = 0.5
    Q = 512

    p = torch.zeros((Q), dtype=torch.float64)

    for label in labels_batch:
        # print(".", sep="")
        p += label.reshape((-1, 512)).sum(axis=0)

    tmpP = p / labels_batch.shape[0]
    w = 1 / ((1 - lamb) * tmpP + lamb / Q)
    scale = (tmpP * w).sum()
    w /= scale

    return w


def createClassMatrix(outName="chroma_loss"):
    labels = np.asarray([32 * int(i / 8) + 16 for i in range(64)] * 8)
    labels_len = len(labels)
    classMat = np.zeros(shape=(labels_len, labels_len))

    for i in range(labels_len):
        gt = labels[i]
        for j in range(labels_len):
            if gt != labels[j]:
                classMat[i, j] = 1
    np.save(outName, classMat.astype(np.float32))
