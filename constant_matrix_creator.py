import numpy as np
import torch


def gatherClassImbalanceInfo(dataloader, outName="imbalance_vector"):
    lamb = 0.5
    Q = 512

    p = torch.zeros((Q))

    counter = 0
    for i, (labels, _) in enumerate(dataloader):
        # print(".", sep="")
        p += labels.reshape((labels.shape[0], -1, 512)).argmax(axis=1).sum(axis=0)
        counter += labels.shape[0]
        print(f"Done {counter} images.")


    tmpP = p / counter
    w = 1 / ((1 - lamb) * tmpP + lamb / Q)
    scale = (tmpP * w).sum()
    w /= scale

    w = w.numpy()
    np.save(outName, w.astype(np.float32))


def createClassMatrix(outName):
    labels = np.asarray([32 * int(i / 8) + 16 for i in range(64)] * 8)
    labels_len = len(labels)
    classMat = np.zeros(shape=(labels_len, labels_len))

    for i in range(labels_len):
        gt = labels[i]
        for j in range(labels_len):
            if gt != labels[j]:
                classMat[i, j] = 1

    np.save(outName, classMat.astype(np.float32))
