from datetime import datetime

import cv2
import torch
from torch.utils.data import IterableDataset
import random
import webdataset as wds
import matplotlib.pyplot as plt
from torchvision import datasets

from CoreElements import rgb2lch, soft_encode_image, lch2rgb, rgb2lchTensor
from train_loop import back_to_color
import pickle
import re
import os

import numpy as np
import json
import tempfile
import io
import PIL.Image


class c_Shorthands(wds.Shorthands):
    def __init__(self):
        super().__init__()

    def c_sh(self, data, bufsize=1000, initial=100, rng=random, handler=None):
        initial = min(initial, bufsize)
        buf = []
        startup = True
        for i, sample in enumerate(data):
            sample_2 = next(data)
            if len(buf) < bufsize:
                try:
                    buf.append(next(data))
                    buf.append(next(data))  # skipcq: PYL-R1708
                except StopIteration:
                    pass
            k = rng.randint(0, int((len(buf) - 1) / 2))
            k *= 2
            sample, buf[k] = buf[k], sample
            sample_2, buf[k + 1] = buf[k + 1], sample_2
            if startup and len(buf) < initial:
                buf.append(sample)
                buf.append(sample_2)
                continue
            startup = False
            yield sample, sample_2
        for i in range(0, len(buf), 2):
            yield buf[i], buf[i + 1]

    def shuffle(self, size, **kw):
        if size < 1:
            return self
        return self.then(self.c_sh, size, **kw)


class SampleEqually(IterableDataset, c_Shorthands, wds.Composable):
    def __init__(self, datasets):
        super().__init__()
        self.datasets = datasets

    def __iter__(self):
        sources = [iter(ds) for ds in self.datasets]
        while True:
            for source in sources:
                try:
                    yield next(source)
                except StopIteration:
                    return


def my_decoder_GT(key, data):
    if "gt" not in key.lower():
        return None
    with io.BytesIO(data) as stream:
        img = PIL.Image.open(stream)
        img.load()
        # img = img.convert("RGB")
    result = np.asarray(img)

    t1 = datetime.now()
    im_GT = rgb2lch(result)
    e1 = datetime.now() - t1
    print(e1)

    t2 = datetime.now()
    aaa = rgb2lchTensor(torch.from_numpy(result.astype(np.float32)).to(torch.device('cuda')))
    e2 = datetime.now() - t2
    print(e2)

    # aaa = lch2rgb(aaa.type(torch.float64).numpy())
    # plt.imshow(aaa)
    # plt.show()
    # im_GT = lch2rgb(im_GT)
    # plt.imshow(im_GT)
    # plt.show()

    im_GT = soft_encode_image(im_GT)

    if type(im_GT) is torch.Tensor:
        return im_GT
    return torch.tensor(im_GT.astype(float))


def my_decoder_BW(key, data):
    if "train" not in key.lower():
        return None
    with io.BytesIO(data) as stream:
        img = PIL.Image.open(stream)
        img.load()
        # img = img.convert("RGB")
    value = np.asarray(img)

    im_BW = cv2.cvtColor(value, cv2.COLOR_RGB2GRAY)
    im_BW = im_BW.reshape((1, 256, 256))
    return torch.tensor(im_BW.astype("uint8"))
