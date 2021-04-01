import bz2
from datetime import datetime
import tarfile
import cv2
import torch
from torch.utils.data import IterableDataset
import random
import webdataset as wds
import matplotlib.pyplot as plt
from torchvision import datasets
from webdataset import Continue, tar_file_iterator, decode

from CoreElements import rgb2lch, soft_encode_image, lch2rgb, rgb2lchTensor, back_to_color
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


def my_decoder_GT_256(key, data):
    if "gt" not in key.lower():
        return None
    with io.BytesIO(data) as stream:
        img = PIL.Image.open(stream)
        img.load()
        img = img.resize((256, 256))

    if torch.cuda.is_available():
        result = np.asarray(img, dtype=np.float16)
        im_GT = rgb2lchTensor(torch.from_numpy(result).cuda())
    else:
        result = np.asarray(img)
        im_GT = rgb2lch(result)

    im_GT = soft_encode_image(im_GT)

    if type(im_GT) is torch.Tensor:
        return im_GT
    return torch.tensor(im_GT.astype(float))


def my_decoder_BW_256(key, data):
    if "train" not in key.lower():
        return None
    with io.BytesIO(data) as stream:
        img = PIL.Image.open(stream)
        img.load()
        img = img.resize((256, 256))
    value = np.asarray(img.convert("L"))

    # im_BW = cv2.cvtColor(value, cv2.COLOR_RGB2GRAY)
    # im_BW = value.reshape((1, 256, 256))
    im_BW = value.reshape((1, 256, 256))
    return torch.tensor(im_BW.astype("uint8"))


def my_decoder_GT_128(key, data):
    if "gt" not in key.lower():
        return None
    with io.BytesIO(data) as stream:
        img = PIL.Image.open(stream)
        img.load()
        img = img.resize((128, 128))

    if torch.cuda.is_available():
        result = np.asarray(img, dtype=np.float16)
        im_GT = rgb2lchTensor(torch.from_numpy(result).cuda())
    else:
        result = np.asarray(img)
        im_GT = rgb2lch(result)

    im_GT = soft_encode_image(im_GT)

    if type(im_GT) is torch.Tensor:
        return im_GT
    return torch.tensor(im_GT.astype(float))


def my_decoder_BW_128(key, data):
    if "train" not in key.lower():
        return None
    with io.BytesIO(data) as stream:
        img = PIL.Image.open(stream)
        img.load()
        img = img.resize((128, 128))
    value = np.asarray(img.convert("L"))

    # im_BW = cv2.cvtColor(value, cv2.COLOR_RGB2GRAY)
    # im_BW = value.reshape((1, 256, 256))
    im_BW = value.reshape((1, 128, 128))
    return torch.tensor(im_BW.astype("uint8"))


def my_decoder_GT_64(key, data):
    if "gt" not in key.lower():
        return None
    with io.BytesIO(data) as stream:
        img = PIL.Image.open(stream)
        img.load()
        img = img.resize((64, 64))

    if torch.cuda.is_available():
        result = np.asarray(img, dtype=np.float16)
        im_GT = rgb2lchTensor(torch.from_numpy(result).cuda())
    else:
        result = np.asarray(img)
        im_GT = rgb2lch(result)

    im_GT = soft_encode_image(im_GT)

    if type(im_GT) is torch.Tensor:
        return im_GT
    return torch.tensor(im_GT.astype(float))


def my_decoder_BW_64(key, data):
    if "train" not in key.lower():
        return None
    with io.BytesIO(data) as stream:
        img = PIL.Image.open(stream)
        img.load()
        img = img.resize((64, 64))
    value = np.asarray(img.convert("L"))

    # im_BW = cv2.cvtColor(value, cv2.COLOR_RGB2GRAY)
    # im_BW = value.reshape((1, 256, 256))
    im_BW = value.reshape((1, 64, 64))
    return torch.tensor(im_BW.astype("uint8"))


def tarfilter(data):
    data = tar_file_iterator(io.BytesIO(data["tar.bz2"]))
    sss = dict()
    for sample in data:
        LL = wds.autodecode.Decoder([my_decoder_tensor, wds.autodecode.basichandlers])
        a = LL.decode({sample[0]: sample[1]})
        sss[sample[0][sample[0].find(".") + 1:]] = a.popitem()[1]

    return sss


def my_decoder_tensor(key, data):
    if "pt" not in key:
        return None
    data = torch.load(io.BytesIO(data),
                      map_location=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    return data.squeeze(0)
