import itertools
from datetime import datetime

import numpy as np
import torch
from skimage import color


def prob2img(probTensor: torch.Tensor):
    num_of_classes = probTensor.shape[3]
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # converts probability distribtiuon to an image
    CLASS_MAP_R = torch.from_numpy(np.asarray([32 * i + 16 for i in range(8)] * 64)).to(device)
    CLASS_MAP_G = torch.from_numpy(np.asarray([32 * int(i / 8) + 16 for i in range(64)] * 8)).to(device)
    CLASS_MAP_B = torch.from_numpy(np.asarray([32 * int(i / 64) + 16 for i in range(num_of_classes)])).to(device)

    eps = 1e-4
    out_img_dim = int(256)

    batch_sz = probTensor.shape[0]
    logits = torch.log(probTensor.reshape(batch_sz, out_img_dim ** 2, num_of_classes) + eps)

    logits_sub = (logits - logits.max(dim=2, keepdim=True).values) / 0.8

    unnormalized = torch.exp(logits_sub)

    probabilities = unnormalized / unnormalized.sum(dim=2, keepdim=True)

    # convert from 255 scale to the appropriate scale for lch, and then convert to rgb
    # treat H differently because it is a circular value (ref: https://en.wikipedia.org/wiki/Mean_of_circular_quantities)
    hAngles = CLASS_MAP_B * (2 * np.pi) / 255.0

    H = torch.atan2((hAngles.cos() * probabilities).sum(dim=2), (hAngles.cos() * probabilities).sum(dim=2))

    out_img = torch.stack(((CLASS_MAP_R * probabilities).sum(dim=2) * 100 / 255.0,
                           (CLASS_MAP_G * probabilities).sum(dim=2) * 100 / 255.0,
                           H), dim=2)

    out_img = out_img.reshape(batch_sz, out_img_dim, out_img_dim, 3)
    out_img = out_img.permute(0, 3, 1, 2)
    #
    # out_img = tf.reshape(tf.py_func(lch2rgb_batch, [out_img], Tout=tf.float32), shape=tf.shape(out_img))
    #
    # out_img = tf.image.resize_images(out_img, size=[IMG_DIM, IMG_DIM], method=tf.image.ResizeMethod.BILINEAR)

    return out_img


def gaussianDistTensor(pt1: torch.Tensor, pt2: torch.Tensor):
    std = 0.25
    sqdist = ((pt1 - pt2) ** 2).sum(axis=1)

    return torch.exp(-sqdist / std)


def gaussianDist(pt1: torch.Tensor, pt2: torch.Tensor):
    std = 0.25
    sqdist = np.sum((pt1 - pt2) ** 2, axis=1)

    return np.exp(-sqdist / std)


def soft_encode_image_tensor(img, device):
    num_of_classes = 512
    soft_encoding = torch.zeros((img.shape[1] ** 2, num_of_classes), device=device)
    img = img.type(torch.long)

    CHROMA_MAX = 100
    img[:, :, 0] = img[:, :, 0] * 255.0 / 100
    img[:, :, 1][img[:, :, 1] > CHROMA_MAX] = CHROMA_MAX
    img[:, :, 1] = img[:, :, 1] * 255.0 / CHROMA_MAX
    img[:, :, 2] = img[:, :, 2] * 255.0 / (2 * np.pi)

    discr_data = img.reshape([img.shape[1] ** 2, -1]) / 32.0
    center = (discr_data % 1 - 0.5)

    discr_data_int = discr_data.type(torch.long)

    rval = discr_data_int[:, 0]
    gval = discr_data_int[:, 1]
    bval = discr_data_int[:, 2]

    rs = [(rval, 0), (torch.clamp(rval + 1, max=7), 1), (torch.clamp(rval - 1, min=0), -1)]
    gs = [(gval, 0), (torch.clamp(gval + 1, max=7), 1), (torch.clamp(gval - 1, min=0), -1)]
    bs = [(bval, 0), ((bval + 1) % 8, 1), ((bval - 1) % 8, -1)]

    # rs = [(rval, 0), (np.minimum(rval + 1, 7), 1), (np.maximum(rval - 1, 0), -1)]
    # gs = [(gval, 0), (np.minimum(gval + 1, 7), 1), (np.maximum(gval - 1, 0), -1)]
    # bs = [(bval, 0), ((bval + 1) % 8, 1), ((bval - 1) % 8, -1)]

    coords = [rs, gs, bs]
    params = list(itertools.product(*coords))

    rng = range(img.shape[1] ** 2)
    for (rv, roff), (gv, goff), (bv, boff) in params:
        indx_1d = rv + (gv << 3) + (bv << 6)
        soft_encoding[rng, indx_1d] += gaussianDistTensor(center, torch.tensor([roff, goff, boff], device=device))
    # se1 = soft_encoding.clone()
    # se2 = soft_encoding.clone()
    # normalize, and clean up for efficient storage
    # soft_encoding = torch.tensor(soft_encoding, dtype=torch.float)
    if torch.cuda.is_available():
        soft_encoding = soft_encoding.type(torch.float16)
    soft_encoding = soft_encoding / torch.sum(soft_encoding, dim=1, keepdims=True)
    soft_encoding[soft_encoding < 1e-4] = 0
    soft_encoding = soft_encoding / torch.sum(soft_encoding, dim=1, keepdims=True)

    # s1 = datetime.now()
    # se1 = se1 / torch.sum(se1, dim=1, keepdims=True)

    # se1[se1 < 1e-4] = 0
    # se1 = se1 / torch.sum(se1, dim=1, keepdims=True)
    # print("Norm m: " + str(datetime.now() - s1))
    #
    # s2 = datetime.now()
    # # se2 =se2.type(torch.float)
    # se2 = se2 / torch.sum(se2, dim=1, keepdims=True)
    # se2[se2 < 1e-4] = 0
    # se2 = se2 / torch.sum(se2, dim=1, keepdims=True)
    # # se2 = se2.type(torch.float)
    # print("T m: " + str(datetime.now() - s2))

    soft_encoding = soft_encoding.reshape((img.shape[1], img.shape[1], num_of_classes))
    return soft_encoding


def soft_encode_image(img):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    img = torch.from_numpy(img).to(device)
    return soft_encode_image_tensor(img, device)

    # soft_encoding = np.zeros((img.shape[1] ** 2, 512))
    # img = img.astype(int)
    #
    # CHROMA_MAX = 100
    # img[:, :, 0] = img[:, :, 0] * 255.0 / 100
    # img[:, :, 1][img[:, :, 1] > CHROMA_MAX] = CHROMA_MAX
    # img[:, :, 1] = img[:, :, 1] * 255.0 / CHROMA_MAX
    # img[:, :, 2] = img[:, :, 2] * 255.0 / (2 * np.pi)
    #
    # discr_data = img.reshape([img.shape[1] ** 2, -1]) / 32.0
    # center = (discr_data % 1 - 0.5)
    #
    # discr_data_int = discr_data.astype(int)
    # rval = discr_data_int[:, 0]
    # gval = discr_data_int[:, 1]
    # bval = discr_data_int[:, 2]
    #
    # rs = [(rval, 0), (np.minimum(rval + 1, 7), 1), (np.maximum(rval - 1, 0), -1)]
    # gs = [(gval, 0), (np.minimum(gval + 1, 7), 1), (np.maximum(gval - 1, 0), -1)]
    # bs = [(bval, 0), ((bval + 1) % 8, 1), ((bval - 1) % 8, -1)]
    #
    # coords = [rs, gs, bs]
    # params = list(itertools.product(*coords))
    #
    # for (rv, roff), (gv, goff), (bv, boff) in params:
    #     indx_1d = rv + (gv << 3) + (bv << 6)
    #     soft_encoding[range(img.shape[1] ** 2), indx_1d] += gaussianDist(center, [roff, goff, boff])
    # # normalize, and clean up for efficient storage
    # soft_encoding = soft_encoding.astype(np.float16)
    #
    # soft_encoding = soft_encoding / np.sum(soft_encoding, axis=1)[:, np.newaxis]
    # soft_encoding[soft_encoding < 1e-4] = 0
    # soft_encoding = soft_encoding / np.sum(soft_encoding, axis=1)[:, np.newaxis]
    #
    # soft_encoding = soft_encoding.reshape((img.shape[1], img.shape[1], 512))
    # return soft_encoding


def rgb2lch(rgb):
    lab = color.rgb2lab(rgb)
    return color.lab2lch(lab)


def lch2rgb(lch):
    lab = color.lch2lab(lch)
    return color.lab2rgb(lab)
