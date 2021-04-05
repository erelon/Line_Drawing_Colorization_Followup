import cv2
import matplotlib
import torchvision
import webdataset as wds
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
import scipy.stats as stats

from CoreElements import prob2RGBimg, rgb2lchTensor, rgb2lch, soft_encode_image
from WebDatasetHelper import my_decoders

from models import siggraph17_L
import pytorch_lightning as pl
from GAN import *
import os
from pytorch_lightning.loggers.neptune import NeptuneLogger


def dummy_func(*args):
    return True


def coll(input):
    imgs, keys = input
    soft_encoded = torch.tensor([], dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    for img in imgs:
        if torch.cuda.is_available():
            im_GT = rgb2lchTensor(torch.from_numpy(img).cuda())
        else:
            im_GT = rgb2lch(img)
        im_GT = soft_encode_image(im_GT)
        if not torch.cuda.is_available():
            im_GT = im_GT.type(torch.float32)
        if torch.isnan(im_GT).any():
            raise ValueError
        soft_encoded = torch.cat([soft_encoded, im_GT.unsqueeze(0)], dim=0)

    if type(imgs) is not torch.Tensor:
        imgs = torch.tensor(imgs, dtype=torch.uint8,
                            device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    soft_encoded = torch.tensor(prob2RGBimg(soft_encoded),
                                device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")) * 255
    soft_encoded = torch.round(soft_encoded)

    return imgs, soft_encoded, keys


if __name__ == '__main__':
    try:
        import multiprocessing as mp

        __spec__ = None
        mp.set_start_method('spawn', force=True)
    except:
        pass

    matplotlib.use('Agg')

    all_tars = []

    neptune_logger = NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiMDM3MjFkYy1jNTE3LTQ4NTAtOTFlNC00ZGY1NGM3Y2M4YmEifQ====",
        project_name="erelon39/Line-recolorize")

    model = GAN()
    if torch.cuda.is_available():
        decods = my_decoders(128)
        for root, dirs, files in os.walk("/home/erelon39/sftp/erelon/df66f8bf-85ef-4dec-aa8f-464dd02ad15c"):
            for file in files:
                if file.endswith(".tar") and "out" not in root and "out" not in file:
                    all_tars.append(os.path.join(root, file))

        dataset = wds.WebDataset(all_tars, length=float("inf")) \
            .decode(decods.simple_decoder).to_tuple("gt.jpg", "__key__",
                                                    handler=dummy_func).batched(16)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=4, collate_fn=coll)

        trainer = pl.Trainer(gpus=1, log_every_n_steps=10, max_epochs=10, profiler=False, precision=16,
                             distributed_backend='ddp', logger=neptune_logger)
    else:
        decods = my_decoders(128)

        for root, dirs, files in os.walk("X:\\erelon\\df66f8bf-85ef-4dec-aa8f-464dd02ad15c"):
            for file in files:
                if file.endswith(".tar") and "out" not in root and "out" not in file:
                    all_tars.append("file:" + os.path.join(root, file))
        dataset = wds.WebDataset(all_tars, length=float("inf")) \
            .decode(decods.simple_decoder).to_tuple("gt.jpg", "__key__", handler=dummy_func).batched(8)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=2, collate_fn=coll)
        trainer = pl.Trainer(log_every_n_steps=10, max_epochs=10, logger=neptune_logger)

    trainer.fit(model, dataloader)
