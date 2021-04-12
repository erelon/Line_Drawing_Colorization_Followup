import cv2
import matplotlib
import torchvision
import webdataset as wds
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
import scipy.stats as stats

from CoreElements import prob2RGBimg, back_to_color, rgb2lch, soft_encode_image
from WebDatasetHelper import my_decoders

from models import siggraph17_L
import pytorch_lightning as pl
from GAN import *
import os
from pytorch_lightning.loggers.neptune import NeptuneLogger


def dummy_func(*args):
    return True


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
        project_name="erelon39/Line-colorize")

    if torch.cuda.is_available():
        decods = my_decoders(128)
        model = siggraph17_L(128, pretrained_path="model_e0_batch_19000_gn.pt")
        for root, dirs, files in os.walk("/home/erelon39/sftp/erelon/df66f8bf-85ef-4dec-aa8f-464dd02ad15c"):
            for file in files:
                if file.endswith(
                        ".tar") and "out" not in root and "out" not in file and "trash" not in root.lower() and "trash" not in file.lower():
                    all_tars.append(os.path.join(root, file))

        dataset = wds.WebDataset(all_tars, length=float("inf")) \
            .decode(decods.my_decoder_GT).decode(decods.my_decoder_BW).to_tuple("gt.jpg", "train.jpg", "__key__",
                                                                                handler=dummy_func).batched(16)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=4)

        trainer = pl.Trainer(gpus=1, log_every_n_steps=10, max_epochs=10, profiler=False, precision=16,
                             distributed_backend='ddp', logger=neptune_logger)
    else:
        decods = my_decoders(128)
        model = siggraph17_L(128, pretrained_path="model_e0_batch_4000.pt")
        for root, dirs, files in os.walk("."):
            for file in files:
                if file.endswith(".tar"):
                    all_tars.append(os.path.join(root, file))
        dataset = wds.WebDataset(all_tars, length=float("inf")).decode(decods.my_decoder_GT).decode(
            decods.my_decoder_BW).to_tuple("gt.jpg", "train.jpg", "__key__",
                                           handler=dummy_func).batched(2)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=0)
        trainer = pl.Trainer(log_every_n_steps=10, max_epochs=10, profiler=True,
                             max_steps=500)  # , logger=neptune_logger)

    # trainer.fit(model, dataloader)

    for labels, input_batch, name in dataloader:
        imgs = model.predict(input_batch, 0)
        gts = back_to_color(labels, show=False)
        fig, Axs = plt.subplots(3, len(name))
        for i, (pred, gt, inp) in enumerate(zip(imgs, gts, input_batch)):
            Axs[0][i].imshow(gt)
            Axs[1][i].imshow(pred)
            Axs[2][i].imshow(inp.reshape(128, 128, 1), cmap='gray')
        fig.savefig("examples/" + str([l for l in name]).strip("[]").replace("\'", "").replace("/", "").replace(",",
                                                                                                                "").replace(
            " ", "_") + "_predicts.jpg")

    # tsize = 128
    # netI = NetI()
    # netG = NetG(tsize)
    # for labels, input_batch, name in dataloader:
    #     with torch.no_grad():
    #         feat_sim = netI(input_batch / 255).detach()
    #     img = prob2RGBimg(labels.type(torch.float))
    #
    #     resized_im = []
    #     for im in img:
    #         resized_im.append(cv2.resize(im, (tsize // 2, tsize // 2)))
    #
    #     img = torch.tensor(resized_im)
    #
    #     mask = mask_gen(tsize // 2)
    #     hint = torch.cat((img.reshape(-1, 3, tsize // 2, tsize // 2) * mask, mask), 1)
    #
    #     fake = netG(input_batch / 255, hint, feat_sim)
    #     for im in fake:
    #         img_fake = torchvision.vutils.make_grid(im.detach().mul(0.5).add(0.5), nrow=4).reshape(tsize, tsize, 3)
    #     x = 5
