import cv2
import torchvision
import torchvision.utils as vutils
import webdataset as wds
import torch
import numpy as np
from pytorch_lightning.profiler import AdvancedProfiler
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
import scipy.stats as stats
from WebDatasetHelper import my_decoder_GT_128, my_decoder_BW_128, SampleEqually, my_decoder_tensor, tarfilter, \
    my_decoder_GT_64, my_decoder_BW_64, my_decoder_GT_256, my_decoder_BW_256
from constant_matrix_creator import gatherClassImbalanceInfo, createClassMatrix
from models import siggraph17_L
from CoreElements import back_to_color, prob2LCHimg, prob2RGBimg
import pytorch_lightning as pl
from GAN import NetI, NetG, NetD, NetF


def mask_gen(size=64):
    maskS = size  # // 4

    mu, sigma = 1, 0.005
    X = stats.truncnorm((0 - mu) / sigma, (1 - mu) / sigma, loc=mu, scale=sigma)

    mask1 = torch.cat(
        [torch.rand(1, 1, maskS, maskS).ge(X.rvs(1)[0]).float() for _ in range(2 // 2)], 0)
    mask2 = torch.cat([torch.zeros(1, 1, maskS, maskS).float() for _ in range(2 // 2)], 0)
    mask = torch.cat([mask1, mask2], 0)

    return mask


if __name__ == '__main__':
    # torchvision.set_image_backend('accimage')
    try:
        import multiprocessing as mp

        __spec__ = None
        mp.set_start_method('spawn', force=True)
    except:
        pass
    w = np.load("imbalance_vector.npy")
    model = siggraph17_L(64, weights=torch.tensor(w), pretrained_path=None)
    if torch.cuda.is_available():
        dataset = wds.WebDataset("train_{0000000..0000001}.tar", length=float("inf")) \
            .decode(my_decoder_GT_128).decode(my_decoder_BW_128).to_tuple("gt.jpg", "train.jpg", "__key__").batched(4)

        # dataset = wds.WebDataset("preprocessed_data_tars.tar", length=float("inf")) \
        #     .map(tarfilter).to_tuple("gt.pt", "train.pt", "__key__").batched(4)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=8)
        trainer = pl.Trainer(gpus=1, log_every_n_steps=100, max_epochs=10, profiler=False,
                             distributed_backend='ddp', precision=16)
    else:
        dataset = wds.WebDataset("train_{0000000..0000001}.tar", length=float("inf")) \
            .decode(my_decoder_GT_64).decode(my_decoder_BW_64).to_tuple("gt.jpg", "train.jpg", "__key__").batched(4)
        # dataset = wds.WebDataset("preprocessed_data_tars.tar", length=float("inf")) \
        #     .map(tarfilter).to_tuple("gt.pt", "train.pt", "__key__").batched(2)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None)
        trainer = pl.Trainer(log_every_n_steps=10, max_epochs=10, profiler=True, max_steps=50)

    trainer.fit(model, dataloader)
    # netI = NetI()
    # netG = NetG(64)
    # for labels, input_batch, name in dataloader:
    #     with torch.no_grad():
    #         feat_sim = netI(input_batch/255).detach()
    #     img = prob2RGBimg(labels.type(torch.float))
    #
    #     resized_im = []
    #     for im in img:
    #         resized_im.append(cv2.resize(im, (64, 64)))
    #
    #     img = torch.tensor(resized_im)
    #
    #     mask = mask_gen()
    #     hint = torch.cat((img.reshape(-1, 3, 64, 64) * mask, mask), 1)
    #
    #     fake = netG(input_batch / 255, hint, feat_sim)
    #     for im in fake:
    #         img_fake = vutils.make_grid(im.detach().mul(0.5).add(0.5), nrow=4).reshape(256, 256, 3)
    #     x = 5
