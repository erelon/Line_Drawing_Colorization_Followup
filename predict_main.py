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


def coll(input):
    org, line, keys = input
    soft_encoded = torch.tensor([], dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                device=torch.device(
                                    "cpu"))  # torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    for img in org:
        im_GT = rgb2lch(img)
        im_GT = soft_encode_image(im_GT, torch.device('cpu'))
        if not torch.cuda.is_available():
            im_GT = im_GT.type(torch.float32)
        else:
            im_GT = im_GT.type(torch.float16)

        if torch.isnan(im_GT).any():
            raise ValueError
        soft_encoded = torch.cat([soft_encoded, im_GT.unsqueeze(0)], dim=0)

    if type(org) is not torch.Tensor:
        org = torch.tensor(org, dtype=torch.uint8, device=torch.device("cpu"))

    soft_encoded = torch.tensor(prob2RGBimg(soft_encoded, torch.device("cpu")),
                                device=torch.device("cpu")) * 255
    soft_encoded = torch.round(soft_encoded)
    soft_encoded[soft_encoded > 255] = 255
    soft_encoded[soft_encoded < 0] = 0

    return org, soft_encoded, line, keys


def dummy_func(*args):
    return True


if __name__ == '__main__':
    all_tars = []
    decods = my_decoders(128)
    model = siggraph17_L(128, pretrained_path="model_e0_batch_27500.pt")
    gan = G(3, 3, 32)
    gan.load_state_dict(torch.load("Gan/gan_gen_e7_batch_36000.pt", map_location=torch.device('cpu')))
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".tar"):
                all_tars.append(os.path.join(root, file))

    dataset = wds.WebDataset(all_tars, length=float("inf")).decode(decods.simple_decoder).decode(
        decods.my_decoder_BW).to_tuple("gt.jpg", "train.jpg", "__key__",
                                       handler=dummy_func).batched(2)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=0, collate_fn=coll)

    for orgs, soft_encoded, line, name in dataloader:
        imgs = model.predict(line, 0)
        gts = soft_encoded / 255
        # gan_se = gan(gts.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).detach()
        # gan_model = gan(torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).detach()
        fig, Axs = plt.subplots(len(name), 4, figsize=(20, 10))
        for axc in Axs:
            for ax in axc:
                ax.axis('off')
        for i, (pred, gt, inp, org) in enumerate(
                zip(imgs, gts, line, orgs)):  # , gan_se, gan_model)):
            Axs[i][1].imshow(gt)
            Axs[i][1].set_title("Soft encoded version")
            Axs[i][2].imshow(pred)
            Axs[i][2].set_title("Model prediction")
            Axs[i][0].imshow(inp.reshape(128, 128, 1), cmap='gray')
            Axs[i][0].set_title("Line version")
            Axs[i][-1].imshow(org)
            Axs[i][-1].set_title("Original image")
            # Axs[i][3].imshow(gan_from_real)
            # Axs[i][3].set_title("GAN result from Soft encoded v.")
            # Axs[i][4].imshow(gan_from_model)
            # Axs[i][4].set_title("GAN result from model prediction")
        # fig.show()

        fig.savefig("examples/" + str([l for l in name]).strip("[]").replace("\'", "").replace("/", "").replace(",",
                                                                                                                "").replace(
            " ", "_") + "_predicts.jpg")
