import webdataset as wds
import torch

from torch.utils.data import TensorDataset
import scipy.stats as stats
from WebDatasetHelper import my_decoders

from models import siggraph17_L
import pytorch_lightning as pl
from GAN import NetI, NetG, NetD, NetF
import os
from pytorch_lightning.loggers.neptune import NeptuneLogger


def mask_gen(size=64):
    maskS = size  # // 4

    mu, sigma = 1, 0.005
    X = stats.truncnorm((0 - mu) / sigma, (1 - mu) / sigma, loc=mu, scale=sigma)

    mask1 = torch.cat(
        [torch.rand(1, 1, maskS, maskS).ge(X.rvs(1)[0]).float() for _ in range(2 // 2)], 0)
    mask2 = torch.cat([torch.zeros(1, 1, maskS, maskS).float() for _ in range(2 // 2)], 0)
    mask = torch.cat([mask1, mask2], 0)

    return mask


def dummy_func(*args):
    return True


if __name__ == '__main__':
    try:
        import multiprocessing as mp

        __spec__ = None
        mp.set_start_method('spawn', force=True)
    except:
        pass

    all_tars = []

    neptune_logger = NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiMDM3MjFkYy1jNTE3LTQ4NTAtOTFlNC00ZGY1NGM3Y2M4YmEifQ====",
        project_name="erelon39/Line-colorize")

    if torch.cuda.is_available():
        decods = my_decoders(128)
        model = siggraph17_L(128, pretrained_path=None)
        for root, dirs, files in os.walk("/home/erelon39/sftp/erelon/df66f8bf-85ef-4dec-aa8f-464dd02ad15c"):
            for file in files:
                if file.endswith(".tar") and "out" not in root and "out" not in file:
                    all_tars.append(os.path.join(root, file))

        dataset = wds.WebDataset(all_tars, length=float("inf")) \
            .decode(decods.my_decoder_GT).decode(decods.my_decoder_BW).to_tuple("gt.jpg", "train.jpg", "__key__",
                                                                                handler=dummy_func).batched(16)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=4)

        checkpoint_callback_sample = pl.callbacks.ModelCheckpoint(monitor='sample')
        checkpoint_callback_loss_error = pl.callbacks.ModelCheckpoint(monitor='loss_error')

        trainer = pl.Trainer(gpus=1, log_every_n_steps=10, max_epochs=10, profiler=False, val_check_interval=500,
                             callbacks=[checkpoint_callback_sample, checkpoint_callback_loss_error],
                             distributed_backend='ddp', precision=16, logger=neptune_logger)
    else:
        decods = my_decoders(64)
        model = siggraph17_L(64, pretrained_path=None)
        for root, dirs, files in os.walk("."):
            for file in files:
                if file.endswith(".tar"):
                    all_tars.append(os.path.join(root, file))
        dataset = wds.WebDataset(all_tars, length=float("inf")) \
            .decode(decods.my_decoder_GT).decode(decods.my_decoder_BW).to_tuple("gt.jpg", "train.jpg", "__key__",
                                                                                handler=dummy_func).batched(4)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=2)
        trainer = pl.Trainer(log_every_n_steps=10, max_epochs=10, profiler=True,
                             max_steps=500)  # , logger=neptune_logger)

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
