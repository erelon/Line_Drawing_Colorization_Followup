import webdataset as wds
import torch

from torch.utils.data import TensorDataset

from DataloaderHelper import collate
from WebDatasetHelper import my_decoder_GT, my_decoder_BW, SampleEqually
from constant_matrix_creator import gatherClassImbalanceInfo, createClassMatrix
from models_lightning import siggraph17_L
from train_loop import train, back_to_color
import pytorch_lightning as pl

if __name__ == '__main__':
    try:
        import multiprocessing as mp

        __spec__ = None
        mp.set_start_method('spawn', force=True)
    except:
        pass
    dataset = wds.WebDataset("train.tar",length=float("inf")) \
        .decode(my_decoder_GT).decode(my_decoder_BW).to_tuple("gt.jpg", "train.jpg", "__key__").batched(4)
    # .decode("torchrgb8").decode(
    # my_decoder_GT).to_tuple("jpg;png", "__key__")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=2)

    # for labels, train_data, name in dataloader:
    #     x = 5
    # TODO:spit train and test
    # createClassMatrix()
    # gatherClassImbalanceInfo(dataloader)
    #
    model = siggraph17_L(pretrained_path=None)
    trainer = pl.Trainer(log_every_n_steps=10, max_epochs=10, profiler=True, max_steps=50)
    trainer.fit(model, dataloader)
