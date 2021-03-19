import webdataset as wds
import torch
from pytorch_lightning.profiler import AdvancedProfiler

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




    # for labels, train_data, name in dataloader:
    #     x = 5
    # TODO:spit train and test
    # createClassMatrix()
    # gatherClassImbalanceInfo(dataloader)
    #
    model = siggraph17_L(pretrained_path=None)
    if torch.cuda.is_available():
        dataset = wds.WebDataset("train_{0000000..0000001}.tar", length=float("inf")) \
            .decode(my_decoder_GT).decode(my_decoder_BW).to_tuple("gt.jpg", "train.jpg", "__key__").batched(4)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=2)
        trainer = pl.Trainer(gpus=1, log_every_n_steps=10, max_epochs=10, profiler=AdvancedProfiler("prof_log.txt"),
                             max_steps=150, distributed_backend='ddp', precision=16)
    else:
        dataset = wds.WebDataset("train_{0000000..0000001}.tar", length=float("inf")) \
            .decode(my_decoder_GT).decode(my_decoder_BW).to_tuple("gt.jpg", "train.jpg", "__key__").batched(1)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None)
        trainer = pl.Trainer(log_every_n_steps=10, max_epochs=10, profiler=True, max_steps=5)
    trainer.fit(model, dataloader)
