import torchvision
import webdataset as wds
import torch
from pytorch_lightning.profiler import AdvancedProfiler

from torch.utils.data import TensorDataset

from WebDatasetHelper import my_decoder_GT, my_decoder_BW, SampleEqually, my_decoder_tensor, tarfilter
from constant_matrix_creator import gatherClassImbalanceInfo, createClassMatrix
from models import siggraph17_L
from CoreElements import back_to_color
import pytorch_lightning as pl

if __name__ == '__main__':
    # torchvision.set_image_backend('accimage')
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
        # dataset = wds.WebDataset("train_{0000000..0000001}.tar", length=float("inf")) \
        #     .decode(my_decoder_GT).decode(my_decoder_BW).to_tuple("gt.jpg", "train.jpg", "__key__").batched(6)
        dataset = wds.WebDataset("preprocessed_data_tars.tar", length=float("inf")) \
            .map(tarfilter).to_tuple("gt.pt", "train.pt", "__key__").batched(4)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=2)
        trainer = pl.Trainer(gpus=1, log_every_n_steps=10, max_epochs=10, profiler=True,
                             max_steps=150, distributed_backend='ddp', precision=16)
    else:
        dataset = wds.WebDataset("preprocessed_data_tars.tar", length=float("inf")) \
            .map(tarfilter).to_tuple("gt.pt","train.pt","__key__").batched(2)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None)
        trainer = pl.Trainer(log_every_n_steps=10, max_epochs=10, profiler=True, max_steps=5)

    trainer.fit(model, dataloader)
