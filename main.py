import matplotlib.pyplot as plt
import webdataset as wds
import torch

from torch.utils.data import TensorDataset

from DataloaderHelper import collate
from WebDatasetHelper import my_decoder_GT, my_decoder_BW, SampleEqually
from models import siggraph17
from train_loop import train

if __name__ == '__main__':

    dataset_gt = wds.WebDataset("GT.tar").decode("rgb8").decode(my_decoder_GT).to_tuple("jpg;png", "__key__")
    dataset_td = wds.WebDataset("train_data.tar").decode("rgb8").decode(my_decoder_BW).to_tuple("jpg;png", "__key__")
    dataset = SampleEqually([dataset_gt, dataset_td])

    dataloader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=8, collate_fn=collate)

    # TODO:spit train and test

    model = siggraph17(pretrained=False)

    model = train(dataloader, model)
