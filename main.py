import webdataset as wds
import torch

from torch.utils.data import TensorDataset

from DataloaderHelper import collate
from WebDatasetHelper import my_decoder_GT, my_decoder_BW, SampleEqually
from constant_matrix_creator import gatherClassImbalanceInfo, createClassMatrix
from models import siggraph17
from train_loop import train, back_to_color

if __name__ == '__main__':
    dataset_gt = wds.WebDataset("GT_train.tar").decode("rgb8").decode(
        my_decoder_GT).to_tuple("jpg;png", "__key__")
    dataset_td = wds.WebDataset("train_data_train.tar").decode("rgb8").decode(
        my_decoder_BW).to_tuple("jpg;png", "__key__")
    dataset = SampleEqually([dataset_gt, dataset_td])
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=16, collate_fn=collate,
                                             prefetch_factor=1)

    # TODO:spit train and test
    # createClassMatrix()
    # gatherClassImbalanceInfo(dataloader)
    #
    model = siggraph17(pretrained_path="model_iter0")
    model = train(dataloader, model)

    # model = siggraph17(pretrained_path="model_iter0")
    # dataset_td = wds.WebDataset("train_data_train.tar").decode("rgb8").decode(
    #     my_decoder_BW).to_tuple("jpg;png", "__key__")
    # dataloader = torch.utils.data.DataLoader(dataset_td)
    # for i, (input_batch) in enumerate(dataloader):
    #     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #     model = model.to(device)
    #     back_to_color(model(torch.tensor(input_batch[0], dtype=torch.uint8).to(device)))
