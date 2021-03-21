import io
import os
import pickle
import shutil
import torch
import webdataset as wds
from torch.utils.data import TensorDataset
import tarfile

from WebDatasetHelper import my_decoder_GT, my_decoder_BW

if __name__ == '__main__':
    dataset = wds.WebDataset("train_0000000.tar", length=float("inf")) \
        .decode(my_decoder_GT).decode(my_decoder_BW).to_tuple("gt.jpg", "train.jpg", "__key__").batched(1)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None)

    for i, (gt, bw, name) in enumerate(dataloader):
        os.makedirs("preprocessed_data/" + name[0][:name[0].find("/")], exist_ok=True)
        torch.save(gt, "preprocessed_data/" + name[0] + ".gt.pt")
        torch.save(bw, "preprocessed_data/" + name[0] + ".train.pt")


    shutil.make_archive("preprocessed_images", format="gztar", root_dir="preprocessed_data",base_dir=".")
