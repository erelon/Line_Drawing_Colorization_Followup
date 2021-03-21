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
    os.makedirs("preprocessed_data_tars", exist_ok=True)

    for i, (gt, bw, name) in enumerate(dataloader):
        print(f"{i} / 1024")
        os.makedirs("preprocessed_data/" + name[0][:name[0].find("/")], exist_ok=True)
        torch.save(gt, "preprocessed_data/" + name[0] + ".gt.pt")
        torch.save(bw, "preprocessed_data/" + name[0] + ".train.pt")

        tar = tarfile.open(f"preprocessed_data_tars/{name[0][:name[0].find('/')]}.tar.bz2", mode="w:bz2")
        tar.add("preprocessed_data/" + name[0][:name[0].find("/") + 1])
        tar.close()

        shutil.rmtree("preprocessed_data/" + name[0][:name[0].find("/")], ignore_errors=True)
        if i == 4:
            break
    shutil.make_archive("preprocessed_data_tars", format="tar", root_dir="preprocessed_data_tars", base_dir=".")
