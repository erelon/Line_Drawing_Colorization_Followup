import torch.nn.functional as F
import gc

import torch
import numpy as np
from torch import Tensor

from CoreElements import prob2img, lch2rgb


def new_loss(predict, gt):
    loss = gt * predict
    loss = loss.sum(dim=1)
    loss = 1 * loss
    loss = -loss.sum()
    return loss / predict.shape[0]


def train(dataloader, model):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_epochs = 10
    train_loader = dataloader
    criterion = new_loss
    optimizer = torch.optim.Adam(model.parameters())
    model = model.to(device)
    for epoch in range(num_epochs):
        model.train()
        t_loss = 0
        for i, (labels, input_batch) in enumerate(train_loader):
            input_batch = input_batch.to(device)
            labels = labels.to(device)
            # forward
            outputs_probs = model(torch.tensor(input_batch, dtype=torch.uint8))
            loss = criterion(outputs_probs, labels)

            print(loss)

            t_loss += loss
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            gc.collect()
            torch.cuda.empty_cache()
            # predict
            # _, predicted = torch.max(outputs.data, 1)
        torch.save(model.state_dict(), f"model_iter{epoch}")
        print(f"Loss for epoch {epoch}: {t_loss / (i + 1)}")
    return model
