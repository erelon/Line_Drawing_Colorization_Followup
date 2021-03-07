import gc
import torch
import numpy as np
import torch.nn.functional as F
from CoreElements import prob2img, lch2rgb


def new_loss(predict, gt, device="cpu"):
    # class_weights = torch.tensor(np.load("imbalance_vector.npy"), dtype=torch.float32).to(device)
    loss = F.KLDivLoss(predict, gt.permute([0,3,1,2]))  # ,weight=class_weights )

    #F.kl_div(F.log_softmax(predict),gt.permute([0,3,1,2]),log_target=True)

    # M = torch.tensor(np.load("chroma_loss.npy"), dtype=torch.float32).to(device)

    # loss += gt * M * torch.log(predict.permute([0, 2, 3, 1]))
    return loss


def back_to_color(labels):
    import numpy as np
    import matplotlib.pyplot as plt
    ims = prob2img(labels)
    for im in ims:
        im = im.detach().cpu().numpy()

        final_img = np.moveaxis(im, [0], [-1])
        # final_img = np.moveaxis(final_img, [1], [0])
        final_im_rgb = lch2rgb(final_img.squeeze())
        plt.imshow(final_im_rgb)
        plt.show()


def train(dataloader, model, epochs=10):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"working on device: {device}")
    num_epochs = epochs
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
            loss = criterion(outputs_probs, labels, device=device)

            print(loss)

            t_loss += loss
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            gc.collect()
            torch.cuda.empty_cache()
            # back_to_color(outputs_probs)
            # predict
            # _, predicted = torch.max(outputs.data, 1)
        torch.save(model.state_dict(), f"model_iter{epoch}")
        print(f"Loss for epoch {epoch}: {t_loss / (i + 1)}")
    return model
