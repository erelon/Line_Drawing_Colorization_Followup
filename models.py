from typing import Any, Optional

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

from CoreElements import back_to_color, prob2LCHimg, prob2RGBimg
from constant_matrix_creator import gatherLiveClassImbalanceInfo
import matplotlib.pyplot as plt


class BaseColor(pl.LightningModule):
    def __init__(self):
        super(BaseColor, self).__init__()

        self.l_cent = 50.
        self.l_norm = 100.
        self.ab_norm = 110.

    def normalize_l(self, in_l):
        return (in_l - self.l_cent) / self.l_norm

    def unnormalize_l(self, in_l):
        return in_l * self.l_norm + self.l_cent

    def normalize_ab(self, in_ab):
        return in_ab / self.ab_norm

    def unnormalize_ab(self, in_ab):
        return in_ab * self.ab_norm


class SIGGRAPHGenerator(BaseColor):
    def __init__(self, im_size, norm_layer=nn.BatchNorm2d, classes=512, imbalance_weights=None):
        super(SIGGRAPHGenerator, self).__init__()

        if imbalance_weights is None:
            self.imbalance_weights = torch.ones(classes)
        else:
            self.imbalance_weights = imbalance_weights
        self.classes = classes
        self.size = im_size
        # Conv1
        TEMP = 4
        SIZE = int(im_size / TEMP)
        model1 = [nn.Conv2d(1, SIZE, kernel_size=3, stride=1, padding=1, bias=True), ]
        model1 += [nn.ReLU(True), ]
        model1 += [nn.Conv2d(SIZE, SIZE, kernel_size=3, stride=1, padding=1, bias=True), ]
        model1 += [nn.ReLU(True), ]
        model1 += [norm_layer(SIZE), ]
        # add a subsampling operation

        # Conv2
        model2 = [nn.Conv2d(SIZE, SIZE * 2, kernel_size=3, stride=1, padding=1, bias=True), ]
        model2 += [nn.ReLU(True), ]
        model2 += [nn.Conv2d(SIZE * 2, SIZE * 2, kernel_size=3, stride=1, padding=1, bias=True), ]
        model2 += [nn.ReLU(True), ]
        model2 += [norm_layer(SIZE * 2), ]
        # add a subsampling layer operation

        # Conv3
        model3 = [nn.Conv2d(SIZE * 2, SIZE * 4, kernel_size=3, stride=1, padding=1, bias=True), ]
        model3 += [nn.ReLU(True), ]
        model3 += [nn.Conv2d(SIZE * 4, SIZE * 4, kernel_size=3, stride=1, padding=1, bias=True), ]
        model3 += [nn.ReLU(True), ]
        model3 += [nn.Conv2d(SIZE * 4, SIZE * 4, kernel_size=3, stride=1, padding=1, bias=True), ]
        model3 += [nn.ReLU(True), ]
        model3 += [norm_layer(SIZE * 4), ]
        # add a subsampling layer operation

        # Conv4
        model4 = [nn.Conv2d(SIZE * 4, SIZE * 8, kernel_size=3, stride=1, padding=1, bias=True), ]
        model4 += [nn.ReLU(True), ]
        model4 += [nn.Conv2d(SIZE * 8, SIZE * 8, kernel_size=3, stride=1, padding=1, bias=True), ]
        model4 += [nn.ReLU(True), ]
        model4 += [nn.Conv2d(SIZE * 8, SIZE * 8, kernel_size=3, stride=1, padding=1, bias=True), ]
        model4 += [nn.ReLU(True), ]
        model4 += [norm_layer(SIZE * 8), ]

        # Conv5
        model5 = [nn.Conv2d(SIZE * 8, SIZE * 8, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), ]
        model5 += [nn.ReLU(True), ]
        model5 += [nn.Conv2d(SIZE * 8, SIZE * 8, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), ]
        model5 += [nn.ReLU(True), ]
        model5 += [nn.Conv2d(SIZE * 8, SIZE * 8, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), ]
        model5 += [nn.ReLU(True), ]
        model5 += [norm_layer(SIZE * 8), ]

        # Conv6
        model6 = [nn.Conv2d(SIZE * 8, SIZE * 8, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), ]
        model6 += [nn.ReLU(True), ]
        model6 += [nn.Conv2d(SIZE * 8, SIZE * 8, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), ]
        model6 += [nn.ReLU(True), ]
        model6 += [nn.Conv2d(SIZE * 8, SIZE * 8, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), ]
        model6 += [nn.ReLU(True), ]
        model6 += [norm_layer(SIZE * 8), ]

        # Conv7
        model7 = [nn.Conv2d(SIZE * 8, SIZE * 8, kernel_size=3, stride=1, padding=1, bias=True), ]
        model7 += [nn.ReLU(True), ]
        model7 += [nn.Conv2d(SIZE * 8, SIZE * 8, kernel_size=3, stride=1, padding=1, bias=True), ]
        model7 += [nn.ReLU(True), ]
        model7 += [nn.Conv2d(SIZE * 8, SIZE * 8, kernel_size=3, stride=1, padding=1, bias=True), ]
        model7 += [nn.ReLU(True), ]
        model7 += [norm_layer(SIZE * 8), ]

        # Conv7
        model8up = [nn.ConvTranspose2d(SIZE * 8, SIZE * 4, kernel_size=4, stride=2, padding=1, bias=True)]
        model3short8 = [nn.Conv2d(SIZE * 4, SIZE * 4, kernel_size=3, stride=1, padding=1, bias=True), ]

        model8 = [nn.ReLU(True), ]
        model8 += [nn.Conv2d(SIZE * 4, SIZE * 4, kernel_size=3, stride=1, padding=1, bias=True), ]
        model8 += [nn.ReLU(True), ]
        model8 += [nn.Conv2d(SIZE * 4, SIZE * 4, kernel_size=3, stride=1, padding=1, bias=True), ]
        model8 += [nn.ReLU(True), ]
        model8 += [norm_layer(SIZE * 4), ]

        # classification output
        model_class = [nn.Conv2d(SIZE * 4, classes, kernel_size=1, padding=0, dilation=1, stride=1, bias=True), ]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8up = nn.Sequential(*model8up)
        self.model8 = nn.Sequential(*model8)
        self.model3short8 = nn.Sequential(*model3short8)

        self.model_class = nn.Sequential(*model_class)

        self.upsample4 = nn.Sequential(*[nn.Upsample(scale_factor=4, mode='bilinear'), ])
        self.softmax = nn.Sequential(*[nn.Softmax(dim=1), ])

    def forward(self, input_A, input_B=None, mask_B=None):
        # if (input_B is None):
        #     input_B = torch.cat((input_A * 0, input_A * 0), dim=1)
        # if (mask_B is None):
        #     mask_B = input_A * 0

        conv1_2 = self.model1(self.normalize_l(input_A))
        # conv1_2 = self.model1(self.normalize_l(input_A))
        conv2_2 = self.model2(conv1_2[:, :, ::2, ::2])
        conv3_3 = self.model3(conv2_2[:, :, ::2, ::2])
        conv4_3 = self.model4(conv3_3[:, :, ::2, ::2])
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)

        conv8_up = self.model8up(conv7_3) + self.model3short8(conv3_3)
        conv8_3 = self.model8(conv8_up)

        classes = self.model_class(conv8_3)
        upsmapeld = self.upsample4(classes)
        # unormlized = self.unnormalize_l(upsmapeld)
        # self.softmax(upsmapeld)
        return upsmapeld

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def CXE(self, predicted, target):
        # stacked_weights = self.imbalance_weights.repeat(target.shape[0], 1)
        stacked_weights = gatherLiveClassImbalanceInfo(target).repeat(target.shape[0], 1)
        weight = torch.gather(stacked_weights, dim=1, index=target.argmax(dim=1).reshape(-1, self.size ** 2)).reshape(
            [-1, self.size, self.size])
        return -(weight * (target * torch.log(predicted)).sum(dim=1)).mean()

    def training_step(self, data, batch_idx):
        labels, input_batch, name = data
        outputs_probs = self(input_batch)

        loss = self.CXE(F.softmax(outputs_probs, dim=1), labels.permute([0, 3, 1, 2]))

        try:
            if batch_idx % 10000 == 0:
                rgbs = prob2RGBimg(F.softmax(self(input_batch[0].unsqueeze(0)), dim=1).detach().permute([0, 2, 3, 1]))
                gt = prob2RGBimg(labels[0].type(torch.float).unsqueeze(0))
                fig, (ax1, ax2) = plt.subplots(1, 2)
                ax1.imshow(gt[0])
                ax2.imshow(rgbs[0])
                self.logger.experiment.log_image('sample', fig)
                plt.close(fig)
        except:
            pass
        self.log('train_loss', loss)
        return loss

    def predict(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None):
        return prob2RGBimg(F.softmax(self(batch), dim=1).permute([0, 2, 3, 1]))

    def online_predict(self, batch, batch_idx):
        import matplotlib.pyplot as plt

        for im in self.predict(batch, batch_idx).detach().numpy():
            plt.imshow(im)


def siggraph17_L(im_size, pretrained_path=None, weights=None):
    model = SIGGRAPHGenerator(im_size, imbalance_weights=weights)
    if (pretrained_path):
        model.load_state_dict(torch.load(pretrained_path, map_location=torch.device("cpu")))
        model.eval()

    return model
