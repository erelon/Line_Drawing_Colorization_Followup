import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.lines import Line2D
from pytorch_lightning import LightningModule
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np


def conv_block(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
                         nn.ELU(True),
                         nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
                         nn.ELU(True),
                         nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0),
                         nn.AvgPool2d(kernel_size=2, stride=2))


def deconv_block(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                         nn.ELU(True),
                         nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
                         nn.ELU(True),
                         nn.UpsamplingNearest2d(scale_factor=2))


def blockUNet(in_c, out_c, name, transposed=False, bn=True, relu=True, dropout=False):
    block = nn.Sequential()
    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    if not transposed:
        block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False))
    else:
        block.add_module('%s_tconv' % name, nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False))
    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if dropout:
        block.add_module('%s_dropout' % name, nn.Dropout2d(0.5, inplace=True))
    return block


class D(nn.Module):
    def __init__(self, nc, ndf, hidden_size):
        super(D, self).__init__()

        # 256
        self.conv1 = nn.Sequential(nn.Conv2d(nc, ndf, kernel_size=3, stride=1, padding=1),
                                   nn.ELU(True))
        # 256
        self.conv2 = conv_block(ndf, ndf)
        # 128
        self.conv3 = conv_block(ndf, ndf * 2)
        # 64
        self.conv4 = conv_block(ndf * 2, ndf * 3)
        # 32
        self.encode = nn.Conv2d(ndf * 3, hidden_size, kernel_size=1, stride=1, padding=0)
        self.decode = nn.Conv2d(hidden_size, ndf, kernel_size=1, stride=1, padding=0)
        # 32
        self.deconv4 = deconv_block(ndf, ndf)
        # 64
        self.deconv3 = deconv_block(ndf, ndf)
        # 128
        self.deconv2 = deconv_block(ndf, ndf)
        # 256
        self.deconv1 = nn.Sequential(nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1),
                                     nn.ELU(True),
                                     nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1),
                                     nn.ELU(True),
                                     nn.Conv2d(ndf, nc, kernel_size=3, stride=1, padding=1),
                                     nn.Tanh())
        """
        self.deconv1 = nn.Sequential(nn.Conv2d(ndf,nc,kernel_size=3,stride=1,padding=1),
                                     nn.Tanh())
        """

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.encode(x)
        x = self.decode(x)
        x = self.deconv4(x)
        x = self.deconv3(x)
        x = self.deconv2(x)
        x = self.deconv1(x)
        return x


class G(nn.Module):
    def __init__(self, input_nc, output_nc, nf):
        super(G, self).__init__()

        # input is 256 x 256
        layer_idx = 1
        name = 'layer%d' % layer_idx
        layer1 = nn.Sequential()
        layer1.add_module(name, nn.Conv2d(input_nc, nf, 4, 2, 1, bias=False))
        # input is 128 x 128
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer2 = blockUNet(nf, nf * 2, name, transposed=False, bn=True, relu=False, dropout=False)
        # input is 64 x 64
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer3 = blockUNet(nf * 2, nf * 4, name, transposed=False, bn=True, relu=False, dropout=False)
        # input is 32
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer4 = blockUNet(nf * 4, nf * 8, name, transposed=False, bn=True, relu=False, dropout=False)
        # input is 16
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer5 = blockUNet(nf * 8, nf * 8, name, transposed=False, bn=True, relu=False, dropout=False)
        # input is 8
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer6 = blockUNet(nf * 8, nf * 8, name, transposed=False, bn=True, relu=False, dropout=False)
        # input is 4
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer7 = blockUNet(nf * 8, nf * 8, name, transposed=False, bn=True, relu=False, dropout=False)
        # input is 2 x  2
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer8 = blockUNet(nf * 8, nf * 8, name, transposed=False, bn=False, relu=False, dropout=False)

        ## NOTE: decoder
        # input is 1
        name = 'dlayer%d' % layer_idx
        d_inc = nf * 8
        dlayer8 = blockUNet(d_inc, nf * 8, name, transposed=True, bn=True, relu=True, dropout=True)

        # import pdb; pdb.set_trace()
        # input is 2
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        d_inc = nf * 8 * 2
        dlayer7 = blockUNet(d_inc, nf * 8, name, transposed=True, bn=True, relu=True, dropout=True)
        # input is 4
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        d_inc = nf * 8 * 2
        dlayer6 = blockUNet(d_inc, nf * 8, name, transposed=True, bn=True, relu=True, dropout=True)
        # input is 8
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        d_inc = nf * 8 * 2
        dlayer5 = blockUNet(d_inc, nf * 8, name, transposed=True, bn=True, relu=True, dropout=False)
        # input is 16
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        d_inc = nf * 8 * 2
        dlayer4 = blockUNet(d_inc, nf * 4, name, transposed=True, bn=True, relu=True, dropout=False)
        # input is 32
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        d_inc = nf * 4 * 2
        dlayer3 = blockUNet(d_inc, nf * 2, name, transposed=True, bn=True, relu=True, dropout=False)
        # input is 64
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        d_inc = nf * 2 * 2
        dlayer2 = blockUNet(d_inc, nf, name, transposed=True, bn=True, relu=True, dropout=False)
        # input is 128
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer1 = nn.Sequential()
        d_inc = nf * 2
        dlayer1.add_module('%s_relu' % name, nn.ReLU(inplace=True))
        dlayer1.add_module('%s_tconv' % name, nn.ConvTranspose2d(d_inc, output_nc, 4, 2, 1, bias=False))
        dlayer1.add_module('%s_tanh' % name, nn.Tanh())

        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4
        self.layer5 = layer5
        self.layer6 = layer6
        self.layer7 = layer7
        self.layer8 = layer8
        self.dlayer8 = dlayer8
        self.dlayer7 = dlayer7
        self.dlayer6 = dlayer6
        self.dlayer5 = dlayer5
        self.dlayer4 = dlayer4
        self.dlayer3 = dlayer3
        self.dlayer2 = dlayer2
        self.dlayer1 = dlayer1

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        # out6 = self.layer6(out5)
        # out7 = self.layer7(out6)
        x = self.layer8(out5)
        x = self.dlayer8(x)
        # dout8_out7 = torch.cat([dout8, out7], 1)
        # dout7 = self.dlayer7(dout8_out7)
        # dout7_out6 = torch.cat([dout8, out6], 1)
        # dout6 = self.dlayer6(dout7_out6)
        x = torch.cat([x, out5], 1)
        x = self.dlayer5(x)
        x = torch.cat([x, out4], 1)
        x = self.dlayer4(x)
        x = torch.cat([x, out3], 1)
        x = self.dlayer3(x)
        x = torch.cat([x, out2], 1)
        x = self.dlayer2(x)
        x = torch.cat([x, out1], 1)
        x = self.dlayer1(x)
        return x


class GAN(LightningModule):
    def __init__(self):
        super().__init__()

        # networks
        self.generator = G(3, 3, 32)
        self.discriminator = D(3, 3, 32)

        self.criterionCAE = nn.L1Loss()
        self.k = 0

    def forward(self, z):
        return self.generator(z)

    def plot_grad_flow(self):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.

        Usage: Plug this function in Trainer class after loss.backwards() as
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        ave_grads = []
        max_grads = []
        layers = []
        for n, p in self.named_parameters():
            if (p.requires_grad) and ("bias" not in n):
                layers.append(n)
                try:
                    ave_grads.append(p.grad.abs().mean())
                    max_grads.append(p.grad.abs().max())
                except:
                    layers.remove(n)
                    pass
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.4, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.4, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        plt.savefig("grads.png")

    # def backward(self, loss, optimizer, optimizer_idx: int, *args, **kwargs) -> None:
    #     if optimizer_idx == 0:
    #         l1, l2 = loss
    #         l1.backward()
    #         l2.backward()
    #     if optimizer_idx == 1:
    #         loss.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()

    def training_step(self, batch, batch_idx, optimizer_idx):
        labels, input_batch, name = batch
        input_batch = input_batch.permute(0, 3, 1, 2) / 255
        labels = labels.permute(0, 3, 1, 2) / 255

        # train generator
        if optimizer_idx == 0:
            # generate images
            self.generated_imgs = self(input_batch)
            g_loss = self.criterionCAE(self.generated_imgs, labels) * .5
            g_loss.backward(retain_graph=True)

            recon_fake = self.discriminator(self.generated_imgs)
            errG = torch.mean(torch.abs(recon_fake - self.generated_imgs))

            self.log("Generator L1 loss", g_loss)
            self.log("Generator error loss", errG)
            self.log("Generator total loss", g_loss + errG)

            return errG

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples
            recon_real = self.discriminator(labels)
            fake = torch.tensor(self.generated_imgs, requires_grad=False)
            recon_fake = self.discriminator(fake)
            # compute L(x)
            errD_real = torch.mean(torch.abs(recon_real - labels))
            # compute L(G(z_D))
            errD_fake = torch.mean(torch.abs(recon_fake - fake))
            # compute L_D
            errD = errD_real - self.k * errD_fake

            balance = (0.7 * errD_real - errD_fake).data
            self.k = min(max(self.k + 0.001 * balance, 0), 1)
            self.log("Discriminator loss", errD)

            if batch_idx % 500 == 0:
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                ax1.imshow(input_batch.permute(0, 2, 3, 1)[0].clone().detach().cpu().type(torch.float32))
                ax2.imshow(self.generated_imgs.permute(0, 2, 3, 1)[0].clone().detach().cpu().type(torch.float32))
                ax3.imshow(labels.permute(0, 2, 3, 1)[0].clone().detach().cpu().type(torch.float32))
                self.log("K value", self.k)
                # self.logger.experiment.log_image('sample', fig)
                plt.close(fig)
                torch.save(self.generator.state_dict(), f"Gan/gan_gen_e{self.current_epoch}_batch_{batch_idx}.pt")
                torch.save(self.discriminator.state_dict(),
                           f"Gan/gan_dis_e{self.current_epoch}_batch_{batch_idx}.pt")

            return errD

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.0)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.0)
        return [opt_g, opt_d], []
