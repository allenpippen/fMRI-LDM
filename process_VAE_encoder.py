import torch
import torch.nn as nn
import utils.dataloader as dl
import models.VAE as vae
from tqdm import tqdm
import numpy as np
import prettytable as pt
import argparse
import json
import nibabel as nib

def arg_parser():
    parser = argparse.ArgumentParser(description='Test VAE model')
    parser.add_argument('--test_sites', type=list, default=['CALTECH', 'CMU', 'NYU', 'SDSU', 'STANFORD', 'TRINITY', 'UM_1', 'UM_2', 'USM', 'YALE'], help='sites for testing')
    parser.add_argument('--train_sites', type=list, default=['KKI', 'LEUVEN_1', 'LEUVEN_2', 'MAX_MUN', 'OHSU', 'OLIN', 'PITT', 'SBL', 'UCLA_1', 'UCLA_2'], help='sites for training')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to train on ...')
    parser.add_argument('--checkpoint', type=str, default="/data/xuruipeng/weights/vae_best.pth", help='path to checkpoint to resume training')
    return parser.parse_args()

class VAE_Encoder(nn.Module):
    def __init__(self, args):
        super(VAE_Encoder, self).__init__()
        self.vae = vae.VAE()
        self.load_checkpoint(args.checkpoint)

    def load_checkpoint(self, checkpoint):
        self.vae.load_state_dict(torch.load(checkpoint))
        print('Model loaded from checkpoint...Done')

    def forward(self, x):
        coronal = x.permute((0, 2, 1, 3))  # [n, 73, 61, 61]
        sagittal = x  # [n, 61, 73, 61]
        axial = x.permute((0, 3, 2, 1))  # [n, 61, 73, 61]

        # encode
        h_coronal = self.vae.VAE_Coronal.encoder(coronal)
        h_sagittal = self.vae.VAE_Sagittal.encoder(sagittal)
        h_axial = self.vae.VAE_Axial.encoder(axial)

        # print(h_coronal.shape, h_sagittal.shape, h_axial.shape)

        # sample
        h_coronal = self.vae.VAE_Coronal.sample(h_coronal)
        h_sagittal = self.vae.VAE_Sagittal.sample(h_sagittal)
        h_axial = self.vae.VAE_Axial.sample(h_axial)

        return h_coronal, h_sagittal, h_axial

class VAE_Decoder(nn.Module):
    def __init__(self, args):
        super(VAE_Decoder, self).__init__()
        self.vae = vae.VAE()
        self.load_checkpoint(args.checkpoint)

    def load_checkpoint(self, checkpoint):
        self.vae.load_state_dict(torch.load(checkpoint))
        print('Model loaded from checkpoint...Done')

    def forward(self, x):
        # coronal = x.permute((0, 2, 1, 3))  # [n, 73, 61, 61]
        # sagittal = x  # [n, 61, 73, 61]
        # axial = x.permute((0, 3, 2, 1))  # [n, 61, 73, 61]
        #
        # # encode
        # h_coronal = self.vae.VAE_Coronal.encoder(coronal)
        # h_sagittal = self.vae.VAE_Sagittal.encoder(sagittal)
        # h_axial = self.vae.VAE_Axial.encoder(axial)
        #
        # # print(h_coronal.shape, h_sagittal.shape, h_axial.shape)
        #
        # # sample
        # h_coronal = self.vae.VAE_Coronal.sample(h_coronal)
        # h_sagittal = self.vae.VAE_Sagittal.sample(h_sagittal)
        # h_axial = self.vae.VAE_Axial.sample(h_axial)
        # print(x.shape) # torch.Size([1, 12, 64, 64])
        # h_coronal = self.vae.VAE_Coronal.sample(h_coronal)
        # h_sagittal = self.vae.VAE_Sagittal.sample(h_sagittal)
        # h_axial = self.vae.VAE_Axial.sample(h_axial)
        h_coronal = x[:, :4]
        h_sagittal = x[:, 4:8]
        h_axial = x[:, 8:]
        # print(h_coronal.shape, h_sagittal.shape, h_axial.shape)
        # decode
        coronal = self.vae.VAE_Coronal.decoder(h_coronal)
        sagittal = self.vae.VAE_Sagittal.decoder(h_sagittal)
        axial = self.vae.VAE_Axial.decoder(h_axial)

        # torch.Size([n, 73, 61, 61]) torch.Size([n, 61, 73, 61]) torch.Size([n, 61, 73, 61])
        # print(coronal.shape, sagittal.shape, axial.shape)

        coronal = coronal.permute((0, 2, 1, 3))  # 冠状面 [n, 61, 73, 61]
        axial = axial.permute((0, 3, 2, 1))  # 横截面 [n, 61, 73, 61]

        # combine the three views
        out = (coronal + sagittal + axial) / 3

        return out

def test(args):

    test_dataset = dl.data_loader_MyData('./datas/ABIDE_CPAC_dataset.json', sites=args.test_sites)

    encoder = VAE_Encoder(args)
    encoder.to(args.device)
    encoder.eval()
    cat = {}

    dx_group = {'1': "ASD", '2': "HC"}
    for dataitem in tqdm(test_dataset):
        img = dl.read_niigz(dataitem['filepath'])
        img = torch.from_numpy(img).float().to(args.device)
        all_outs = []
        for s in range(img.shape[0]):
            h_coronal, h_sagittal, h_axial = encoder(img[s].unsqueeze(0))
            outs = torch.cat((h_coronal, h_sagittal, h_axial), dim=1)
            outs = outs[0]
            outs = outs.cpu().detach().numpy()
            all_outs.append(outs.tolist())

        torch.save(torch.tensor(all_outs), f"./datas/VAE_encoder_test_outputs/"+dataitem['SUB_ID']+"@"+dx_group[dataitem['DX_GROUP']]+".pth")

    return cat

if __name__ == '__main__':

    args = arg_parser()
    res = test(args)


