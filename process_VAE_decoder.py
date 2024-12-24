import torch
import torch.nn as nn
import utils.dataloader as dl
import models.VAE as vae
from tqdm import tqdm
import numpy as np
import prettytable as pt
import argparse
import json
import os
import nibabel as nib
from torch.utils.data import Dataset, DataLoader


def arg_parser():
    parser = argparse.ArgumentParser(description='Test VAE model')
    parser.add_argument('--test_sites', type=list, default=['CALTECH', 'CMU', 'NYU', 'SDSU', 'STANFORD', 'TRINITY', 'UM_1', 'UM_2', 'USM', 'YALE'], help='sites for testing')
    parser.add_argument('--train_sites', type=list, default=['KKI', 'LEUVEN_1', 'LEUVEN_2', 'MAX_MUN', 'OHSU', 'OLIN', 'PITT', 'SBL', 'UCLA_1', 'UCLA_2'], help='sites for training')
    parser.add_argument('--device', type=str, default='cuda:1', help='device to train on ...')
    parser.add_argument('--checkpoint', type=str, default="/data/xuruipeng/weights/vae_best.pth", help='path to checkpoint to resume training')
    parser.add_argument('--data_dictory', type=str, default="/data/xuruipeng/datas/new/LDM_test_HC_outputs",  help='')
    parser.add_argument('--mask', type=str, default="./datas/mask.nii.gz", help='')

    return parser.parse_args()

def get_data_list(filepath='./datas/VAE_encoder_test_outputs'):
    # 读取文件夹下所有pth文件
    files = os.listdir(filepath)
    files = [os.path.join(filepath, f) for f in files]
    return files


def load_pth_data(filepath):
    data = torch.load(filepath)
    # 按维度0取均值
    # data = torch.mean(data, dim=0)
    # 调整维度 0,1,2,3 -> 1,2,3,0
    # data = data.permute(1, 2, 3, 0)
    return data

class VAE_Decoder(nn.Module):
    def __init__(self, args):
        super(VAE_Decoder, self).__init__()
        self.vae = vae.VAE()
        self.load_checkpoint(args.checkpoint)

    def load_checkpoint(self, checkpoint):
        self.vae.load_state_dict(torch.load(checkpoint))
        print('Model loaded from checkpoint...Done')

    def forward(self, x):
        # coronal = x.permute((0, 2, 1, 3))  # 冠状面 [n, 73, 61, 61]
        # sagittal = x  # 矢状面 [n, 61, 73, 61]
        # axial = x.permute((0, 3, 2, 1))  # 横截面 [n, 61, 73, 61]
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

def get_data_list(filepath='./datas/VAE_encoder_test_outputs'):
    # 读取文件夹下所有pth文件
    files = os.listdir(filepath)
    files = [os.path.join(filepath, f) for f in files]
    return files


def load_pth_data(filepath):
    data = torch.load(filepath)
    # data = torch.mean(data, dim=0)
    # data = data.permute(1, 2, 3, 0)
    return data

def test(args):

    brain_mask = nib.load(args.mask)
    brain_mask = brain_mask.get_fdata()

    test_dataset_files = get_data_list()
    raw_test_dataset = dl.data_loader_MyData('./datas/ABIDE_CPAC_dataset.json', sites=args.test_sites)

    encoder = VAE_Decoder(args)
    encoder.to(args.device)
    encoder.eval()
    cat = {}

    dx_group = {'1': "ASD", '2': "HC"}

    for dataitem in tqdm(raw_test_dataset):
        filepath = dataitem['filepath']
        filename = dataitem['SUB_ID']+'@'+dx_group[dataitem['DX_GROUP']]
        row_filename = filepath.split('/')[-1].split('.')[0]

        raw_img = nib.load(filepath)
        img_affine = raw_img.affine
        raw_img = raw_img.get_fdata()
        raw_img = np.transpose(raw_img, (3, 0, 1, 2))

        min_value = np.min(raw_img)
        max_value = np.max(raw_img)
        # print(min_value, max_value)

        img = load_pth_data(os.path.join(args.data_dictory, filename+'.pth'))
        img = img.float().to(args.device)
        all_outs = []
        for s in range(img.shape[0]):
            # print(s)
            # print(img[s])
            out = encoder(img[s].unsqueeze(0))
            out = out[0]
            # print(out.shape)
            out = out.cpu().detach().numpy()
            # mean = np.mean(out)
            #print(mean)
            # out = (out - mean) * (max_value - min_value)

            out = out * brain_mask
            # print(out)
            all_outs.append(out.tolist())
            # break

        # torch.save(torch.tensor(all_outs), f"./datas/VAE_encoder_train_outputs/"+dataitem['SUB_ID']+"@"+dx_group[dataitem['DX_GROUP']]+".pth")
        # print(all_outs)
        data = np.array(all_outs)
        data = np.transpose(data, (1, 2, 3, 0))
        # print(data.shape)

        new_image = nib.Nifti1Image(data, img_affine)

        if not os.path.exists(args.data_dictory+f"_recon/"):
            os.makedirs(args.data_dictory+f"_recon/")
        nib.save(new_image, args.data_dictory+f"_recon/"+row_filename+"_gen.nii")
        # break
    return cat

if __name__ == '__main__':

    args = arg_parser()
    res = test(args)


