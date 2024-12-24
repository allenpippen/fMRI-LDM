import os
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
from models import LDM, Encoder, Tokenizer, Scheduler

def arg_parser():
    parser = argparse.ArgumentParser(description='Test VAE model')
    parser.add_argument('--train_test', type=str, default='test', help='train or test')
    parser.add_argument('--asd_hc', type=str, default='asd', help='asd or hc')
    parser.add_argument('--test_sites', type=list, default=['CALTECH', 'CMU', 'NYU', 'SDSU', 'STANFORD', 'TRINITY', 'UM_1', 'UM_2', 'USM', 'YALE'], help='sites for testing')
    parser.add_argument('--train_sites', type=list, default=['KKI', 'LEUVEN_1', 'LEUVEN_2', 'MAX_MUN', 'OHSU', 'OLIN', 'PITT', 'SBL', 'UCLA_1', 'UCLA_2'], help='sites for training')
    parser.add_argument('--device', type=str, default='cuda:1', help='device to train on ...')
    parser.add_argument('--checkpoint', type=str, default="/data/xuruipeng/weights/vae_best.pth", help='path to checkpoint to resume training')
    parser.add_argument('--frame', type=int, default=8, help='frame number')
    parser.add_argument('--LDM_checkpoint', type=str, default='/data/xuruipeng/weights/unet_new_400.pth', help='checkpoint path')
    parser.add_argument('--repeat_num', type=int, default='16', help='model name')
    parser.add_argument('--sample_num', type=int, default='100', help='model name')
    return parser.parse_args()

args = arg_parser()


if args.asd_hc == 'asd':
    save_dictory = '/home/data1/xuruipeng/C_zero_ASD_outputs/recon'
else:
    save_dictory = '/home/data1/xuruipeng/C_zero_HC_outputs/recon'



class VAE_Encoder(nn.Module):
    def __init__(self, args):
        super(VAE_Encoder, self).__init__()
        self.vae = vae.VAE()
        self.load_checkpoint(args.checkpoint)

    def load_checkpoint(self, checkpoint):
        self.vae.load_state_dict(torch.load(checkpoint))
        print('VAE Encoder Model loaded from checkpoint...Done')

    def forward(self, x):
        coronal = x.permute((0, 2, 1, 3))
        sagittal = x
        axial = x.permute((0, 3, 2, 1)) 

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
        print('VAE Decoder Model loaded from checkpoint...Done')

    def forward(self, x):

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

        coronal = coronal.permute((0, 2, 1, 3))
        axial = axial.permute((0, 3, 2, 1)) 

        # combine the three views
        out = (coronal + sagittal + axial) / 3

        return out

encoder = Encoder.getEncoder().to(args.device)
scheduler = Scheduler.get_scheduler()
tokenizer = Tokenizer.get_tokenizer()
unet = LDM.UNet(args.frame).to(args.device)
vae_encoder = VAE_Encoder(args).to(args.device)
vae_decoder = VAE_Decoder(args).to(args.device)

unet.load_state_dict(torch.load(args.LDM_checkpoint, map_location=args.device))
encoder.eval()
# vae.eval()
unet.eval()
vae_encoder.eval()
vae_decoder.eval()

brain_mask = nib.load("./datas/mask.nii.gz")
brain_mask = brain_mask.get_fdata()

# brain_mask = np.stack([brain_mask] * args.frame, axis=0)

# dx_group = {'1': "Autism Spectrum Disorder", '2': "Healthy Control"}
dx_group = {'asd': "Autism Spectrum Disorder", 'hc': "Healthy Control"}

def generate(text, pre_frame=None):

    #[1, 77]
    # print(text)
    pos = tokenizer(text,
                    padding='max_length',
                    max_length=77,
                    truncation=True,
                    return_tensors='pt').input_ids.to(args.device)

    pos = encoder(pos)
    out_encoder = pos

    out_vae = torch.randn_like(pre_frame).to(args.device)
    # out_vae_c = torch.cat((out_vae, pre_frame), dim=1)

    scheduler.set_timesteps(50, device=args.device)
    with torch.no_grad():
        for time in scheduler.timesteps:

            noise = out_vae
            noise = scheduler.scale_model_input(noise, time)

            #[2, 4, 64, 64],[2, 77, 768],scala -> [2, 4, 64, 64]
            pred_noise = unet(out_vae=torch.cat((noise, pre_frame), dim=1), out_encoder=out_encoder, time=time)

            #[1, 4, 64, 64]
            out_vae = scheduler.step(pred_noise, time, out_vae).prev_sample

    out_vae = 1 / 0.18215 * out_vae
    #[1, 4, 64, 64] -> [1, 3, 512, 512]
    # print('out', out_vae.shape)
    return out_vae

img_affine = np.array([[-3., -0., 0., 90.], [-0., 3., -0., -126.], [0., 0., 3., -72.], [0., 0., 0., 1.]])

data_bar = tqdm(range(args.sample_num))
with torch.no_grad():
    for _ in data_bar:
        all_outs = []
        data = torch.zeros((1, 12*args.frame, 64, 64)).to(args.device)
        for s in range(args.repeat_num):
            data_bar.set_postfix({'processing step': '{}/{}'.format(s, args.repeat_num)})
            data = generate(dx_group[args.asd_hc], data)

            outs = torch.cat([data[:, _*12:(_+1)*12, :, :] for _ in range(args.frame)], dim=0)

            # print(outs.shape)

            # VAE decode process
            outs = vae_decoder(outs)
            # print(outs.shape)
            all_outs.append(outs)

        all_outs = torch.cat(all_outs, dim=0)
        # print(all_outs.shape)
        all_outs = all_outs.cpu().numpy()

        mask = np.stack([brain_mask] * all_outs.shape[0], axis=0)
        all_outs = all_outs * brain_mask
        all_outs = np.transpose(all_outs, (1, 2, 3, 0))

        new_image = nib.Nifti1Image(all_outs, img_affine)

        if not os.path.exists(save_dictory):
            os.makedirs(save_dictory)
        nib.save(new_image, save_dictory + f"/" + args.asd_hc + str(_) + "_gen.nii")
        # break