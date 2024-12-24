import os
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from models import LDM, Encoder, Tokenizer, Scheduler
from tqdm import tqdm

def arg_parser():
    parser = argparse.ArgumentParser(description='test VAE model')
    parser.add_argument('--frame', type=int, default=8, help='frame number')
    parser.add_argument('--checkpoint', type=str, default='./weights/unet_new_250.pth', help='checkpoint path')
    parser.add_argument('--mod', type=str, default='HC', help='ASD, HC, label_train, label_test')
    return parser.parse_args()

args = arg_parser()

save_path_map = {'ASD': '/data/xuruipeng/datas/new/LDM_test_ASD_outputs', # /data/xuruipeng/datas/new/LDM_test_ASD_outputs ,
                 'HC': '/data/xuruipeng/datas/new/LDM_train_HC_outputs',
                 'label_train': './datas/150/LDM_train_label_outputs',
                 'label_test': './datas/150/LDM_test_label_outputs'}

if not os.path.exists(save_path_map[args.mod]):
    os.makedirs(save_path_map[args.mod])

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
def get_data_list(filepath='./datas/VAE_encoder_train_outputs'):

    files = os.listdir(filepath)
    files = [os.path.join(filepath, f) for f in files]
    return files

def load_pth_data(filepath):
    data = torch.load(filepath)
    return data

class MyDataset(Dataset):
    def __init__(self):
        self.data = get_data_list()
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d, l, filename = self.get_data_and_label(idx)
        return d, l, filename

    def get_data_and_label(self, idx):
        data = load_pth_data(self.data[idx])
        filename = self.data[idx].split('/')[-1]
        label = self.data[idx].split('/')[-1][:-4].split('@')[-1]
        return data, label, filename


encoder = Encoder.getEncoder().to(device)
scheduler = Scheduler.get_scheduler()
tokenizer = Tokenizer.get_tokenizer()
unet = LDM.UNet(args.frame).to(device)

unet.load_state_dict(torch.load(args.checkpoint, map_location=device))
encoder.eval()

unet.eval()

#准备测试
def generate(text, pre_frame=None):

    #[1, 77]
    pos = tokenizer(text,
                    padding='max_length',
                    max_length=77,
                    truncation=True,
                    return_tensors='pt').input_ids.to(device)
    # neg = tokenizer('',
    #                 padding='max_length',
    #                 max_length=77,
    #                 truncation=True,
    #                 return_tensors='pt').input_ids.to(device)

    #[1, 77, 768]
    pos = encoder(pos)
    # neg = encoder(neg)
    # print('pre_frame',pre_frame.shape)
    #[1+1, 77, 768] -> [2, 77, 768]
    # out_encoder = torch.cat((neg, pos), dim=0)
    out_encoder = pos

    out_vae = torch.randn_like(pre_frame).to(device)
    out_vae_c = torch.cat((out_vae, pre_frame), dim=1)

    scheduler.set_timesteps(50, device=device)
    with torch.no_grad():
        for time in scheduler.timesteps:
            # print(time)

            #[1+1, 4, 64, 64] -> [2, 4, 64, 64]
            # noise = torch.cat((out_vae_c, out_vae_c), dim=0)
            noise = out_vae_c
            noise = scheduler.scale_model_input(noise, time)

            #[2, 4, 64, 64],[2, 77, 768],scala -> [2, 4, 64, 64]
            pred_noise = unet(out_vae=noise, out_encoder=out_encoder, time=time)

            #[2, 4, 64, 64] -> [1, 4, 64, 64]
            # pred_noise = pred_noise[0] + 7.5 * (pred_noise[1] - pred_noise[0])

            #[1, 4, 64, 64]
            out_vae = scheduler.step(pred_noise, time, out_vae).prev_sample

    out_vae = 1 / 0.18215 * out_vae
    #[1, 4, 64, 64] -> [1, 3, 512, 512]
    # print('out', out_vae.shape)
    return out_vae

def main_for_test(mod='ASD'):
    mod_map = {'ASD': "Autism Spectrum Disorder", 'HC': "Healthy Control"}
    if mod in mod_map.keys():
        text = mod_map[mod]
    dataset = MyDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    # print(len(dataloader))
    # return
    for data, label, filename in tqdm(dataloader):
        data = data.squeeze(0)
        label = label[0]
        filename = filename[0]
        # print(data.shape, label)
        length = data.shape[0]
        if mod == 'label_train' or mod == 'label_test':
            text = mod_map[label]
        frame_num = length // args.frame
        delta = length - frame_num * args.frame
        res = data[:delta+args.frame]
        data = data[delta:delta+args.frame]
        data = data.view(1, -1, 64, 64).to(device)

        for i in range(frame_num-1):
            data = generate(text, data)
            tmp = data.view(8, 12, 64, 64).cpu()

            # print(res.device, tmp.device)
            res = torch.cat((res, tmp), dim=0)
            # break
        # print(res.shape)
        torch.save(res, os.path.join(save_path_map[mod], filename))
        # break

if __name__ == '__main__':

    main_for_test(args.mod)