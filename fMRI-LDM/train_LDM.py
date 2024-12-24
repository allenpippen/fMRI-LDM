import os
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from models import LDM, Encoder, Tokenizer, Scheduler
from tqdm import tqdm

def arg_parser():
    parser = argparse.ArgumentParser(description='Train VAE model')
    parser.add_argument('--batch_size', type=int, default=2, help='train batch size')
    parser.add_argument('--frame', type=int, default=8, help='frame number')
    parser.add_argument('--accumulation_steps', type=int, default=4, help='gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=400, help='number of epochs')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path')
    return parser.parse_args()

args = arg_parser()

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
        d, l = self.get_data_and_label(idx)
        return d, l

    def get_data_and_label(self, idx):
        data = load_pth_data(self.data[idx])
        label = self.data[idx].split('/')[-1][:-4].split('@')[-1]
        return data, label
class ResNet18(torch.nn.Module):
    def __init__(self, input_channel):
        super(ResNet18, self).__init__()
        self.resnet18 = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=False)
        self.resnet18.conv1 = torch.nn.Conv2d(input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet18.fc = torch.nn.Linear(512, 2)

    def forward(self, x):
        return self.resnet18(x)


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

encoder = Encoder.getEncoder().to(device)
scheduler = Scheduler.get_scheduler()
tokenizer = Tokenizer.get_tokenizer()
unet = LDM.UNet(args.frame).to(device)
if args.checkpoint:
    unet.load_state_dict(torch.load(args.checkpoint))
    print('load checkpoint from', args.checkpoint)
#prepare to train
encoder.requires_grad_(False)
unet.requires_grad_(True)

encoder.eval()
unet.train()
cls = ResNet18(96).to(device)
cls.train()
optimizer = torch.optim.AdamW(unet.parameters(),
                              lr=1e-5,
                              betas=(0.9, 0.999),
                              weight_decay=0.01,
                              eps=1e-8)

criterion = torch.nn.MSELoss()
cls_criterion = torch.nn.CrossEntropyLoss()

loader = DataLoader(MyDataset(), batch_size=1, shuffle=True)

label_map = {'ASD': "Autism Spectrum Disorder", 'HC': "Healthy Control"}
cls_lable_map = {'ASD': 1, 'HC': 0}
# print(label_map.keys())
def get_loss(out_vae, out_pre, out_encoder, cls_lable):
    # print(out_vae.shape, out_pre.shape, out_encoder.shape)
    #0.18215 = vae.config.scaling_factor
    out_vae = out_vae * 0.18215

    noise = torch.randn_like(out_vae)

    #1000 = scheduler.num_train_timesteps
    #1 = batch size
    noise_step = torch.randint(0, 1000, (1, )).long().to(device)
    out_vae_noise = scheduler.add_noise(out_vae, noise, noise_step)

    out_input = torch.cat([out_vae_noise, out_pre], dim=1)
    # print(out_input.shape)
    out_unet = unet(out_vae=out_input,
                    out_encoder=out_encoder,
                    time=noise_step)
    # ops_unet = unet(out_vae=out_input, out_encoder=opsit_out_encoder, time=noise_step)
    # print(out_unet.shape, noise.shape)
    out_unet_ = out_unet - noise

    # print(out_unet_.shape, out_unet.shape, noise.shape)
    # ops_unet_ = ops_unet - noise

    # out_unet_ = out_unet_.reshape(out_unet.shape[0], -1)
    # ops_unet_ = ops_unet_.reshape(out_unet.shape[0], -1)
    #
    # distence = cos(out_unet_, ops_unet_)
    # print(distence.shape, distence)
    # label = torch.zeros(distence.shape[0]).float().to(device)

    # print(label.shape, label)

    logits = cls(out_unet_)
    cls_lable = cls_lable.repeat(out_unet_.shape[0])


    #[1, 4, 64, 64],[1, 4, 64, 64]
    return criterion(out_unet, noise) + cls_criterion(logits, cls_lable)

def train():
    loss_sum = 0
    for epoch in range(args.epochs):
        loss_sum = 0
        train_bar = tqdm(loader, colour='blue')
        train_bar.set_description(f'Epoch {epoch+1}/{args.epochs}')
        accelerate_cnt = 0
        for i, data in enumerate(train_bar):
            d, l = data
            d = d[0].to(device) # [n, 12*frames, 64, 64]
            clip_nums = d.size(0) // args.frame
            d = d[:clip_nums*args.frame]
            l = l[0]
            if l == 'ASD':
                ops = 'HC'
            else:
                ops = 'ASD'
            cls_label = torch.tensor([cls_lable_map[l]]).to(device)
            with torch.no_grad():
                out_encoder_ids = tokenizer(label_map[l],
                    padding='max_length',
                    max_length=77,
                    truncation=True,
                    return_tensors='pt').input_ids.to(device)
                # [1, 77] -> [1, 77, 768]
                out_encoder = encoder(out_encoder_ids)
                ops_out_encoder_ids = tokenizer(label_map[ops], padding='max_length', max_length=77, truncation=True, return_tensors='pt').input_ids.to(device)
                ops_out_encoder = encoder(ops_out_encoder_ids)
            for j in range(clip_nums):
                target = d[j*args.frame:(j+1)*args.frame]
                target = target.view(1, -1, 64, 64)
                if j == 0:
                    pre = torch.zeros(target.size()).to(device)
                else:
                    pre = d[(j-1)*args.frame:j*args.frame]
                    pre = pre.view(1, -1, 64, 64)
                if args.batch_size == 1:
                    out_encoder_target_batch = target
                    out_encoder_pre_batch = pre
                    out_encoder_batch = out_encoder
                    # ops_out_encoder_batch = ops_out_encoder
                    # print(out_encoder_target_batch.shape, out_encoder_pre_batch.shape, out_encoder_batch.shape)
                elif (j + 1) % args.batch_size == 1:
                    out_encoder_target_batch = target
                    out_encoder_pre_batch = pre
                    out_encoder_batch = out_encoder
                    # ops_out_encoder_batch = ops_out_encoder
                else:
                    out_encoder_target_batch = torch.cat([out_encoder_target_batch, target], dim=0)
                    out_encoder_pre_batch = torch.cat([out_encoder_pre_batch, pre], dim=0)
                    out_encoder_batch = torch.cat([out_encoder_batch, out_encoder], dim=0)
                    # ops_out_encoder_batch = torch.cat([ops_out_encoder_batch, ops_out_encoder], dim=0)
                # print(out_encoder_target_batch.shape, out_encoder_pre_batch.shape, out_encoder_batch.shape)
                if (j + 1) % args.batch_size == 0:
                    # print("train")
                    loss = get_loss(out_encoder_target_batch, out_encoder_pre_batch, out_encoder_batch, cls_label)
                    loss.backward()
                    loss_sum += loss.item()
                    train_bar.set_postfix(loss=loss.item(), loss_sum=loss_sum)
                    accelerate_cnt += 1
                    if accelerate_cnt % args.accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()

        if epoch % 1 == 0:
            print(epoch, loss_sum)
            with open('./logs/loss@plus@plus@'+str(args.frame)+'@'+str(args.batch_size)+'@'+'.txt', 'a') as f:
                f.write(f'{epoch} {loss_sum}\n')
        if (epoch+1) % 50 == 0:
            print('save weights ... ')
            torch.save(unet.state_dict(), f'/data/xuruipeng/weights/unet_new_{epoch+1}.pth')

        # break


if __name__ == '__main__':
    train()