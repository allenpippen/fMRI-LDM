

import torch
import torch.nn as nn
import utils.dataloader as dl
import models.VAE as vae
from tqdm import tqdm
import numpy as np
import prettytable as pt
import argparse

def arg_parser():
    parser = argparse.ArgumentParser(description='Train VAE model')
    parser.add_argument('--batch_size', type=int, default=8, help='train batch size')
    parser.add_argument('--test_batch_size', type=int, default=16, help='train batch size')
    parser.add_argument('--epochs', type=int, default=40, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
    parser.add_argument('--save_path', type=str, default='/data/xuruipeng/weights/vae_best.pth', help='path to save model')
    parser.add_argument('--eval_step', type=int, default=3, help='evaluate every n epochs')
    parser.add_argument('--early_stop', type=int, default=10, help='early stop if loss does not decrease for n epochs')
    parser.add_argument('--test_sites', type=list, default=['CALTECH', 'CMU', 'NYU', 'SDSU', 'STANFORD', 'TRINITY', 'UM_1', 'UM_2', 'USM', 'YALE'], help='sites for testing')
    parser.add_argument('--train_sites', type=list, default=['KKI', 'LEUVEN_1', 'LEUVEN_2', 'MAX_MUN', 'OHSU', 'OLIN', 'PITT', 'SBL', 'UCLA_1', 'UCLA_2'], help='sites for training')
    parser.add_argument('--device', type=str, default='cuda:1', help='device to train on ...')
    parser.add_argument('--checkpoint', type=str, default="/data/xuruipeng/weights/vae_best.pth", help='path to checkpoint to resume training')
    parser.add_argument('--accelerate', type=int, default=2, help='')
    return parser.parse_args()

def train(args):

    vae_model = vae.VAE().to(args.device)

    if args.checkpoint is not None:
        vae_model.load_state_dict(torch.load(args.checkpoint))
        print('Model loaded from checkpoint...Done  ')

    optimizer = torch.optim.Adam(vae_model.parameters(), lr=args.lr)

    train_dataset = dl.MyDataset('./datas/ABIDE_CPAC_dataset.json', sites=args.train_sites)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataset = dl.MyDataset('./datas/ABIDE_CPAC_dataset.json', sites=args.test_sites)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

    criterion = nn.MSELoss()
    best_loss = 1e10
    early_stop = 0
    early_stop_flag = False

    for epoch in range(args.epochs):
        vae_model.train()
        train_loss = 0
        train_bar = tqdm(train_dataloader, colour='blue')
        for i, train_data_item in enumerate(train_bar):
            train_bar.desc = "Train epoch[{}/{}]".format(epoch + 1, args.epochs)
            datas, lbel = train_data_item
            datas = datas.float()
            datas = datas.to(args.device)
            optimizer.zero_grad()
            # print(datas.shape) # [1, TPs, 61, 73, 61]

            batchnum = datas.shape[1] // args.batch_size
            for j in range(batchnum):
                # break
                data_batch = datas[0][j * args.batch_size:(j + 1) * args.batch_size]
                # data_batch = data_batch.to(args.device)
                outs = vae_model(data_batch)
                # print(data_batch.shape, outs.shape)
                loss = criterion(outs, data_batch)
                # l2_norm = sum(p.pow(2).sum() for p in vae_model.parameters())
                # loss += 0.1 * l2_norm
                loss.backward()
                # optimizer.step()
                train_loss += loss.item()
                train_bar.set_postfix(current_loss=loss.item(), total_loss=train_loss)
                # break
            if datas.shape[1] % args.batch_size != 0:
                data_batch = datas[0][batchnum * args.batch_size:]
                # data_batch = data_batch.to(args.device)
                outs = vae_model(data_batch)
                loss = criterion(outs, data_batch)
                # l2_norm = sum(p.pow(2).sum() for p in vae_model.parameters())
                # loss += 0.1 * l2_norm
                loss.backward()
                # optimizer.step()
                train_loss += loss.item()
                train_bar.set_postfix(current_loss=loss.item(), total_loss=train_loss)
            if i == 0 or i == len(train_bar) - 1 or (i % args.accelerate) == 0:
                optimizer.step()
                optimizer.zero_grad()

            torch.save(vae_model.state_dict(), "/data/xuruipeng/weights/vae_tmp.pth")

            # break
        if (epoch + 1) % args.eval_step == 0 or epoch == 0 or epoch == args.epochs - 1:
            vae_model.eval()
            test_loss = 0
            test_bar = tqdm(test_dataloader, colour='green')
            for i, test_data_item in enumerate(test_bar):
                test_bar.desc = "Test  epoch[{}/{}]".format(epoch + 1, args.epochs)
                datas, lbel = test_data_item
                datas = datas.float()
                test_batchnum = datas.shape[1] // args.test_batch_size
                datas = datas.to(args.device)
                with torch.no_grad():
                    for j in range(test_batchnum):
                        data_batch = datas[0][j * args.test_batch_size:(j + 1) * args.test_batch_size]
                        # data_batch = data_batch.to(args.device)
                        outs = vae_model(data_batch)
                        loss = criterion(outs, data_batch)
                        test_loss += loss.item()
                        test_bar.set_postfix(current_loss=loss.item(), total_loss=test_loss)
                        # break
                    if datas.shape[1] % args.test_batch_size != 0:
                        data_batch = datas[0][test_batchnum * args.test_batch_size:]
                        # data_batch = data_batch.to(args.device)
                        outs = vae_model(data_batch)
                        loss = criterion(outs, data_batch)
                        test_loss += loss.item()
                        test_bar.set_postfix(current_loss=loss.item(), total_loss=test_loss)

            if test_loss < best_loss:
                print("Saving model weight... Loss decreased from {:.4f} to {:.4f}".format(best_loss, test_loss))
                best_loss = test_loss
                torch.save(vae_model.state_dict(), args.save_path)
                early_stop = 0

            with open('./logs/VAE_test.txt', 'a') as f:
                f.write(f'{epoch} {test_loss}\n')

        with open('./logs/VAE_train.txt', 'a') as f:
            f.write(f'{epoch} {train_loss}\n')

        early_stop += 1

        if early_stop >= args.early_stop:
            print("Early stop at epoch {}".format(epoch + 1))
        # break
    torch.save(vae_model.state_dict(), "/data/xuruipeng/weights/vae_final.pth")

if __name__ == '__main__':

    args = arg_parser()
    train(args)
