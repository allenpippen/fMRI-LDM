import torch
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import nibabel as nib
import prettytable as pt
import os

def data_loader_MyData(filename, sites=None):
    # 读取json文件
    with open(filename, 'r') as f:
        data = json.load(f)
    if sites is not None:
        datas = []
        for i in range(len(data)):
            if data[i]['SITE_ID'] in sites:
                datas.append(data[i])
        return datas
    else:
        return data

def get_data_list(filepath='./datas/VAE_encoder_test_outputs'):
    # 读取文件夹下所有pth文件
    files = os.listdir(filepath)
    files = [os.path.join(filepath, f) for f in files]
    return files

def read_niigz(filepath):
    img = nib.load(filepath)
    img_affine = img.affine
    img = img.get_fdata()
    img = np.transpose(img, (3, 0, 1, 2))
    # img = (img - np.min(img)) / (np.max(img) - np.min(img))
    # img = np.expand_dims(img, axis=1)
    return img, img_affine

class MyDataset(Dataset):
    def __init__(self, filepath, sites=None):
        self.data = data_loader_MyData(filename=filepath, sites=sites)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.get_fmri(idx), int(self.data[idx]['DX_GROUP'])-1

    def get_fmri(self, idx):
        data = read_niigz(self.data[idx]['filepath'])
        return data


if __name__ == '__main__':
    dataset = MyDataset('../datas/ABIDE_CPAC_dataset.json')
    test_sites = ['CALTECH', 'CMU', 'NYU', 'SDSU', 'STANFORD', 'TRINITY', 'UM_1', 'UM_2', 'USM', 'YALE']
    train_sites = ['KKI', 'LEUVEN_1', 'LEUVEN_2', 'MAX_MUN', 'OHSU', 'OLIN', 'PITT', 'SBL', 'UCLA_1', 'UCLA_2']
    print("Train sites:", train_sites)
    print("Test sites:", test_sites)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    # for i, data in enumerate(dataloader):
    #     data = data.squeeze(0)
    #     print(data.shape)
    #     if i == 10:
    #         break

    tb = pt.PrettyTable()
    tb.field_names = ['Sites', 'Num_sub', 'Num_data']
    train_dataset = MyDataset('../datas/ABIDE_CPAC_dataset.json', sites=train_sites)
    test_dataset = MyDataset('../datas/ABIDE_CPAC_dataset.json', sites=test_sites)
    train_length = len(train_dataset)
    test_length = len(test_dataset)
    # print(train_length, test_length)

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    train_num_data = 0

    for i, data in enumerate(train_dataloader):
        train_num_data += data[0].shape[1]
        # break
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    test_num_data = 0
    for i, data in enumerate(test_dataloader):
        test_num_data += data.shape[1]
        # break
    tb.add_row(['Train '+str(len(train_sites))+' sites', train_length, train_num_data])
    tb.add_row(['Test '+str(len(test_sites))+' sites', test_length, test_num_data])
    tb.add_row(['Total', train_length+test_length, train_num_data+test_num_data])
    print(tb)