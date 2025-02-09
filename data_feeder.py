import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import feature_extract
import os

def load_label(label_file):
    labels = {}
    wav_lists = []
    encode = {'spoof': 0, 'bonafide': 1}
    with open(label_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if len(line) > 1:
                wav_id = line[1]
                wav_lists.append(wav_id)
                try:
                    tmp_label = encode[line[4]]
                except:
                    tmp_label = encode[line[5]]
                labels[wav_id] = tmp_label

    return labels, wav_lists

def load_data(flac_path, dataset, label_file, mode="train", feature_type="fft", ext="flac"):
    if mode!="eval":
        ids, data, label = load_train_data(flac_path, dataset, label_file, feature_type="fft", ext=ext)
        return ids, data,label
    else:
        data, folder_list, flag = load_eval_data(dataset, label_file, feature_type="fft", ext=ext)
        # Path to the ASVspoof2019LA evaluation set WAV files
        return data, folder_list, flag

def load_train_data(flac_path, dataset, label_file, feature_type="fft", ext="flac"):
    labels, wav_lists = load_label(label_file)
    final_data = []
    final_label = []
    ids = []

    for wav_id in tqdm(wav_lists, desc="load {} data".format(dataset)):
        ids.append(wav_id)
        label = labels[wav_id]
        
        wav_path = os.path.join(flac_path, f"{wav_id}.{ext}")

        if os.path.exists(wav_path):
            final_data.append(wav_path)
            final_label.append(label)
        else:
            print("can not open {}".format(wav_path))
        
    return ids, final_data, final_label

def load_eval_data(dataset, scp_file, feature_type="fft", ext="flac"):
    wav_lists = []
    folder_list={}
    flag = {}
    with open(scp_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if len(line) > 1:
                wav_id = line[1]
                wav_lists.append(wav_id)
                folder_list[wav_id]=line[-1]
                if line[-2] == '-':
                    flag[wav_id] = 'A00'
                else:
                    flag[wav_id] = line[-2]
    return wav_lists, folder_list, flag

def collate_fn_padd(batch):
    labels = torch.tensor([t[1] for t in batch])
    batch = [t[0].permute(1,0) for t in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch)
    batch = batch.permute(2,1,0)
    return batch, labels

class ASVDataSet(Dataset):

    def __init__(self, data, label, wav_ids=None, transform=True, mode="train", lengths=None, feature_type="fft", aug=None):
        super(ASVDataSet, self).__init__()
        self.data = data
        self.label = label
        self.wav_ids = wav_ids
        self.transform = transform
        self.lengths = lengths
        self.mode = mode
        self.feature_type=feature_type
        self.aug = aug

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        each_data, each_label = self.data[idx], self.label[idx]
        each_data = feature_extract.extract(each_data, self.feature_type, self.aug)
        if self.transform:
            each_data=torch.Tensor(each_data)
        return each_data, each_label

if __name__ == '__main__':
    print("Done")