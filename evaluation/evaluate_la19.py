import time
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from torch.autograd import Variable
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from compute_eer import get_eer, calculate_tDCF_EER
from data_feeder import ASVDataSet, load_data
import feature_extract
from model import AASIST, FFN
from tqdm import tqdm
import logging
import json 
from sklearn.metrics import det_curve

torch.set_num_threads(5)

def save_evaluation(model, output_dir, feature, test_loader, ids, device):
    model.eval()
    score_list = []
    labels = []
    w = open(os.path.join(output_dir, "la19.txt"),'a')
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(test_loader, desc="Evaluating LA19 data")):
            data, target = data.to(device), target.to(device)
            output = model(data)
            # result = output[0]
            result = output
            result = result.to("cpu")
            result = (result[:, 1]).data.cpu().numpy().ravel() 
            score_list.extend(result)
            labels.extend(target.view(-1).type(torch.int64).to("cpu").tolist())

    for i in range(len(ids)):
        lab = "spoof" if labels[i] == 0 else "bonafide"
        w.write(f"{ids[i]} {lab} {str(score_list[i])}\n")

    return score_list, labels

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Mamba Trials Setup')
    parser.add_argument("-o", "--out_fold", type=str, help="output folder", default='/netscratch/yelkheir/standard/outputs/')
    parser.add_argument('--test-batch-size', type=int, default=32, help='input batch size for testing (default: 32)')
    parser.add_argument('--feature_type', default='fft')
    parser.add_argument('--batch-size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--eval_protocol', type=str, default="/netscratch/yelkheir/datasets/asvsoof/eval.txt", help='.txt file: path, label Eval')
    parser.add_argument('--eval_path', type=str, default="/ds/audio/LA_19/ASVspoof2019_LA_eval/flac/", help='flac folder: Eval')
    parser.add_argument('--ASV_path', type=str, default="/ds/audio/LA_19/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt", help='flac folder: Eval')

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    logging.info(f"Using CUDA: {use_cuda} (Available: {torch.cuda.is_available()})")
    
    device = "cuda" if use_cuda else "cpu"
    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1, 'pin_memory': True, 'shuffle': False})

    # protocol_path
    eval_protocol = args.eval_protocol
    eval_flac = args.eval_path
    
    eval_ids, eval_data, eval_label = load_data(eval_flac, "eval LA19", eval_protocol, mode="train")
    eval_dataset = ASVDataSet(eval_data, eval_label, mode="eval")
    eval_dataloader = DataLoader(eval_dataset, **kwargs)
    
    with open(os.path.join(args.out_fold, "config.json"), 'r') as file:
        config = json.load(file)

    if config["back"] == "AASIST":
        model = AASIST(config).to(device)
    else:
        model = FFN(config).to(device)
 
    model.load_state_dict(torch.load(os.path.join(args.out_fold, "TF_best.pt")))
    scores, targets = save_evaluation(model, args.out_fold, args.feature_type, eval_dataloader, eval_ids, device)

    # get EERs and Min Detection Error Tradeoff
    EER = get_eer(targets, scores)
    print(EER)
    asv_key_file = args.ASV_path

    calculate_tDCF_EER(
        os.path.join(args.out_fold, "la19.txt"),
        asv_key_file,
        os.path.join(args.out_fold, "results_la19.txt")
    )

if __name__ == '__main__':
    s = time.time()
    main()
    e = time.time()
    total = round(e - s)
    h = total // 3600
    m = (total-3600*h) // 60
    s = total - 3600*h - 60*m
    print(f"cost time {h}:{m}:{s}")
