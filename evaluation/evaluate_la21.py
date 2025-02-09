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
import compute_eer_2 as em
from compute_eer import get_eer, calculate_tDCF_EER
from data_feeder import ASVDataSet, load_data
import feature_extract
from model import AASIST, FFN
from tqdm import tqdm
import logging
import json 
from sklearn.metrics import det_curve
import pandas 

torch.set_num_threads(5)

def save_evaluation(model, output_dir, feature, test_loader, ids, device):
    model.eval()
    score_list = []
    labels = []
    w = open(os.path.join(output_dir, "la21.txt"),'a')
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(test_loader, desc="Evaluating LA21 data")):
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

Pspoof = 0.05
cost_model = {
    'Pspoof': Pspoof,  # Prior probability of a spoofing attack
    'Ptar': (1 - Pspoof) * 0.99,  # Prior probability of target speaker
    'Pnon': (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker
    'Cmiss': 1,  # Cost of tandem system falsely rejecting target speaker
    'Cfa': 10,  # Cost of tandem system falsely accepting nontarget speaker
    'Cfa_spoof': 10,  # Cost of tandem system falsely accepting spoof
}

def load_asv_metrics(asv_key_file, asv_scr_file, phase='eval'):
    # Load organizers' ASV scores
    asv_key_data = pandas.read_csv(asv_key_file, sep=' ', header=None)
    asv_scr_data = pandas.read_csv(asv_scr_file, sep=' ', header=None)[asv_key_data[7] == phase]
    idx_tar = asv_key_data[asv_key_data[7] == phase][5] == 'target'
    idx_non = asv_key_data[asv_key_data[7] == phase][5] == 'nontarget'
    idx_spoof = asv_key_data[asv_key_data[7] == phase][5] == 'spoof'

    # Extract target, nontarget, and spoof scores from the ASV scores
    tar_asv = asv_scr_data[2][idx_tar]
    non_asv = asv_scr_data[2][idx_non]
    spoof_asv = asv_scr_data[2][idx_spoof]
    eer_asv, asv_threshold = em.compute_eer(tar_asv, non_asv)
    [Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv] = em.obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold)

    return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv

def performance(cm_scores, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, invert=False):
    bona_cm = cm_scores[cm_scores[5]=='bonafide']['2_x'].values
    spoof_cm = cm_scores[cm_scores[5]=='spoof']['2_x'].values

    if invert==False:
        eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]
    else:
        eer_cm = em.compute_eer(-bona_cm, -spoof_cm)[0]

    if invert==False:
        tDCF_curve, _ = em.compute_tDCF(bona_cm, spoof_cm, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, cost_model, False)
    else:
        tDCF_curve, _ = em.compute_tDCF(-bona_cm, -spoof_cm, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, cost_model, False)

    min_tDCF_index = np.argmin(tDCF_curve)
    min_tDCF = tDCF_curve[min_tDCF_index]

    return min_tDCF, eer_cm


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Mamba Trials Setup')
    parser.add_argument("-o", "--out_fold", type=str, help="output folder", default='/netscratch/yelkheir/standard/outputs/')
    parser.add_argument('--test-batch-size', type=int, default=32, help='input batch size for testing (default: 32)')
    parser.add_argument('--feature_type', default='fft')
    parser.add_argument('--batch-size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--eval_protocol', type=str, default="/ds/audio/LA_21/keys/LA/CM/trial_metadata.txt", help='.txt file: path, label Eval')
    parser.add_argument('--eval_path', type=str, default="/ds/audio/LA_21/ASVspoof2021_LA_eval/flac/", help='flac folder: Eval')
    parser.add_argument('--CM_path', type=str, default="/ds/audio/LA_21/keys/LA/CM/trial_metadata.txt", help='flac folder: Eval')
    parser.add_argument('--ASV_scores', type=str, default="/ds/audio/LA_21/keys/LA/ASV/ASVTorch_Kaldi/score.txt", help='flac folder: Eval')
    parser.add_argument('--ASV_path', type=str, default="/ds/audio/LA_21/keys/LA/ASV/trial_metadata.txt", help='flac folder: Eval')

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
    
    eval_ids, eval_data, eval_label = load_data(eval_flac, "eval LA21", eval_protocol, mode="train")
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

    asv_key_file = args.ASV_path
    asv_scr_file = args.ASV_scores
    cm_key_file = args.CM_path

    Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv = load_asv_metrics(asv_key_file, asv_scr_file)
    cm_data = pandas.read_csv(cm_key_file, sep=' ', header=None)
    submission_scores = pandas.read_csv(os.path.join(args.out_fold, "la21.txt"), sep=' ', header=None, skipinitialspace=True)

    if len(submission_scores) != len(cm_data):
        print('CHECK: submission has %d of %d expected trials.' % (len(submission_scores), len(cm_data)))
        exit(1)

    cm_scores = submission_scores.merge(cm_data[cm_data[7] == "eval"], left_on=0, right_on=1, how='inner')
    min_tDCF, eer_cm = performance(cm_scores, Pfa_asv, Pmiss_asv, Pfa_spoof_asv)

    out_data = "min_tDCF: %.4f\n" % min_tDCF
    out_data += "eer: %.2f\n" % (100*eer_cm)

    print(out_data)
    
    # just in case that the submitted file reverses the sign of positive and negative scores
    min_tDCF2, eer_cm2 = performance(cm_scores, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, invert=True)

    if min_tDCF2 < min_tDCF:
        print(
            'CHECK: we negated your scores and achieved a lower min t-DCF. Before: %.3f - Negated: %.3f - your class labels are swapped during training... this will result in poor challenge ranking' % (
            min_tDCF, min_tDCF2))

    if min_tDCF == min_tDCF2:
        print(
            'WARNING: your classifier might not work correctly, we checked if negating your scores gives different min t-DCF - it does not. Are all values the same?')


    w = open(os.path.join(args.out_fold, "results_la21.txt"),"w")
    w.write(out_data)

if __name__ == '__main__':
    s = time.time()
    main()
    e = time.time()
    total = round(e - s)
    h = total // 3600
    m = (total-3600*h) // 60
    s = total - 3600*h - 60*m
    print(f"cost time {h}:{m}:{s}")
