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
import shutil

from optim import ScheduledOptim, set_seed
from data_feeder import ASVDataSet, load_data, collate_fn_padd

from model import AASIST, FFN

from tqdm import tqdm
import logging
import json 
from sklearn.metrics import det_curve

# torch.set_num_threads(5)
        
def get_eer(y_true, y_score):
    """
    A simple function to compute eer.
    More sophisticated metrics are available in this framework.
    This is a quick and easy implementation to use during validation.
    """
    fpr, fnr, thresholds = det_curve(y_true, y_score)
    index = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[index] + fnr[index]
    eer = eer/2
    return eer

def setup_logging(output_dir):
    """Sets up logging to log the progress of the model."""
    logging.basicConfig(
        filename=os.path.join(output_dir, 'training_log.log'),
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

def save_config(args, output_dir):
    """Saves the training configuration to a JSON file."""
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    logging.info(f"Configuration saved to {config_path}")

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    total_num = 0

    weight = torch.FloatTensor([0.5, 0.5]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    for batch_idx, (data, target) in enumerate(train_loader):
        data,target = data.to(device), target.to(device)
        data = data.permute(1,0,2)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.item()*data.size(0)
        total_num += data.size(0)
        loss.backward()
        optimizer.step()
        if args.dry_run:
            break

    return train_loss/total_num

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    total_num = 0
    correct = 0
    score_list = []
    labels = []

    weight = torch.FloatTensor([0.5, 0.5]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.permute(1,0,2)
            output = model(data)

            loss = criterion(output, target)
            test_loss += loss.item()*data.size(0)
            total_num += data.size(0)  

            result = (output[:, 1]  
                       ).data.cpu().numpy().ravel() 

            score_list.extend(result)
            labels.extend(target.view(-1).type(torch.int64).to("cpu").tolist())
    test_loss /= total_num
    eer = get_eer(labels, score_list)
    return test_loss, eer

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Mamba Trials Setup')
    parser.add_argument("-o", "--out_fold", type=str, help="output folder", default='./')
    parser.add_argument('--batch-size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=32, help='input batch size for testing (default: 32)')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.00001)')
    parser.add_argument('--warmup', type=float, default=1000, help='warmup steps (default: 1000)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False, help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=400, help='how many batches to wait before logging status')
    parser.add_argument('--save-model', action='store_true', default=True, help='For saving the current model')
    parser.add_argument('--feature_type', default='fft_full')
    parser.add_argument('--gpu', type=str, help='GPU index', default="1")

    parser.add_argument('--train_protocol', type=str, default="/netscratch/yelkheir/datasets/had/train.txt", help='.txt fileL path, label Train')
    parser.add_argument('--dev_protocol', type=str, default="/netscratch/yelkheir/datasets/had/dev.txt", help='.txt file: path, label Dev')
    parser.add_argument('--eval_protocol', type=str, default="/netscratch/yelkheir/datasets/had/eval.txt", help='.txt file: path, label Eval')
    parser.add_argument('--train_path', type=str, default="/ds/audio/Half-Truth/HAD/HAD_train/conbine", help='flac folder: Train')
    parser.add_argument('--dev_path', type=str, default="/ds/audio/Half-Truth/HAD/HAD_dev/conbine", help='flac folder: Dev')
    parser.add_argument('--eval_path', type=str, default="/ds/audio/Half-Truth/HAD/HAD_test/conbine", help='flac folder: Eval')
    parser.add_argument('--early_stopping', type=int, default=10, help='early stopping')

    parser.add_argument('--model', type=str, default="w2v", help='architectur multiconv type')
    parser.add_argument('--small', type=bool, default=0, help='small model or large')
    parser.add_argument('--n_layers', type=int, default=24, help='Number of layers')
    parser.add_argument('--back', type=str, default="AASIST", help='AASIST Model or FFN')

    args = parser.parse_args()
    config_model = {"model": args.model, "small": args.small, "n_layers": args.n_layers}
    
    args.out_fold = args.out_fold+f"models/{args.seed}/had/{args.back}/{args.seed}_{args.model}_{args.small}_{args.n_layers}_{args.back}"

    # Create output directories if they do not exist
    os.makedirs(args.out_fold, exist_ok=True)
    os.makedirs(os.path.join(args.out_fold, 'checkpoint'), exist_ok=True)

    setup_logging(args.out_fold)
    save_config(args, args.out_fold)

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = "cuda" if use_cuda else "cpu"
    dtype = torch.float32
    set_seed(args.seed)

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1, 'pin_memory': True, 'shuffle': True})

    # protocol_path
    train_protocol = args.train_protocol
    dev_protocol = args.dev_protocol
    
    # flac_folder
    train_flac = args.train_path
    dev_flac = args.dev_path
        
    if args.back == "AASIST":
        model = AASIST(config_model).to(device)
    else:
        model = FFN(config_model).to(device)
 
    model.ssl.freeze_model()

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr = args.lr, weight_decay=1e-4
        ),
        args.warmup
    )


    _ ,train_data, train_label = load_data(train_flac, "train", train_protocol, mode="train",  ext="wav")
    train_dataset = ASVDataSet(train_data, train_label, mode="train", feature_type=args.feature_type)
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn_padd, **kwargs)

    _ ,dev_data, dev_label = load_data(dev_flac, "dev", dev_protocol, mode="train",  ext="wav")
    dev_dataset = ASVDataSet(dev_data, dev_label, mode="train", feature_type=args.feature_type)
    dev_dataloader = DataLoader(dev_dataset, collate_fn=collate_fn_padd, **kwargs)

    # Early stopping parameters
    patience = args.early_stopping  
    counter = 0 
    best_loss = float('inf') 

    out = open(f"{args.out_fold}/writer.txt", "w")

    for epoch in range(1, args.epochs + 1):
        logging.info(f"Epoch {epoch} / {args.epochs}")
        
        # Training step
        train_loss = train(args, model, device, train_dataloader, optimizer, epoch)
        
        # Validation step
        loss, eer = test(model, device, dev_dataloader)

        weights = model.get_weights()
        weights_cpu = weights.cpu().tolist()

        logging.info(f"Epoch {epoch} completed with training loss: {train_loss}")
        logging.info(f"Epoch {epoch} completed with validation loss: {loss}")
        logging.info(f"Epoch {epoch} completed with validation EER: {eer*100}")

        out.write(f"Epoch {epoch} completed with training loss: {train_loss}\n")
        out.write(f"Epoch {epoch} completed with validation loss: {loss}\n")
        out.write(f"Epoch {epoch} completed with validation EER: {eer*100}\n")
        out.write("# Weights\n")  # Add a section header for clarity
        out.write(", ".join(map(str, weights_cpu)) + "\n")

        # Save model checkpoint
        torch.save(model.state_dict(), os.path.join(args.out_fold, 'checkpoint', f'senet_epoch_{epoch}.pt'))
        torch.save(optimizer.state_dict(), os.path.join(args.out_fold, 'checkpoint', f'op_epoch_{epoch}.pt'))

        # Check if the current loss is the best
        if loss < best_loss:
            best_loss = loss
            counter = 0  # Reset counter if there is an improvement
            if args.save_model:
                torch.save(model.state_dict(), os.path.join(args.out_fold, 'TF_best.pt'))
                torch.save(optimizer.state_dict(), os.path.join(args.out_fold, 'op.pt'))
                logging.info("Best model saved")
                out.write(f"Best model saved\n")

        else:
            counter += 1
            logging.info(f"No improvement for {counter} epoch(s)")
            out.write(f"No improvement for {counter} epoch(s)\n")

        # Early stopping condition
        if counter >= patience:
            logging.info(f"Early stopping triggered after {patience} epochs without improvement")
            out.write(f"Early stopping triggered after {patience} epochs without improvement\n")


            break

        out.flush()

if __name__ == '__main__':
    s = time.time()
    main()
    e = time.time()
    total = round(e - s)
    h = total // 3600
    m = (total-3600*h) // 60
    s = total - 3600*h - 60*m
    print(f"cost time {h}:{m}:{s}")
