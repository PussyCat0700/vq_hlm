import argparse
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from constants import KEY_LM_HIDDEN_STATES
from dataloading import get_chunked_h5dataloader
import logging
import os
from vq_models import get_model
from utils import load_config
import random
import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

lr = 3e-4
train_epochs = 1
num_codes = 1024
num_quantizers = 1
is_multi_codebook = False
seed = 1234
device = "cuda" if torch.cuda.is_available() else "cpu"


def update_global(args):
    global num_codes, num_quantizers, is_multi_codebook, lr, train_epochs
    model_config = load_config(args.model_config)
    num_codes = model_config.get('codebook_size', num_codes)
    num_quantizers = model_config.get('num_quantizers', num_quantizers)
    is_multi_codebook = num_quantizers > 1
    lr = model_config.get('lr', lr)
    train_epochs = model_config.get('epoch',train_epochs)


def load_checkpoint(model, optimizer, ckpt_path):
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['step']


def prepare_token(model, data_loader, save_dir):
    model.to(device)
    model.eval()
    tokens = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            x = batch[KEY_LM_HIDDEN_STATES].to(device)
            # shape of x: BxTxD
            out, indices, cmt_loss = model(x)
            # shape of out: BxTxD, shape of indices: BxTxN
            codebook_num = indices.shape[-1]
            tokens.append(indices.reshape(-1,codebook_num).cpu())
    return torch.vstack(tokens)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    seed_everything(seed)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", default='conf/data/example.yaml')
    parser.add_argument("--model_config", default='conf/models/vectorquantize.yaml')
    parser.add_argument("--ckpt_dir", default='./checkpoints')
    parser.add_argument("--test", action='store_true')
    args = parser.parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    update_global(args)

    train_dataloader = get_chunked_h5dataloader(config_path=args.data_config, split='train', shuffle=False)
    val_dataloader = get_chunked_h5dataloader(config_path=args.data_config, split='validation', shuffle=False)
    test_dataloader = get_chunked_h5dataloader(config_path=args.data_config, split='test', shuffle=False)

    model = get_model(args.model_config)
    logging.info("Loading best checkpoint for processing")
    load_checkpoint(model, None, os.path.join(args.ckpt_dir, 'best_checkpoint.pt'))
    data_config = config = load_config(args.data_config)
    # save_dir = data_config['h5_file_path']
    save_dir = '/home/yxwang/Dataset/vqhlm/ResidualSimVQ'

    logging.info("Preparing train tokens")
    train_tokens = prepare_token(model, train_dataloader, save_dir)
    torch.save(train_tokens, os.path.join(save_dir, 'train_tokens.pt'))
    logging.info("Preparing validation tokens")
    val_tokens = prepare_token(model, val_dataloader, save_dir)
    torch.save(val_tokens, os.path.join(save_dir, 'val_tokens.pt'))
    logging.info("Preparing test tokens")
    test_tokens = prepare_token(model, test_dataloader, save_dir)
    torch.save(test_tokens, os.path.join(save_dir, 'test_tokens.pt'))
