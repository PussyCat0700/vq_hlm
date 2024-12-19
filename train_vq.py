import argparse
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from constants import KEY_LM_HIDDEN_STATES
from dataloading import get_chunked_h5dataloader
import logging
import os
from models import get_model
from utils import load_config
import random
import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

lr = 3e-4
train_epochs = 10
num_codes = 1024
num_quantizers = 1
is_multi_codebook = False
seed = 1234
device = "cuda" if torch.cuda.is_available() else "cpu"


def update_global(args):
    global num_codes, num_quantizers, is_multi_codebook
    data_config = load_config(args.model_config)
    num_codes = data_config.get('codebook_size', 1024)
    num_quantizers = data_config.get('num_quantizers', 1)
    is_multi_codebook = num_quantizers > 1


def save_checkpoint(model, optimizer, step, ckpt_path):
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, ckpt_path)

def load_checkpoint(model, optimizer, ckpt_path):
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['step']

def compute_loss(model, x, alpha=10):
    out, indices, cmt_loss = model(x)
    rec_loss = (out - x).abs().mean()
    cmt_loss = cmt_loss.mean()
    total_loss = rec_loss + alpha * cmt_loss
    return rec_loss, cmt_loss, total_loss, indices

def evaluate(model, eval_loader, split: str, writer: SummaryWriter = None, step: int = None):
    global num_quantizers
    model.to(device)
    model.eval()
    eval_rec_loss = 0
    index_counts = {i: torch.zeros(num_codes).int().to(device) for i in range(num_quantizers)}

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc=f"Running on {split}"):
            x = batch[KEY_LM_HIDDEN_STATES].to(device)
            rec_loss, cmt_loss, total_loss, indices = compute_loss(model, x)
            eval_rec_loss += rec_loss.item()
            for codebook_idx in range(num_quantizers):
                sub_indices = indices[..., codebook_idx] if num_quantizers > 1 else indices  # [B, T]
                for indice in sub_indices:
                    frequency = torch.bincount(indice.flatten(), minlength=num_codes)
                    index_counts[codebook_idx] += frequency

    eval_rec_loss /= len(eval_loader)
    individual_utilizations = []
    for codebook_idx in range(num_quantizers):
        utilized_indices_in_codebook = torch.count_nonzero(index_counts[codebook_idx]).item()
        individual_utilizations.append(utilized_indices_in_codebook / num_codes * 100)

    logging.info(f"{split} Reconstruction Loss: {eval_rec_loss:.4f}")
    for codebook_idx in range(num_quantizers):
        logging.info(f'{split} Active Percentage (Codebook {codebook_idx+1}): {individual_utilizations[codebook_idx]:.4f}')

    if writer:
        writer.add_scalar(f'Loss/{split}', eval_rec_loss, step)
        for codebook_idx in range(num_quantizers):
            writer.add_scalar(f'Active_Codebook_{codebook_idx+1}/{split}', individual_utilizations[codebook_idx], step)
    return eval_rec_loss, index_counts

def save_histogram(args, index_counts):
    global num_quantizers
    import csv
    import matplotlib.pyplot as plt
    for codebook_idx, index_count in enumerate(index_counts):
        filename = f'index_frequencies_{codebook_idx}'
        index_count = index_counts[codebook_idx].tolist()
        with open(os.path.join(args.ckpt_dir, f'{filename}.csv'), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Index', 'Frequency'])  # 写入表头
            for idx, count in enumerate(index_count):
                writer.writerow([idx, count])  # 写入每个索引的频次

        logging.info(f"Index frequencies saved to '{filename}.csv'")

        plt.bar(range(num_codes), index_count, edgecolor='black', alpha=0.7)
        plt.title("Frequency of Codebook Indices (Entire Dataset)")
        plt.xlabel("Codebook Index")
        plt.ylabel("Frequency")
        plt.xticks(range(0, num_codes, 50))
        plt.savefig(os.path.join(args.ckpt_dir, f'{filename}.png'))


def train(model, args, train_loader, val_loader=None, train_epochs=1, alpha=10, validate_every=1000, writer=None, resume_from_step=0):
    model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    step = resume_from_step

    # Load checkpoint if resuming
    if resume_from_step > 0:
        ckpt_path = os.path.join(args.ckpt_dir, 'latest_checkpoint.pt')
        step = load_checkpoint(model, opt, ckpt_path)
        logging.info(f"Resumed from step {resume_from_step}")
    
    for epoch in range(train_epochs):
        logging.info(f"Starting epoch {epoch}")
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            opt.zero_grad()
            x = batch[KEY_LM_HIDDEN_STATES].to(device)
            rec_loss, cmt_loss, total_loss, indices = compute_loss(model, x, alpha)
            total_loss.backward()

            opt.step()
            active_percent = indices.unique().numel() / num_codes * 100
            pbar.set_postfix(
                rec_loss=f"{rec_loss.item():.3f}",
                cmt_loss=f"{cmt_loss.item():.3f}",
                active=f"{active_percent:.1f}%"
            )

            if writer:
                writer.add_scalar('Loss/Train', total_loss.item(), step)
                writer.add_scalar('Active/Train', active_percent, step)
            step += 1

            if val_loader and step % validate_every == 0:
                val_loss, _ = evaluate(model, val_loader, "Validation", writer, step)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, opt, step, os.path.join(args.ckpt_dir, 'best_checkpoint.pt'))

        save_checkpoint(model, opt, step, os.path.join(args.ckpt_dir, 'latest_checkpoint.pt'))

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

    train_dataloader = get_chunked_h5dataloader(config_path=args.data_config, split='train')
    val_dataloader = get_chunked_h5dataloader(config_path=args.data_config, split='validation')
    test_dataloader = get_chunked_h5dataloader(config_path=args.data_config, split='test')

    model = get_model(args.model_config)

    writer = SummaryWriter(log_dir=os.path.join(args.ckpt_dir, 'logs'))
    if not args.test:
        train(model, args, train_dataloader, val_dataloader, train_epochs=train_epochs, writer=writer)

    # Test using best checkpoint
    logging.info("Loading best checkpoint for testing")
    load_checkpoint(model, None, os.path.join(args.ckpt_dir, 'best_checkpoint.pt'))
    _, index_count = evaluate(model, test_dataloader, "Test", writer)
    save_histogram(args, index_count)

    writer.close()
