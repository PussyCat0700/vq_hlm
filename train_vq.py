import argparse
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from constants import KEY_LM_HIDDEN_STATES, \
    KEY_EVAL_REC_LOSS, KEY_EVAL_INDEX_COUNTS, KEY_EVAL_UTIL_LIST
import csv
import matplotlib.pyplot as plt
from dataloading import get_chunked_h5dataloader
import logging
import os
from vq_models import get_model
from utils import load_config, seed_everything


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

lr = 3e-4
max_train_epochs = 1
num_codes = 1024
num_quantizers = 1
is_multi_codebook = False
seed = 1234
device = "cuda" if torch.cuda.is_available() else "cpu"
criterion = torch.nn.MSELoss()


def update_global(args):
    global num_codes, num_quantizers, is_multi_codebook, lr, max_train_epochs
    model_config = load_config(args.model_config)
    num_codes = model_config.get('codebook_size', num_codes)
    num_quantizers = model_config.get('num_quantizers', num_quantizers)
    is_multi_codebook = num_quantizers > 1
    lr = model_config.get('lr', args.lr)
    max_train_epochs = model_config.get('epoch', max_train_epochs)
    return max_train_epochs > 1


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
    rec_loss = criterion(out, x)
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
    return {
        KEY_EVAL_REC_LOSS: eval_rec_loss,
        KEY_EVAL_INDEX_COUNTS: index_counts,
        KEY_EVAL_UTIL_LIST: individual_utilizations,
    }

def save_histogram(args, eval_ret):
    index_counts = eval_ret[KEY_EVAL_INDEX_COUNTS]
    utilizations = eval_ret[KEY_EVAL_UTIL_LIST]
    plt.bar(range(len(utilizations)), utilizations, edgecolor='black', alpha=0.7)
    plt.title("Utilization rate of All Codebooks (Entire Dataset)")
    plt.xlabel("Codebook Layer")
    plt.savefig(os.path.join(args.ckpt_dir, f'codebook_utilization.png'))
    global num_quantizers
    codebooks_info_dir = os.path.join(args.ckpt_dir, 'codebooks')
    os.makedirs(codebooks_info_dir, exist_ok=True)
    for codebook_idx, index_count in enumerate(index_counts):
        filename = f'index_frequencies_{codebook_idx}'
        index_count = index_counts[codebook_idx].tolist()
        with open(os.path.join(codebooks_info_dir, f'{filename}.csv'), mode='w', newline='') as file:
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
        plt.savefig(os.path.join(codebooks_info_dir, f'{filename}.png'))


def train(model, args, train_loader, val_loader=None, max_train_epochs=1, alpha=10, validate_every=1000, writer=None):
    model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_val_loss = float('inf')
    best_epoch = 0
    step = 0
    epoch = start_epoch = 0
    potential_resume_path = os.path.join(args.ckpt_dir, 'latest_checkpoint.pt')
    no_improvement_counter = 0
    should_halt = False
    if args.scheduler:
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(opt, 'min', patience=0)

    # Load checkpoint if resuming
    if os.path.isfile(potential_resume_path):
        step = load_checkpoint(model, opt, potential_resume_path)
        start_epoch = step // len(train_loader)
        logging.info(f"Resumed from step {step}")
    
    for epoch in range(start_epoch, max_train_epochs):
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
                val_loss = evaluate(model, val_loader, "Validation", writer, step)[KEY_EVAL_REC_LOSS]
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    no_improvement_counter = 0
                    save_checkpoint(model, opt, step, os.path.join(args.ckpt_dir, 'best_checkpoint.pt'))
                else:
                    no_improvement_counter += 1
                    if no_improvement_counter >= args.patience and args.patience > 0:
                        should_halt = True
                        break
                if args.scheduler:
                    scheduler.step(val_loss)
                    curr_lr_groups = scheduler.get_last_lr()
                    writer.add_scalar('LR/Train', curr_lr_groups[0], step)
        save_checkpoint(model, opt, step, os.path.join(args.ckpt_dir, 'latest_checkpoint.pt'))
        if should_halt:
            break
    print(f'Stopped on {epoch=}')
    print(f'{best_epoch=}')

if __name__ == '__main__':
    seed_everything(seed)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", default='conf/data/example.yaml')
    parser.add_argument("--model_config", default='conf/models/vectorquantize.yaml')
    parser.add_argument("--ckpt_dir", default='./checkpoints')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--patience", type=int, default=0,
                        help='setting patience>0 will enable early stopping.')
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--scheduler", action='store_true', 
                        help='ReduceLRonPlateau')
    args = parser.parse_args()
    print(f"checkpoint dir: {args.ckpt_dir}")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    is_max_epochs_set = update_global(args)

    if is_max_epochs_set:
        # series full run mode with pre-set epochs
        print(f'Full mode enabled with {max_train_epochs} epochs.')
    elif args.patience > 0:
        # infinite epochs until patience runs out
        max_train_epochs = 10001 # infinite
        print(f'Full mode enabled with early stopping patience {args.patience}.')
    else:
        # toy setting for exploration
        max_train_epochs = 1
        print(f"Training {max_train_epochs} epochs for toy setting.")
    train_dataloader = get_chunked_h5dataloader(config_path=args.data_config, split='train')
    val_dataloader = get_chunked_h5dataloader(config_path=args.data_config, split='validation')
    test_dataloader = get_chunked_h5dataloader(config_path=args.data_config, split='test')

    model = get_model(args.model_config)

    writer = SummaryWriter(log_dir=os.path.join(args.ckpt_dir, 'logs'))
    if not args.test:
        train(model, args, train_dataloader, val_dataloader, max_train_epochs=max_train_epochs, writer=writer)

    # Test using best checkpoint
    logging.info("Loading best checkpoint for testing")
    load_checkpoint(model, None, os.path.join(args.ckpt_dir, 'best_checkpoint.pt'))
    eval_ret = evaluate(model, test_dataloader, "Test", writer)
    save_histogram(args, eval_ret)

    writer.close()
