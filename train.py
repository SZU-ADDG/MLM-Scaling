import torch
from utils.train_utils import seed_all
from torch.distributed import init_process_group, destroy_process_group
import os
import argparse
# from dataset import SmileDataset, SmileCollator
from dataset import SmileDataset, SmileCollator
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from tokenizer import SmilesTokenizer
from model import GPTConfig, GPT
from trainer import TrainerConfig, Trainer
import datasets


def ddp_setup(rank: int, world_size: int):
    #初始化使用nccl后端
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12337"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def main(rank: int, world_size: int, args):
    print(args.run_name)
    #设置随机种子的值
    seed_value = 42
    seed_all(seed_value)
    ddp_setup(rank, world_size)

    device = torch.device(f'cuda:{rank}')  # 逻辑编号 cuda:0 对应 os.environ["CUDA_VISIBLE_DEVICES"]中的第一个gpu
    batch_size = args.batch_size

    train_names = "train"
    val_names = "validation"
    tokenizer = SmilesTokenizer('./vocabs/vocab.txt')
    tokenizer.bos_token = "[BOS]"
    tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("[BOS]")
    tokenizer.eos_token = "[EOS]"
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("[EOS]")
    tokenizer.sep_token = "[SEP]"
    tokenizer.sep_token_id = tokenizer.convert_tokens_to_ids("[SEP]")
    raw_datasets = datasets.load_from_disk(args.dataset_path)
    traindata = SmileDataset(raw_datasets, data_type=train_names, tokenizer=tokenizer)
    validdata = SmileDataset(raw_datasets, data_type=val_names, tokenizer=tokenizer)

    collator = SmileCollator(tokenizer)
    train_dataloader = DataLoader(traindata, batch_size=batch_size, shuffle=False,
                                  sampler=DistributedSampler(traindata), collate_fn=collator, num_workers=5)
    valid_dataloader = DataLoader(validdata, batch_size=batch_size, shuffle=False,
                                  sampler=DistributedSampler(validdata), collate_fn=collator, num_workers=5)

    """ 
    GPT-1 like network roughly 125M params
    n_layer = 12
    n_head = 12
    n_embd = 768
    """
    if args.model_size == '1M':
        mconf = GPTConfig(vocab_size=tokenizer.vocab_size, n_layer=4, n_head=4, n_embd=160)
    elif args.model_size == '4M':
        mconf = GPTConfig(vocab_size=tokenizer.vocab_size, n_layer=5, n_head=4, n_embd=256)
    elif args.model_size == '16M':
        mconf = GPTConfig(vocab_size=tokenizer.vocab_size, n_layer=5, n_head=4, n_embd=512)
    elif args.model_size == '56M':
        mconf = GPTConfig(vocab_size=tokenizer.vocab_size, n_layer=6, n_head=16, n_embd=768)
    elif args.model_size == '85M':
        mconf = GPTConfig(vocab_size=tokenizer.vocab_size, n_layer=12, n_head=12, n_embd=768)
    elif args.model_size == '152M':
        mconf = GPTConfig(vocab_size=tokenizer.vocab_size, n_layer=12, n_head=16, n_embd=1024)
    elif args.model_size == '278M':
        mconf = GPTConfig(vocab_size=tokenizer.vocab_size, n_layer=22, n_head=16, n_embd=1024)
    elif args.model_size == '650M':
        mconf = GPTConfig(vocab_size=tokenizer.vocab_size, n_layer=13, n_head=32, n_embd=2048)
    model = GPT(mconf).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] 参数总量：{total_params:,}")
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=False)
    def parse_total_tokens(run_name: str) -> int:
        if "100M" in run_name: return int(1e8)
        if "300M" in run_name: return int(3e8)
        if "1B"   in run_name: return int(1e9)
        if "3B"   in run_name: 
            return int(3e9)
        else:
            return int(3e10)
        
    total_tokens = parse_total_tokens(args.run_name) * args.epochs
    warmup_tokens = int(0.02 * total_tokens)
    
    lr = 5e-4
    if args.model_size == "320M": lr = 2e-4
    if args.model_size == "650M": lr = 1e-4
    
    tconf = TrainerConfig(
        max_epochs=args.epochs,
        batch_size=batch_size,
        learning_rate=lr,
        lr_decay=True,
        warmup_tokens=warmup_tokens,
        final_tokens=total_tokens,
        ckpt_path=f'./weights/{args.run_name}.pt',
        generate=False
    )

    # tconf = TrainerConfig(max_epochs=args.epochs, batch_size=batch_size, learning_rate=5e-4, lr_decay=True, warmup_tokens=5*1e7,
    #                       final_tokens=4*190e7, ckpt_path=f'./weights/{args.run_name}.pt', generate=False)
    trainer = Trainer(model, train_dataloader, valid_dataloader, tconf, tokenizer, device, rank)
    # wandb.init(project="lig_gpt", name=args.run_name)
    # wandb.init(mode="disabled")
    trainer.train()

    destroy_process_group()


if __name__ == '__main__':
    """
        world_size: 所有的进程数量
        rank: 全局的进程id
    """
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--run_name', default='SMILES-5M-100M-Epoch2', help='name of .pt file')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--model_size', default=None)
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--world_size', default=2, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--dataset_path', type=str, help="path to dataset file.")
    # /share/home/tm866079609100000/a866071650/doomx/MolGen/DATA/SMILES-100M
    opt = parser.parse_args()
    # wandb.init(mode="disabled")
    # wandb.init(project="lig_gpt", name=opt.run_name)
    world_size = opt.world_size
    mp.spawn(main, args=(world_size, opt), nprocs=world_size)
