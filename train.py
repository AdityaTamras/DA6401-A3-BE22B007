import sacrebleu
import sys
import torch
import math
import wandb
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from typing import Optional
from dataset import Multi30kDataset
from lr_scheduler import NoamScheduler
from model import (
    Transformer, EncoderLayer, DecoderLayer,
    Encoder, Decoder, PositionalEncoding,
    MultiHeadAttention, PositionwiseFeedForward,
    make_src_mask, make_tgt_mask,
    scaled_dot_product_attention,
)

def parse_args():
    p=argparse.ArgumentParser(description="DA6401-A3 Transformer Assignment")

    p.add_argument("--experiment", required=True, help="Experiment name or number")
    p.add_argument("--variant", default=None, help="Sub-variant within the experiment")
    p.add_argument("--wandb_project", default="DA6401-A3")
    p.add_argument("--wandb_entity",  default=None)
    p.add_argument("--d_model", type=int,   default=512)
    p.add_argument("--N", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--d_ff", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--min_freq", type=int, default=3)
    p.add_argument("--warmup_steps", type=int, default=4000)
    p.add_argument("--smoothing", type=float, default=0.1)
    p.add_argument("--fixed_lr", type=float, default=1e-4, help="Fixed learning rate for exp-1 'fixed' variant")
    p.add_argument("--log_grad_steps", type=int, default=1000, help="Number of steps to log gradient norms (exp-2)")

    return p.parse_args()

class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, pad_idx, smoothing=0.1):
        super().__init__()
        self.vocab_size=vocab_size
        self.pad_idx=pad_idx
        self.smoothing=smoothing

    def forward(self, logits, target):
        fill_val=self.smoothing/(self.vocab_size-1)
        smooth_dist=torch.full_like(logits, fill_val)
        smooth_dist.scatter_(1, target.unsqueeze(1), 1-self.smoothing)
        smooth_dist[target==self.pad_idx]=0
        loss=F.kl_div(F.log_softmax(logits, dim=-1), smooth_dist, reduction='sum')
        non_pad_count=(target!=self.pad_idx).sum()
        return loss/non_pad_count
    
def run_epoch(data_iter, model, loss_fn, optimizer=None, scheduler=None, epoch_num=0, is_train=True, device='cpu', grad_hook=None, max_grad_steps=None):
    model.train() if is_train==True else model.eval()

    running_loss=0
    token_count=0
    step=0

    context=torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for src, tgt in data_iter:
            src=src.to(device)
            tgt=tgt.to(device)
            src_mask=make_src_mask(src)
            tgt_mask=make_tgt_mask(tgt[:, :-1])
            if is_train:
                optimizer.zero_grad()
            logits=model(src, tgt[:, :-1], src_mask, tgt_mask)
            logits_flat=logits.reshape(-1, logits.size(-1))
            tgt_flat=tgt[:, 1:].reshape(-1)
            loss=loss_fn(logits_flat, tgt_flat)
            running_loss+=loss.item()*(tgt_flat!=loss_fn.pad_idx).sum().item()
            token_count+=(tgt_flat!=loss_fn.pad_idx).sum().item()
            if is_train:
                loss.backward()
                if grad_hook is not None and (max_grad_steps is None or step<max_grad_steps):
                    grad_hook(step, model)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            step+=1
    avg_loss=running_loss/max(token_count, 1)
    return avg_loss

def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device='cpu'):
    memory=model.encode(src, src_mask)
    ys=torch.ones(1, 1).fill_(start_symbol).long().to(device)
    for _ in range(max_len):
        tgt_mask=make_tgt_mask(ys)
        logits=model.decode(memory, src_mask, ys, tgt_mask)
        next_token=logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ys=torch.cat([ys, next_token], dim=1)
        if next_token.item()==end_symbol:
            break
    return ys

def evaluate_bleu(model, test_dataloader, tgt_vocab, device='cpu', max_len=100):
    model.eval()

    idx_to_token={v: k for k, v in tgt_vocab.items()}
    sos_idx=tgt_vocab["<sos>"]
    eos_idx=tgt_vocab["<eos>"]
    pad_idx=tgt_vocab["<pad>"]
    skip_ids={sos_idx, eos_idx, pad_idx}

    def indices_to_str(indices):
        tokens=[]
        for idx in indices:
            if idx in skip_ids:
                continue
            tokens.append(idx_to_token.get(idx, "<unk>"))
        return " ".join(tokens)
    
    hypotheses, references = [], []
    with torch.no_grad():
        for src_batch, tgt_batch in test_dataloader:
            src_batch=src_batch.to(device)
            tgt_batch=tgt_batch.to(device)

            for i in range(src_batch.size(0)):
                src=src_batch[i].unsqueeze(0)
                tgt=tgt_batch[i].unsqueeze(0)

                src_mask=make_src_mask(src)
                output=greedy_decode(model, src, src_mask, max_len, sos_idx, eos_idx, device)
                hypotheses.append(indices_to_str(output.squeeze(0).tolist()))
                references.append(indices_to_str(tgt.squeeze(0).tolist()))

    bleu=sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score

def save_checkpoint(model, optimizer, scheduler, epoch, path="checkpoint.pt"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'model_config': {
            'src_vocab_size': model.src_embedding.num_embeddings,
            'tgt_vocab_size': model.tgt_embedding.num_embeddings,
            'd_model': model.d_model,
            'N': len(model.encoder.layers),
            'num_heads': model.encoder.layers[0].self_attn.num_heads,
            'd_ff': model.encoder.layers[0].positionwise_ff.linear1.out_features,
            'dropout': model.pos_encoding.dropout.p
        }
    }, path)

def load_checkpoint(path, model, optimizer=None, scheduler=None):
    checkpoint=torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and checkpoint.get('optimizer_state_dict'):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch']

def build_data(cfg):

    train_raw=Multi30kDataset(split="train")
    train_raw.build_vocab(min_freq=cfg["min_freq"])
    src_vocab=train_raw.src_vocab
    tgt_vocab=train_raw.tgt_vocab
    src_data, tgt_data = train_raw.process_data()

    test_raw=Multi30kDataset(split="test")
    test_raw.src_vocab=src_vocab
    test_raw.tgt_vocab=tgt_vocab
    test_raw._de_tokenized=test_raw._tokenize(test_raw.spacy_de, test_raw.dataset["de"])
    test_raw._en_tokenized=test_raw._tokenize(test_raw.spacy_en, test_raw.dataset["en"])
    test_src, test_tgt=test_raw.process_data()

    pad_idx=src_vocab["<pad>"]

    def collate_fn(batch):
        src_batch, tgt_batch=zip(*batch)
        max_src=max(len(s) for s in src_batch)
        max_tgt=max(len(t) for t in tgt_batch)
        src_padded=torch.tensor([s+[pad_idx]*(max_src-len(s)) for s in src_batch], dtype=torch.long)
        tgt_padded=torch.tensor([t+[pad_idx]*(max_tgt-len(t)) for t in tgt_batch], dtype=torch.long)
        return src_padded, tgt_padded
    
    full_data=list(zip(src_data, tgt_data))
    val_size=int(0.1*len(full_data))   
    train_size=len(full_data)-val_size
    print(f"Train dataset size : {train_size} | Val dataset size : {val_size} | Test : {len(test_src)}")
    train_data, val_data = random_split(full_data, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    bs=cfg["batch_size"]

    train_loader=DataLoader(train_data, batch_size=bs, shuffle=True, collate_fn=collate_fn)
    val_loader=DataLoader(val_data, batch_size=bs, shuffle=False, collate_fn=collate_fn)
    test_loader=DataLoader(list(zip(test_src, test_tgt)), batch_size=bs, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout=nn.Dropout(p=dropout)
        self.embedding=nn.Embedding(max_len, d_model)

    def forward(self, x):
        positions=torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.dropout(x + self.embedding(positions))
    
def build_transformer(cfg, src_vocab_size, tgt_vocab_size, use_learned_pe=False):
    d_model=cfg["d_model"]
    num_heads=cfg["num_heads"]
    d_ff=cfg["d_ff"]
    dropout=cfg["dropout"]
    N=cfg["N"]

    model=nn.Module.__new__(Transformer)
    nn.Module.__init__(model)

    model.d_model=d_model
    model.src_embedding=nn.Embedding(src_vocab_size, d_model)
    model.tgt_embedding=nn.Embedding(tgt_vocab_size, d_model)

    if use_learned_pe:
        model.pos_encoding=LearnedPositionalEncoding(d_model, dropout)
    else:
        model.pos_encoding=PositionalEncoding(d_model, dropout)

    enc_layer=EncoderLayer(d_model, num_heads, d_ff, dropout)
    model.encoder=Encoder(enc_layer, N)

    dec_layer=DecoderLayer(d_model, num_heads, d_ff, dropout)
    model.decoder=Decoder(dec_layer, N)

    model.linear=nn.Linear(d_model, tgt_vocab_size)
    return model


def _encode(self, src, src_mask):
    x=self.src_embedding(src)*math.sqrt(self.d_model)
    x=self.pos_encoding(x)
    return self.encoder(x, src_mask)


def _decode(self, memory, src_mask, tgt, tgt_mask):
    x=self.tgt_embedding(tgt)*math.sqrt(self.d_model)
    x=self.pos_encoding(x)
    return self.linear(self.decoder(x, memory, src_mask, tgt_mask))


def _forward(self, src, tgt, src_mask, tgt_mask):
    return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)


import types


def attach_forward_methods(model):
    model.encode=types.MethodType(_encode, model)
    model.decode=types.MethodType(_decode, model)
    model.forward=types.MethodType(_forward, model)
    return model


def scaled_dot_product_attention_no_scale(Q, K, V, mask=None):
    raw_scores=torch.matmul(Q, K.transpose(-2, -1))

    if mask is not None:
        raw_scores=raw_scores.masked_fill(mask, -float('inf'))

    attn_weights=F.softmax(raw_scores, dim=-1)
    return torch.matmul(attn_weights, V), attn_weights


def patch_no_scale(model):
    import model as model_module
    original=model_module.scaled_dot_product_attention
    model_module.scaled_dot_product_attention=scaled_dot_product_attention_no_scale
    return original


def restore_scale(original_fn):
    import model as model_module
    model_module.scaled_dot_product_attention=original_fn


def train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    tgt_vocab,
    cfg,
    run_name,
    device,
    use_fixed_lr=False,
    fixed_lr=1e-4,
    grad_log_steps=None,
    log_attention=False,
    train_loader_for_attn=None
):
    pad_idx=tgt_vocab["<pad>"]
    tgt_vocab_sz=model.linear.out_features

    optimizer=torch.optim.Adam(
        model.parameters(),
        lr=fixed_lr if use_fixed_lr else 1.0,
        betas=(0.9, 0.98),
        eps=1e-9
    )

    if use_fixed_lr:
        scheduler=None
    else:
        scheduler=NoamScheduler(
            optimizer,
            d_model=cfg["d_model"],
            warmup_steps=cfg["warmup_steps"]
        )

    loss_fn=LabelSmoothingLoss(
        vocab_size=tgt_vocab_sz,
        pad_idx=pad_idx,
        smoothing=cfg["smoothing"]
    )

    global_step=[0]
    grad_log_done=[False]

    def grad_hook(step, mdl):
        if grad_log_done[0]:
            return

        q_norm=0.0
        k_norm=0.0
        count=0

        for layer in mdl.encoder.layers:
            mha=layer.self_attn

            if mha.W_q.weight.grad is not None:
                q_norm+=mha.W_q.weight.grad.norm().item()
                k_norm+=mha.W_k.weight.grad.norm().item()
                count+=1

        if count:
            wandb.log({
                f"{run_name}/grad_norm_Q": q_norm/count,
                f"{run_name}/grad_norm_K": k_norm/count,
                "global_step": global_step[0]
            })

        global_step[0]+=1

        if global_step[0]>=(grad_log_steps or 0):
            grad_log_done[0]=True

    hook=grad_hook if grad_log_steps else None

    print(f"\n{'='*60}")
    print(f"Run: {run_name}")
    print(f"LR mode: {'fixed='+str(fixed_lr) if use_fixed_lr else 'Noam'}")
    print(f"{'='*60}")

    best_val_loss=float('inf')
    best_ckpt=f"best_{run_name}.pt"

    for epoch in range(cfg["num_epochs"]):
        train_loss=run_epoch(
            train_loader,
            model,
            loss_fn,
            optimizer,
            scheduler,
            epoch_num=epoch,
            is_train=True,
            device=device,
            grad_hook=hook,
            max_grad_steps=grad_log_steps
        )

        val_loss=run_epoch(
            val_loader,
            model,
            loss_fn,
            is_train=False,
            device=device
        )

        current_lr=optimizer.param_groups[0]["lr"]

        confidence=_compute_confidence(model, val_loader, tgt_vocab, device)
        val_accuracy=_compute_val_accuracy(model, val_loader, tgt_vocab, device)

        print(
            f"Epoch {epoch+1:>3} | train_loss={train_loss:.4f} "
            f"| val_loss={val_loss:.4f} | val_acc={val_accuracy:.4f} | lr={current_lr:.7f} "
            f"| confidence={confidence:.4f}"
        )

        wandb.log({
            f"{run_name}/train_loss": train_loss,
            f"{run_name}/val_loss": val_loss,
            f"{run_name}/val_accuracy": val_accuracy,
            f"{run_name}/lr": current_lr,
            f"{run_name}/confidence": confidence,
            "epoch": epoch+1
        })

        if val_loss<best_val_loss:
            best_val_loss=val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, path=best_ckpt)

        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            path=f"ckpt_{run_name}_epoch{epoch+1}.pt"
        )

    if log_attention and train_loader_for_attn is not None:
        _log_attention_maps(
            model,
            train_loader_for_attn,
            tgt_vocab,
            device,
            run_name
        )

    load_checkpoint(best_ckpt, model)

    bleu=evaluate_bleu(
        model,
        test_loader,
        tgt_vocab,
        device=device
    )

    print(f"\n[{run_name}] Test BLEU (best ckpt): {bleu:.2f}")

    wandb.log({
        f"{run_name}/test_bleu": bleu
    })

    return bleu


def _compute_confidence(model, val_loader, tgt_vocab, device):
    model.eval()
    pad_idx=tgt_vocab["<pad>"]

    with torch.no_grad():
        for src, tgt in val_loader:
            src=src.to(device)
            tgt=tgt.to(device)

            src_mask=make_src_mask(src)
            tgt_mask=make_tgt_mask(tgt[:, :-1])

            logits=model(src, tgt[:, :-1], src_mask, tgt_mask)

            probs=F.softmax(logits, dim=-1)

            max_probs=probs.max(dim=-1).values

            tgt_out=tgt[:, 1:]
            non_pad=(tgt_out!=pad_idx)

            confidence=max_probs[non_pad].mean().item()
            return confidence

    return 0.0

def _compute_val_accuracy(model, val_loader, tgt_vocab, device):
    model.eval()

    pad_idx=tgt_vocab["<pad>"]

    correct=0
    total=0

    with torch.no_grad():
        for src, tgt in val_loader:
            src=src.to(device)
            tgt=tgt.to(device)

            src_mask=make_src_mask(src)
            tgt_mask=make_tgt_mask(tgt[:, :-1])

            logits=model(src, tgt[:, :-1], src_mask, tgt_mask)

            preds=logits.argmax(dim=-1)

            tgt_out=tgt[:, 1:]

            non_pad=(tgt_out!=pad_idx)

            correct+=((preds==tgt_out) & non_pad).sum().item()
            total+=non_pad.sum().item()

    return correct/max(total, 1)



def _log_attention_maps(model, loader, tgt_vocab, device, run_name):
    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import numpy as np

    model.eval()

    captured={}

    import model as model_module

    original_sdpa=model_module.scaled_dot_product_attention

    def capturing_sdpa(Q, K, V, mask=None):
        out, w=original_sdpa(Q, K, V, mask)
        captured['weights']=w.detach().cpu()
        return out, w

    model_module.scaled_dot_product_attention=capturing_sdpa

    with torch.no_grad():
        for src, tgt in loader:
            src=src[:1].to(device)
            tgt=tgt[:1].to(device)

            src_mask=make_src_mask(src)
            tgt_mask=make_tgt_mask(tgt[:, :-1])

            _=model(src, tgt[:, :-1], src_mask, tgt_mask)
            break

    model_module.scaled_dot_product_attention=original_sdpa

    if 'weights' not in captured:
        print("[warn] Could not capture attention weights.")
        return

    weights=captured['weights'][0]

    num_heads=weights.size(0)

    fig, axes=plt.subplots(
        2,
        num_heads//2,
        figsize=(3*num_heads//2, 6)
    )

    axes=axes.flatten()

    for h in range(num_heads):
        ax=axes[h]

        ax.imshow(weights[h].numpy(), cmap='viridis', aspect='auto')

        ax.set_title(f"Head {h+1}")
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")

    plt.suptitle("Last Encoder Layer - Per-Head Attention Weights")
    plt.tight_layout()

    wandb.log({
        f"{run_name}/attention_heads": wandb.Image(fig)
    })

    plt.close(fig)

    rollout=weights.mean(dim=0).numpy()

    I=np.eye(rollout.shape[0], rollout.shape[1])

    rollout_r=(rollout+I)/2.0
    rollout_r=rollout_r/rollout_r.sum(axis=-1, keepdims=True)

    fig2, ax2=plt.subplots(figsize=(6, 5))

    ax2.imshow(rollout_r, cmap='Blues', aspect='auto')

    ax2.set_title(
        "Attention Rollout (last encoder layer, head-avg + residual)"
    )

    ax2.set_xlabel("Source token position")
    ax2.set_ylabel("Source token position")

    plt.tight_layout()

    wandb.log({
        f"{run_name}/attention_rollout": wandb.Image(fig2)
    })

    plt.close(fig2)

    print(f"Attention maps logged to W&B under '{run_name}'.")


def experiment_noam_vs_fixed(
    args,
    cfg,
    train_loader,
    val_loader,
    test_loader,
    src_vocab,
    tgt_vocab,
    device
):
    src_sz=len(src_vocab)
    tgt_sz=len(tgt_vocab)

    variants=["noam", "fixed"] if args.variant is None else [args.variant]

    for variant in variants:
        use_fixed=(variant=="fixed")

        run_name=f"exp1_{variant}_lr"

        model=build_transformer(cfg, src_sz, tgt_sz)

        model=attach_forward_methods(model).to(device)

        train_model(
            model,
            train_loader,
            val_loader,
            test_loader,
            tgt_vocab,
            cfg,
            run_name,
            device,
            use_fixed_lr=use_fixed,
            fixed_lr=args.fixed_lr
        )


def experiment_scaling_ablation(
    args,
    cfg,
    train_loader,
    val_loader,
    test_loader,
    src_vocab,
    tgt_vocab,
    device
):
    src_sz=len(src_vocab)
    tgt_sz=len(tgt_vocab)

    variants=["scale", "no_scale"] if args.variant is None else [args.variant]

    for variant in variants:
        import model as model_module

        original_sdpa=model_module.scaled_dot_product_attention

        if variant=="no_scale":
            model_module.scaled_dot_product_attention=scaled_dot_product_attention_no_scale
            print("[exp-2] Running WITHOUT √d_k scaling.")
        else:
            print("[exp-2] Running WITH √d_k scaling (default).")

        run_name=f"exp2_{variant}"

        model=build_transformer(cfg, src_sz, tgt_sz)

        model=attach_forward_methods(model).to(device)

        train_model(
            model,
            train_loader,
            val_loader,
            test_loader,
            tgt_vocab,
            cfg,
            run_name,
            device,
            grad_log_steps=args.log_grad_steps
        )

        model_module.scaled_dot_product_attention=original_sdpa


def experiment_attention_rollout(
    args,
    cfg,
    train_loader,
    val_loader,
    test_loader,
    src_vocab,
    tgt_vocab,
    device
):
    src_sz=len(src_vocab)
    tgt_sz=len(tgt_vocab)

    run_name="exp3_attention_rollout"

    model=build_transformer(cfg, src_sz, tgt_sz)

    model=attach_forward_methods(model).to(device)

    train_model(
        model,
        train_loader,
        val_loader,
        test_loader,
        tgt_vocab,
        cfg,
        run_name,
        device,
        log_attention=True,
        train_loader_for_attn=train_loader
    )


def experiment_pos_enc_ablation(
    args,
    cfg,
    train_loader,
    val_loader,
    test_loader,
    src_vocab,
    tgt_vocab,
    device
):
    src_sz=len(src_vocab)
    tgt_sz=len(tgt_vocab)

    variants=["sinusoidal", "learned"] if args.variant is None else [args.variant]

    for variant in variants:
        use_learned=(variant=="learned")

        run_name=f"exp4_{variant}_pe"

        model=build_transformer(
            cfg,
            src_sz,
            tgt_sz,
            use_learned_pe=use_learned
        )

        model=attach_forward_methods(model).to(device)

        print(f"[exp-4] Positional encoding: {variant}")

        train_model(
            model,
            train_loader,
            val_loader,
            test_loader,
            tgt_vocab,
            cfg,
            run_name,
            device
        )


def experiment_label_smoothing(
    args,
    cfg,
    train_loader,
    val_loader,
    test_loader,
    src_vocab,
    tgt_vocab,
    device
):
    src_sz=len(src_vocab)
    tgt_sz=len(tgt_vocab)

    if args.variant is None:
        runs=[("smooth", 0.1), ("no_smooth", 0.0)]

    elif args.variant=="smooth":
        runs=[("smooth", args.smoothing)]

    else:
        runs=[("no_smooth", 0.0)]

    for variant, eps in runs:
        run_name=f"exp5_{variant}"

        this_cfg=dict(cfg)

        this_cfg["smoothing"]=eps

        print(f"[exp-5] Label smoothing ε = {eps}")

        model=build_transformer(this_cfg, src_sz, tgt_sz)

        model=attach_forward_methods(model).to(device)

        train_model(
            model,
            train_loader,
            val_loader,
            test_loader,
            tgt_vocab,
            this_cfg,
            run_name,
            device
        )


EXPERIMENT_MAP={
    "1": experiment_noam_vs_fixed,
    "noam_vs_fixed": experiment_noam_vs_fixed,
    "2": experiment_scaling_ablation,
    "scaling_ablation": experiment_scaling_ablation,
    "3": experiment_attention_rollout,
    "attention_rollout": experiment_attention_rollout,
    "4": experiment_pos_enc_ablation,
    "pos_enc_ablation": experiment_pos_enc_ablation,
    "5": experiment_label_smoothing,
    "label_smoothing": experiment_label_smoothing
}


def main():
    args=parse_args()

    exp_key=args.experiment.strip().lower()

    if exp_key not in EXPERIMENT_MAP:
        valid=", ".join(sorted(EXPERIMENT_MAP.keys()))

        sys.exit(
            f"Unknown experiment '{args.experiment}'. "
            f"Valid choices: {valid}"
        )

    cfg={
        "d_model": args.d_model,
        "N": args.N,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "dropout": args.dropout,
        "warmup_steps": args.warmup_steps,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "min_freq": args.min_freq,
        "smoothing": args.smoothing
    }

    device="cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    print(
        f"Experiment: {args.experiment} | "
        f"Variant: {args.variant or 'all'}"
    )

    print(f"Config: {cfg}\n")

    run_display_name=f"{args.experiment}" + (
        f"_{args.variant}" if args.variant else ""
    )

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_display_name,
        config={
            **cfg,
            "experiment": args.experiment,
            "variant": args.variant,
            "device": device
        }
    )

    train_loader, val_loader, test_loader, src_vocab, tgt_vocab=build_data(cfg)

    EXPERIMENT_MAP[exp_key](
        args,
        cfg,
        train_loader,
        val_loader,
        test_loader,
        src_vocab,
        tgt_vocab,
        device
    )

    wandb.finish()


if __name__=="__main__":
    main()



    




        


