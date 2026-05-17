import sacrebleu
import torch
import wandb
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from typing import Optional
from dataset import Multi30kDataset
from lr_scheduler import NoamScheduler
from model import Transformer, make_src_mask, make_tgt_mask

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
    
def run_epoch(data_iter, model, loss_fn, optimizer=None, scheduler=None, epoch_num=0, is_train=True, device='cpu'):
    model.train() if is_train==True else model.eval()

    running_loss=0
    batch_acc=0
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
        running_loss+=loss
        batch_acc+=logits.size(0)*logits.size(1)
        if is_train:
            loss.backward()
            optimizer.step()
            scheduler.step()
    avg_loss=running_loss/batch_acc
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

    hypotheses=[]
    references=[]
    sos_idx=tgt_vocab["<sos>"]
    eos_idx=tgt_vocab["<eos>"]
    pad_idx=tgt_vocab["<pad>"]
    skip_ids={sos_idx, eos_idx, pad_idx}

    def indices_to_str(indices):
        tokens=[]
        for idx in indices:
            if idx in skip_ids:
                continue
            idx_to_token={v:k for k,v in tgt_vocab.items()}
            tokens.append(idx_to_token.get(idx, "<unk>"))
        return " ".join(tokens)
    
    with torch.no_grad():
        for src_batch, tgt_batch in test_dataloader:
            src_batch=src_batch.to(device)
            tgt_batch=tgt_batch.to(device)

            for i in range(src_batch.size(0)):
                src=src_batch[i].unsqueeze(0)
                tgt=tgt_batch[i].unsqueeze(0)

                src_mask=make_src_mask(src)
                output=greedy_decode(model, src, src_mask, max_len, sos_idx, eos_idx, device)
                pred_indices=output.squeeze(0).tolist()
                hypothesis=indices_to_str(pred_indices)

                ref_indices=tgt.squeeze(0).tolist()
                reference=indices_to_str(ref_indices)

                hypotheses.append(hypothesis)
                references.append(reference)
            
    bleu=sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score

def save_checkpoint(model, optimizer, scheduler, epoch, path="checkpoint.pt"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'model_config': {
            'src_vocab_size': model.src_embedding.num_embeddings,
            'tgt_vocab_size': model.tgt_embedding.num_embeddings,
            'd_model': model.d_model,
            'N': len(model.encoder.layers),
            'num_heads': model.encoder.layers[0].self_attn.num_heads,
            'd_ff': model.encoder.layers[0].positionwise_ff.linear1.out_features,
            'dropout': model.pos_encoding.dropout.p
        }
    })

def load_checkpoint(path, model, optimizer, scheduler):
    checkpoint=torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch']

def run_training_experiment():
    config={
       "d_model": 512,
        "N": 6,
        "num_heads": 8,
        "d_ff": 2048,
        "dropout": 0.1,
        "warmup_steps": 4000,
        "num_epochs": 10,
        "batch_size": 32,
        "min_freq": 3,
        "smoothing": 0.1 
    }

    # wandb.init(project="da6401-a3", config=config)
    # cfg=wandb.config

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_dataset_raw=Multi30kDataset(split="train")
    train_dataset_raw.build_vocab(min_freq=config["min_freq"])
    src_vocab=train_dataset_raw.src_vocab
    tgt_vocab=train_dataset_raw.tgt_vocab
    src_data, tgt_data=train_dataset_raw.process_data()

    test_dataset_raw = Multi30kDataset(split="test")
    test_dataset_raw.src_vocab=src_vocab
    test_dataset_raw.tgt_vocab=tgt_vocab
    test_dataset_raw._de_tokenized=test_dataset_raw._tokenize(
        test_dataset_raw.spacy_de, test_dataset_raw.dataset["de"]
    )
    test_dataset_raw._en_tokenized=test_dataset_raw._tokenize(
        test_dataset_raw.spacy_en, test_dataset_raw.dataset["en"]
    )
    test_src_data, test_tgt_data=test_dataset_raw.process_data()

    pad_idx=src_vocab["<pad>"]

    def collate_fn(batch):
        src_batch, tgt_batch = zip(*batch)
        src_lens = [len(s) for s in src_batch]
        tgt_lens = [len(t) for t in tgt_batch]
        max_src=max(src_lens)
        max_tgt=max(tgt_lens)
        src_padded=torch.tensor(
            [s+[pad_idx]*(max_src-len(s)) for s in src_batch], dtype=torch.long
        )
        tgt_padded=torch.tensor(
            [t+[pad_idx]*(max_tgt-len(t)) for t in tgt_batch], dtype=torch.long
        )
        return src_padded, tgt_padded
    
    full_data=list(zip(src_data, tgt_data))
    val_size=int(0.1*len(full_data))   
    train_size=len(full_data)-val_size
    print(f"Train dataset size : {train_size} | Val dataset size : {val_size}")
    train_data, val_data = random_split(full_data, [train_size, val_size])

    train_loader=DataLoader(train_data, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_loader=DataLoader(val_data, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)
    test_loader=DataLoader(list(zip(test_src_data, test_tgt_data)), batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)

    src_vocab_size=len(src_vocab)
    tgt_vocab_size=len(tgt_vocab)

    model=Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=config["d_model"],
        N=config["N"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"]
    ).to(device)

    optimizer=torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler=NoamScheduler(optimizer, d_model=config["d_model"], warmup_steps=config["warmup_steps"])
    loss_fn=LabelSmoothingLoss(vocab_size=tgt_vocab_size, pad_idx=pad_idx, smoothing=config["smoothing"])

    print("Loaded model, optimizer, scheduler and loss function. Starting training.....")
    for epoch in range(config["num_epochs"]):
        train_loss=run_epoch(train_loader, model, loss_fn, optimizer, scheduler, epoch, is_train=True, device=device)
        val_loss=run_epoch(val_loader, model, loss_fn, None, None, epoch, is_train=False, device=device)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | lr: {current_lr:.6f}")

        # wandb.log({
        #     "epoch": epoch+1,
        #     "train_loss": train_loss,
        #     "val_loss": val_loss,
        #     "lr": current_lr
        #     })
    
        save_checkpoint(model, optimizer, scheduler, epoch, path=f"checkpoint_epoch{epoch+1}.pt")
    
    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device)
    print(f"Test BLEU: {bleu:.2f}")
    wandb.log({"test_bleu": bleu})
    wandb.finish()

if __name__=="__main__":
    run_training_experiment()


        
        
        



    




        


