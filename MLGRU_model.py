from common import *
from Models.MLGRU import *
from Preprocessing.MLGRU_pre import *

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    pcs, murmurs, outs = zip(*batch)
    pcs = torch.stack(pcs, dim=0)   # [B, L]
    murmurs = torch.stack(murmurs, dim=0)
    outs = torch.stack(outs, dim=0).unsqueeze(1)  # [B,1]
    return pcs, murmurs, outs

def train_one_epoch(model, loader, opt, crit_m, crit_o, device):
    model.train()
    running_loss = 0.0
    n = 0
    
    
    correct_murmur, total_murmur = 0, 0
    correct_outcome, total_outcome = 0, 0

    print("training")
    for batch in loader:
        if batch is None:
            continue

        pcg, murmur_label, outcome_label = [b.to(device) for b in batch]

        opt.zero_grad()
        murmur_logits, outcome_logit, _ = model(pcg)

        
        loss_m = crit_m(murmur_logits, murmur_label)
        loss_o = crit_o(outcome_logit.squeeze(1), outcome_label.squeeze(1))
        loss = loss_m + loss_o

        loss.backward()
        opt.step()

        
        running_loss += loss.item() * pcg.size(0)
        n += pcg.size(0)

        
        murmur_preds = murmur_logits.argmax(dim=1)
        correct_murmur += (murmur_preds == murmur_label).sum().item()
        total_murmur += murmur_label.size(0)

        
        outcome_probs = torch.sigmoid(outcome_logit.squeeze(1))
        outcome_preds = (outcome_probs > 0.5).long()
        correct_outcome += (outcome_preds == outcome_label.long().squeeze(1)).sum().item()
        total_outcome += outcome_label.size(0)

    
    avg_loss = running_loss / max(1, n)
    murmur_acc = correct_murmur / max(1, total_murmur)
    outcome_acc = correct_outcome / max(1, total_outcome)

    return avg_loss, murmur_acc, outcome_acc


def main(data_path, metadata_txt, subset_size=7000):
    print("Loading dataset metadata...")
    ds = PCGDataset(data_path, metadata_txt, seq_len=SEQ_LEN)

    
    if subset_size and subset_size < len(ds):
        indices = list(range(subset_size))
        ds = torch.utils.data.Subset(ds, indices)
        print(f"Using subset of {subset_size} examples")

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = MurmurOutcomeMLGRU_Attn(d=64, seq_len=SEQ_LEN).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    crit_m = nn.CrossEntropyLoss()
    crit_o = nn.BCEWithLogitsLoss()

    best_score = -float("inf")
    best_epoch = -1
    best_model_path = "best_model.pth"

    for epoch in range(EPOCHS):
        loss, acc_murmur, acc_outcome = train_one_epoch(model, loader, opt, crit_m, crit_o, DEVICE)
        avg_acc = (acc_murmur + acc_outcome) / 2

        print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Murmur Acc: {acc_murmur:.4f}, Outcome Acc: {acc_outcome:.4f}")

        if avg_acc > best_score:
            best_score = avg_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved at epoch {epoch+1} with Avg Acc: {avg_acc:.4f}")

    print(f"Training complete. Best model was from epoch {best_epoch} with Avg Acc: {best_score:.4f}")



    
    torch.save(model.state_dict(), "mlgru_pcg_attn.pt")
    print("Saved model -> mlgru_pcg_attn.pt")

    
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            pcg, murmur_label, outcome_label = [b.to(DEVICE) for b in batch]
            murmur_logits, outcome_logit, attn = model(pcg)
            print("murmur_logits.shape:", murmur_logits.shape)
            print("outcome_logit.shape:", outcome_logit.shape)
            print("attn.shape:", attn.shape)
            break

if __name__ == "__main__":
    
    data_path = "/content/training_data"    
    metadata_txt = "/content/training_data" 
    main(data_path, metadata_txt)
