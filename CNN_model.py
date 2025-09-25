from common import *

from Models.CNN import *
from Preprocessing.CNN_pre import *

def collate_fn(batch):
    # Filter out Nones
    batch = [item for item in batch if item is not None]
    if not batch:
        return None  # Handle empty batch
    return torch.utils.data.default_collate(batch)



def train(model, dataloader, optimizer, criterion_murmur, criterion_outcome, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        if batch is None:
            continue
        pcg, spec, meta, outcome_label, murmur_label = batch
        pcg, spec, meta = pcg.to(device), spec.to(device), meta.to(device)
        murmur_label, outcome_label = murmur_label.to(device), outcome_label.to(device)

        optimizer.zero_grad()
        murmur_pred, outcome_pred = model(pcg, spec, meta)
        loss_m = criterion_murmur(murmur_pred, murmur_label)
        loss_o = criterion_outcome(outcome_pred, outcome_label)
        loss = loss_m + loss_o
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


if __name__ == "__main__":
    data_path = "/content/training_data"
    metadata_txt = "/content/training_data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PCGDataset(data_path, metadata_txt)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    model = MurmurOutcomeNet().to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4)
    criterion_murmur = nn.CrossEntropyLoss()
    criterion_outcome = nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(10)):
        loss = train(model, dataloader, optimizer, criterion_murmur, criterion_outcome, device)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

