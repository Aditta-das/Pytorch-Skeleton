from config import config
from tqdm import tqdm
from utils import loss_calculation
import torch

def train(model, epoch, data_loader, optimizer, device):
    model.train()
    loop = tqdm(data_loader)
    final_loss = 0
    for batch_idx, d in enumerate(loop):
        data = d["data"].to(device=config.DEVICE)
        target = d["label"].to(device=config.DEVICE)

        optimizer.zero_grad()

        output = model(data) # edit this
        loss = loss_calculation(output, target.unsqueeze(1))
        loss.backward()

        optimizer.step()
        final_loss += loss.item()
        loop.set_description(f"EPOCH: {epoch} | ITERATION : {batch_idx}/{len(data_loader)} | LOSS: {loss.item()}")
        loop.set_postfix(loss=loss.item())
    return final_loss / len(data_loader)

def test(model, data_loader, device):
    test_loss = 0
    model.eval()
    loop = tqdm(data_loader)
    with torch.no_grad():
        for batch_idx, data in enumerate(loop):
            data = data["data"].to(device=config.DEVICE)
            target = data["label"].to(device=config.DEVICE)
        
            output = model(data) # edit this
            _, predictions = torch.max(output, 1)
            loss = loss_calculation(output, target.unsqueeze(1))
            test_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        return test_loss / len(data_loader)