import torch
from config import config
import os
from dataset import TriggerDataset
from torch.utils.data import DataLoader
from utils import optimizer_func, scheduler_func, save_model, load_checkpoint, plot_graph
from model import WakeupModel
from train_test import train, test

def main():
    train_dataset = TriggerDataset(
        
    )
    test_dataset = TriggerDataset(
        
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.TEST_BATCH_SIZE
    )
    model = WakeupModel()
    # print(model)
    model.to(device=config.DEVICE)
    optimizer = optimizer_func(model)
    scheduler = scheduler_func(optimizer)

    best_train_acc, best_epoch = 0, 0
    avg_train_losses, avg_test_losses = [], []

    if config.LOAD_MODEL:
        filename = os.listdir("/media/aditta/UBUNTU 20_0/WakeMeUp/wakeword/save_model")[-1]
        load_checkpoint(
            torch.load(filename)
        )

    for epoch in range(config.EPOCHS):
        avg_train_loss, avg_train_acc = train(
            model, epoch, train_loader, optimizer, device=config.DEVICE
        )
        
        avg_test_loss = test(
            model, test_loader, optimizer, device=config.DEVICE
        )

        avg_train_losses.append(avg_train_loss.item())
        avg_test_losses.append(avg_test_loss.item())

        scheduler.step(avg_test_loss)
        if avg_train_acc > best_train_acc:
            best_train_acc = avg_train_acc
            best_epoch = epoch
            filename = f"/media/aditta/UBUNTU 20_0/WakeMeUp/wakeword/save_model/best_model_at_epoch_{best_epoch}.bin"
            checkpoint = {
                "state_dict": model.state_dict(), 
                "optimizer": optimizer.state_dict()
            }
            # save model
            save_model(checkpoint, filename)
    # plot graph 
    plot_graph(avg_train_losses, avg_test_losses)

if __name__ == "__main__":
    main()