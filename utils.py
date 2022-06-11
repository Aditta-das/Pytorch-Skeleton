from config import config
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model import WakeupModel
import joblib

def loss_calculation(output, target):
    loss = nn.BCEWithLogitsLoss(reduction="mean")(output, target)
    return loss

def optimizer_func(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR)
    return optimizer

def scheduler_func(optimizer):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=config.PATIENCE)
    return scheduler

def save_model(state, filename):
    print("=> Save model")
    torch.save(state, filename)

def save_encoder(enc_var, filename="enc_save.joblib"):
    print("=> Save Encoder")
    joblib.dump(enc_var, filename, compress=3)

def load_checkpoint(checkpoint):
    print("=> Load Checkpoint")
    model = WakeupModel().to(config.DEVICE)
    optimizer = optimizer_func(model)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def plot_graph(train_losses, test_losses):
    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Loss")
    plt.plot(test_losses,label="val")
    plt.plot(train_losses,label="train")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()