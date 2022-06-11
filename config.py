import torch

class config:
    FOLDS = 5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
    SAVE_MODEL = "model.bin"
    EPOCHS = 100
    # SCHEDULER = NONE
    LR = 0.01
    TRAIN_BATCH_SIZE = 32
    TEST_BATCH_SIZE = 16
    PATIENCE = 5
    LOAD_MODEL = False
    