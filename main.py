import hydra
from omegaconf import OmegaConf
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import torch.nn as nn
from trainer import train
import torch
import os
from transformer import Transformer
from dataPreprocessing import train_data, valid_data, sp, collate_fn

@hydra.main(version_base=None, config_path="./config", config_name="train.yaml")
def main(cfg):
    OmegaConf.to_yaml(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    train_dl = DataLoader(train_data, batch_size=cfg.train.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_dl = DataLoader(valid_data, batch_size=cfg.train.batch_size, shuffle=False, collate_fn=collate_fn)

    model = Transformer(
        src_dim=37000,
        tgt_dim=37000,
        embed_dim=cfg.model.embed_dim,
        n_heads=cfg.model.n_heads,
        num_encoder_layers=cfg.model.num_encoder_layers,
        num_decoder_layers=cfg.model.num_decoder_layers
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=sp.pad_id())
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)
    epoch = 0

    best_model_weights = None
    best_val_loss = float("inf")

    if os.path.exists("checkpoint.pth"):
        checkpoint = torch.load("checkpoint.pth")
        logging.info(f"Loading checkpoint successfully")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_model_weights = checkpoint['best_model_weights']
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        best_val_loss = checkpoint['best_val_loss']
        logging.info(f"Resuming training from epoch {epoch+1} with loss {loss:.4f}")

    train(model, train_dl, valid_dl, criterion, optimizer, epoch, cfg.train.epochs, device, best_model_weights, best_val_loss)

if __name__ == "__main__":
    main()