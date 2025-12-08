import hydra
from omegaconf import OmegaConf
import torch.optim as optim
import logging
import torch.nn as nn
from trainer import train
import torch
import os
from transformer import Transformer
from dataPreprocessing import train_dl, valid_dl, en_vocab, ko_vocab

@hydra.main(version_base=None, config_path="./config", config_name="train.yaml")
def main(cfg):
    OmegaConf.to_yaml(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model = Transformer(
        src_dim=len(ko_vocab),
        tgt_dim=len(en_vocab),
        embed_dim=cfg.model.embed_dim,
        n_heads=cfg.model.n_heads,
        num_encoder_layers=cfg.model.num_encoder_layers,
        num_decoder_layers=cfg.model.num_decoder_layers
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=en_vocab["<pad>"])
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)
    epoch = 0

    best_model_weights = None

    if os.path.exists("checkpoint.pth"):
        checkpoint = torch.load("checkpoint.pth")
        logging.info(f"Loading checkpoint successfully")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_model_weights = checkpoint['best_model_weights']
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        logging.info(f"Resuming training from epoch {epoch+1} with loss {loss:.4f}")

    train(model, train_dl, valid_dl, criterion, optimizer, epoch, cfg.train.epochs, device, best_model_weights)

if __name__ == "__main__":
    main()