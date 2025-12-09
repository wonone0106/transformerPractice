import torch
import os
import logging

def train(model, train_loader, valid_loader, criterion, optimizer, start_epoch, num_epochs, device, best_model_weights=None):
    model.to(device)
    best_val_loss = float("inf")

    logging.info(f"Starting training at epoch {start_epoch} for {num_epochs} epochs.")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0

        for data in train_loader:
            src_inputs, tgt_inputs = data["src"].to(device), data["tgt"].to(device)
            labels = data["label"].to(device)

            optimizer.zero_grad()

            outputs = model(src_inputs, tgt_inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * src_inputs.size(0)

        avg_train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for data in valid_loader:
                src_inputs, tgt_inputs = data["src"].to(device), data["tgt"].to(device)
                labels = data["label"].to(device)

                outputs = model(src_inputs, tgt_inputs)
                loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                
                val_loss += loss.item() * src_inputs.size(0)

        avg_val_loss = val_loss / len(valid_loader.dataset)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_weights = model.state_dict().copy()
        
        logging.info("Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}".format(epoch + 1, num_epochs, avg_train_loss, avg_val_loss))
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss,
            'best_model_weights': best_model_weights
        }, "checkpoint.pth")
        
        
        logging.info("Saved checkpoint at {}".format(epoch + 1))

    if best_model_weights:
        model.load_state_dict(best_model_weights)
        torch.save(model.state_dict(), f'best_model_{best_val_loss:.2f}.pth')
        logging.info(f"Best model weights saved with Val Loss: {best_val_loss:.2f}")
        os.remove("checkpoint.pth")
    logging.info("Training complete.")