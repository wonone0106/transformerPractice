import torch
import os
import logging

def train(model, train_loader, valid_loader, criterion, optimizer, start_epoch, num_epochs, device, best_model_weights=None):
    model.to(device)
    best_accuracy = 0.0
    best_model_weights = None
    logging.info(f"Starting training at epoch {start_epoch} for {num_epochs} epochs.")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0

        for src_inputs, tgt_inputs in train_loader:
            src_inputs, tgt_inputs = src_inputs.to(device), tgt_inputs[:,:-1].to(device)
            labels = tgt_inputs[:,1:].to(device)

            optimizer.zero_grad()

            outputs = model(src_inputs, tgt_inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * src_inputs.size(0)

        avg_train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for src_inputs, tgt_inputs in valid_loader:
                src_inputs, tgt_inputs = src_inputs.to(device), tgt_inputs[:,:-1].to(device)
                labels = tgt_inputs[:,1:].to(device)

                outputs = model(src_inputs, tgt_inputs)
                _, predicted = torch.max(outputs, dim=-1)

                total += labels.numel()
                correct += (predicted == labels).sum().item()
                
        accuracy = correct / total
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_weights = model.state_dict().copy()
        
        logging.info("Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}".format(epoch + 1, num_epochs, avg_train_loss, accuracy))
        
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
        torch.save(model.state_dict(), f'best_model_{best_accuracy:.2f}.pth')
        logging.info(f"Best model weights saved with accuracy: {best_accuracy:.2f}")
        os.remove("checkpoint.pth")
    logging.info("Training complete.")