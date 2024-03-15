import torch
import torch.nn as nn
from torch.optim import Adam
from models.network import TractoGNN
from data_handling import SubjectDataHandler

class TractoGNNTrainer(object):
    def __init__(self, logger, params):
        self.device = params['device']
        self.network = TractoGNN(logger=logger, params=params).to(self.device)
        self.data_handler = SubjectDataHandler(logger=logger, params=params) 
        self.optimizer = Adam(self.network.parameters(), lr=params['learning_rate'])
        self.num_epochs = params['epochs']
        self.criterion = nn.KLDivLoss(reduction='none')
        self.graph = self.data_handler.graph.to(self.device)

    def calculate_loss(self, outputs, labels, padding_mask):
        """
        Calculate the masked loss using KLDivLoss for sequences with padding.

        Parameters:
        - outputs (Tensor): Log probabilities of shape [batch_size, seq_length, 730].
        - labels (Tensor): True probabilities with shape [batch_size, seq_length, 730].
        - padding_mask (Tensor): A boolean tensor of shape [batch_size, seq_length] where True
        indicates valid points and False indicates padded points.

        Returns:
        - loss (Tensor): Scalar tensor representing the average loss over all valid points.
        """

        # Calculate the element-wise loss
        elementwise_loss = self.criterion(outputs, labels)

        # Apply the padding mask to ignore loss for padded values
        # We need to unsqueeze the padding_mask to make it broadcastable to elementwise_loss shape
        masked_loss = elementwise_loss * padding_mask.unsqueeze(-1)
        
        # Calculate the average loss per valid sequence element
        loss = masked_loss.sum() / padding_mask.sum()

        return loss
        
    def train_epoch(self, data_loader):
        self.model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        for node_sequences_batch, labels, lengths, padding_mask in data_loader:
            labels = labels.to(self.device)
            node_sequences_batch = node_sequences_batch.to(self.device)

            # Forward pass
            outputs = self.model(self.graph, node_sequences_batch, lengths, padding_mask, self.data_handler.casuality_mask)
            loss = self.calc_loss(outputs, labels, ~padding_mask)

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * node_sequences_batch.size(0)
            total_correct += (outputs.argmax(dim=-1) == labels).sum().item()
            total_samples += node_sequences_batch.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

    def validate(self, data_loader):
        self.model.eval()
        total_loss, total_correct, total_samples = 0, 0, 0
        with torch.no_grad():
            for node_sequences_batch, labels, lengths, padding_mask in data_loader:
                labels = labels.to(self.device)
                node_sequences_batch = node_sequences_batch.to(self.device)

                # Forward pass
                outputs = self.model(self.graph, node_sequences_batch, lengths, padding_mask, self.data_handler.casuality_mask)
                loss = self.calc_loss(outputs, labels, ~padding_mask)

                total_loss += loss.item() * node_sequences_batch.size(0)
                total_correct += (outputs.argmax(dim=-1) == labels).sum().item()
                total_samples += node_sequences_batch.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

    def train(self):
        train_stats, val_stats = [], []
        for epoch in range(self.num_epochs):
            train_loss, train_accuracy = self.train_epoch(self.data_handler.train_loader)
            val_loss, val_accuracy = self.validate(self.data_handler.val_loader)

            train_stats.append((train_loss, train_accuracy))
            val_stats.append((val_loss, val_accuracy))

            print(f'Epoch {epoch + 1}/{self.num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
            
        return train_stats, val_stats



  