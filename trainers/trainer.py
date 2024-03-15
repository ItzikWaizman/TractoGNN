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
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.graph = self.data_handler.graph.to(self.device)

    def calc_loss(self, outputs, labels, lengths):
        # Need to use mask to mask out the padded values
        # apply loss mask
        #
        self.criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        pass

    def train_epoch(self, data_loader):
        self.model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        for node_sequences_batch, labels, lengths in data_loader:
            labels = labels.to(self.device)
            node_sequences_batch = node_sequences_batch.to(self.device)

            # Forward pass
            outputs = self.model(self.graph, node_sequences_batch, lengths)
            loss = self.calc_loss(outputs, labels)

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
            for node_sequences_batch, labels, lengths in data_loader:
                labels = labels.to(self.device)
                node_sequences_batch = node_sequences_batch.to(self.device)

                # Forward pass
                outputs = self.model(self.graph, node_sequences_batch, lengths)
                loss = self.calc_loss(outputs, labels)

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



  