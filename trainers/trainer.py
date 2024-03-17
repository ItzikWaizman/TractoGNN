import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from models.network import TractoGNN
from data_handling import SubjectDataHandler


class TractoGNNTrainer(object):
    def __init__(self, logger, params):
        logger.info("Create TractoGNNTrainer object")
        self.logger = logger
        self.device = params['device']
        self.network = TractoGNN(logger=logger, params=params).to(self.device)
        self.train_data_handler = SubjectDataHandler(logger=logger, params=params, train=True)
        self.val_data_handler = SubjectDataHandler(logger=logger, params=params, train=False)
        self.optimizer = Adam(self.network.parameters(), lr=params['learning_rate'])
        self.num_epochs = params['epochs']
        self.criterion = nn.KLDivLoss(reduction='none')
        self.train_graph = self.train_data_handler.graph.to(self.device)
        self.val_graph = self.val_data_handler.graph.to(self.device)
        self.train_casuality_mask = self.train_data_handler.casuality_mask.to(self.device)
        self.val_casuality_mask = self.val_data_handler.casuality_mask.to(self.device)
        self.params = params

    def calc_loss(self, outputs, labels, padding_mask):
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
        loss = masked_loss.sum()

        return loss
        
    def train_epoch(self, data_loader):
        self.network.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        with tqdm(data_loader, desc='Training', unit='batch') as progress_bar:
            for node_sequences_batch, labels, lengths, padding_mask in progress_bar:
                labels = labels.to(self.device)
                node_sequences_batch = node_sequences_batch.to(self.device)
                padding_mask = padding_mask.to(self.device)

                # Forward pass
                outputs = self.network(self.train_graph, node_sequences_batch, padding_mask, self.train_casuality_mask)
                loss = self.calc_loss(outputs, labels, ~padding_mask) / lengths.sum()

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                top1_pred_indices = torch.argmax(outputs, dim=-1)
                top1_label_indices = torch.argmax(labels, dim=-1)
                top_k_label_indices = torch.topk(labels, k=self.params['k'], dim=-1)[1]
                correct_top_1 = top1_pred_indices == top1_label_indices
                correct_top_k = torch.any(torch.eq(top1_pred_indices.unsqueeze(-1), top_k_label_indices), dim=-1)
                acc_top_1 = torch.sum(correct_top_1 * (~padding_mask)) / lengths.sum()
                acc_top_k = torch.sum(correct_top_k * (~padding_mask)) / lengths.sum()

                progress_bar.set_postfix({'loss': loss.item(),
                                          'acc': acc_top_1.item(),
                                          f'top{self.params["k"]}': acc_top_k.item()})

        return total_loss / len(data_loader), acc_top_1, acc_top_k

    def validate(self, data_loader):
        self.logger.info("TractoGNNTrainer: Validation phase")
        self.network.eval()
        total_loss, total_correct, total_samples = 0, 0, 0
        with torch.no_grad():
            for node_sequences_batch, labels, lengths, padding_mask in data_loader:
                labels = labels.to(self.device)
                node_sequences_batch = node_sequences_batch.to(self.device)
                padding_mask = padding_mask.to(self.device)

                # Forward pass
                outputs = self.network(self.val_graph, node_sequences_batch, padding_mask, self.val_casuality_mask)
                loss = self.calc_loss(outputs, labels, ~padding_mask) / lengths.sum()

                total_loss += loss.item()

                top1_pred_indices = torch.argmax(outputs, dim=-1)
                top1_label_indices = torch.argmax(labels, dim=-1)
                top_k_label_indices = torch.topk(labels, k=self.params['k'], dim=-1)[1]
                correct_top_1 = top1_pred_indices == top1_label_indices
                correct_top_k = torch.any(torch.eq(top1_pred_indices.unsqueeze(-1), top_k_label_indices), dim=-1)
                acc_top_1 = torch.sum(correct_top_1 * (~padding_mask)) / lengths.sum()
                acc_top_k = torch.sum(correct_top_k * (~padding_mask)) / lengths.sum()

        return total_loss / len(data_loader), acc_top_1.item(), acc_top_k.item()

    def train(self):
        train_stats, val_stats = [], []
        for epoch in range(self.num_epochs):
            self.logger.info("TractoGNNTrainer: Training Epoch")
            train_loss, train_acc, train_acc_top_k = self.train_epoch(self.train_data_handler.data_loader)
            val_loss, val_acc, val_acc_top_k = self.validate(self.val_data_handler.data_loader)

            train_stats.append((train_loss, train_acc, train_acc_top_k))
            val_stats.append((val_loss, val_acc, val_acc_top_k))

            print(f'Epoch {epoch + 1}/{self.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                  f'Val Acc: {val_acc:.4f}, Val Top {self.params["k"]} Acc: {val_acc_top_k:.4f}')
            
        return train_stats, val_stats
