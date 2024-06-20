import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from models.network import TractoTransformer
from data_handling import *


class TractoGNNTrainer(object):
    def __init__(self, logger, params):
        logger.info("Create TractoGNNTrainer object")
        self.logger = logger
        self.device = params['device']
        self.network = TractoTransformer(logger=logger, params=params).to(self.device)
        self.train_data_handler = SubjectDataHandler(logger=logger, params=params, mode=TRAIN)
        self.val_data_handler = SubjectDataHandler(logger=logger, params=params, mode=VALIDATION)
        self.train_dwi_data = self.train_data_handler.dwi.to(self.device)
        self.val_dwi_data = self.val_data_handler.dwi.to(self.device)
        self.optimizer = Adam(self.network.parameters(), lr=params['learning_rate'])
        self.num_epochs = params['epochs']
        self.criterion = nn.MSELoss(reduction='none')
        self.train_causality_mask = self.train_data_handler.causality_mask.to(self.device)
        self.val_causality_mask = self.val_data_handler.causality_mask.to(self.device)
        self.trained_model_path = params['trained_model_path']
        self.params = params
    

    def calc_loss(self, outputs, labels, valid_mask):
        """
        Calculate the masked loss using MSELoss for padded sequences.

        Parameters:
        - outputs (Tensor): Predictions of shape [batch_size, seq_length, 2].
        - labels (Tensor): True values with shape [batch_size, seq_length-1, 2].
        - valid_mask (Tensor): A boolean tensor of shape [batch_size, seq_length-1] where True
                               indicates valid points and False indicates padded points.

        Returns:
        - loss (Tensor): Scalar tensor representing the average loss over all valid points.
        """

        # Calculate the element-wise loss
        elementwise_loss = self.criterion(outputs[:, :-1], labels)

        # Apply the padding mask to ignore loss for padded values
        # We need to unsqueeze the padding_mask to make it broadcastable to elementwise_loss shape
        masked_loss = elementwise_loss * valid_mask[: , :-1].unsqueeze(-1)
        
        # Calculate the average loss per valid sequence element
        loss = masked_loss.sum() / valid_mask.sum()

        return loss
        
    def train_epoch(self, data_loader):
        self.network.train()
        total_loss = 0
        with tqdm(data_loader, desc='Training', unit='batch') as progress_bar:
            for streamline_voxels_batch, labels, lengths, padding_mask in progress_bar:
                labels = labels.to(self.device)
                streamline_voxels_batch = streamline_voxels_batch.to(self.device)
                padding_mask = padding_mask.to(self.device)

                # Forward pass
                outputs = self.network(self.train_dwi_data, streamline_voxels_batch, padding_mask, self.train_causality_mask)
                loss = self.calc_loss(outputs, labels, ~padding_mask)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
                curr_loss = loss.item()
                total_loss += loss.item()

                progress_bar.set_postfix({'loss': curr_loss})

        return total_loss / len(data_loader)

    def validate(self, data_loader):
        self.logger.info("TractoGNNTrainer: Validation phase")
        self.network.eval()
        total_loss = 0
        with torch.no_grad():
            for streamline_voxels_batch, labels, lengths, padding_mask in data_loader:
                labels = labels.to(self.device)
                streamline_voxels_batch = streamline_voxels_batch.to(self.device)
                padding_mask = padding_mask.to(self.device)

                # Forward pass
                outputs = self.network(self.val_dwi_data, streamline_voxels_batch, padding_mask, self.val_causality_mask)
                loss = self.calc_loss(outputs, labels, ~padding_mask)

                total_loss += loss.item()

        return total_loss / len(data_loader)

    def train(self):
        train_stats, val_stats = [], []
        for epoch in range(self.num_epochs):
            self.logger.info("TractoGNNTrainer: Training Epoch")
            train_loss = self.train_epoch(self.train_data_handler.data_loader)
            val_loss = self.validate(self.val_data_handler.data_loader)

            train_stats.append(train_loss)
            val_stats.append(val_loss)

            self.logger.info(f'Epoch {epoch + 1}/{self.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        torch.save(self.network.state_dict, self.trained_model_path)
        return train_stats, val_stats
