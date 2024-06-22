import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
import torch.optim as optim
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
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max',
                                                              factor=params['decay_factor'],
                                                              patience=params['decay_LR_patience'],
                                                              threshold=params['threshold'],
                                                              threshold_mode='abs',
                                                              min_lr=params['min_lr'])

        self.num_epochs = params['epochs']
        self.criterion = nn.KLDivLoss(reduction='none')
        self.train_causality_mask = self.train_data_handler.causality_mask.to(self.device)
        self.val_causality_mask = self.val_data_handler.causality_mask.to(self.device)
        self.trained_model_path = params['trained_model_path']
        self.params = params
    

    def calc_loss(self, outputs, labels, valid_mask):
        """
        Calculate the masked loss using KLDivLoss for padded sequences.

        Parameters:
        - outputs (Tensor): Log probabilities of shape [batch_size, seq_length, 725].
        - labels (Tensor): True probabilities with shape [batch_size, seq_length, 725].
        - valid_mask (Tensor): A boolean tensor of shape [batch_size, seq_length] where True
                                 indicates valid points and False indicates padded points.

        Returns:
        - loss (Tensor): Scalar tensor representing the average loss over all valid points.
        """

        # Calculate the element-wise loss
        elementwise_loss = self.criterion(outputs, labels)

        # Apply the padding mask to ignore loss for padded values
        # We need to unsqueeze the padding_mask to make it broadcastable to elementwise_loss shape
        masked_loss = elementwise_loss * valid_mask.unsqueeze(-1)
        
        # Calculate the average loss per valid sequence element
        loss = masked_loss.sum()

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
                loss = self.calc_loss(outputs, labels, ~padding_mask) / lengths.sum()

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                curr_loss = loss.item()
                total_loss += loss.item()

                top1_pred_indices = torch.argmax(outputs, dim=-1)
                top1_label_indices = torch.argmax(labels, dim=-1)
                top_k_label_indices = torch.topk(labels, k=self.params['k'], dim=-1)[1]
                correct_top_1 = top1_pred_indices == top1_label_indices
                correct_top_k = torch.any(torch.eq(top1_pred_indices.unsqueeze(-1), top_k_label_indices), dim=-1)
                acc_top_1 = torch.sum(correct_top_1 * (~padding_mask)) / lengths.sum()
                acc_top_k = torch.sum(correct_top_k * (~padding_mask)) / lengths.sum()

                progress_bar.set_postfix({'loss': curr_loss,
                                          'acc': acc_top_1.item(),
                                          f'top{self.params["k"]}': acc_top_k.item()})


        average_loss = total_loss / len(data_loader)

        return average_loss, acc_top_1.item(), acc_top_k.item()

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
                loss = self.calc_loss(outputs, labels, ~padding_mask) / lengths.sum()

                curr_loss = loss.item()
                total_loss += loss.item()

                top1_pred_indices = torch.argmax(outputs, dim=-1)
                top1_label_indices = torch.argmax(labels, dim=-1)
                top_k_label_indices = torch.topk(labels, k=self.params['k'], dim=-1)[1]
                correct_top_1 = top1_pred_indices == top1_label_indices
                correct_top_k = torch.any(torch.eq(top1_pred_indices.unsqueeze(-1), top_k_label_indices), dim=-1)
                acc_top_1 = torch.sum(correct_top_1 * (~padding_mask)) / lengths.sum()
                acc_top_k = torch.sum(correct_top_k * (~padding_mask)) / lengths.sum()

        average_loss = total_loss / len(data_loader)

        if self.params['decay_LR']:
            self.scheduler.step(average_loss)

        return average_loss, acc_top_1.item(), acc_top_k.item()

    def train(self):
        train_stats, val_stats = [], []
        for epoch in range(self.num_epochs):
            self.logger.info("TractoGNNTrainer: Training Epoch")
            train_loss, train_acc, train_acc_top_k = self.train_epoch(self.train_data_handler.data_loader)
            val_loss, val_acc, val_acc_top_k = self.validate(self.val_data_handler.data_loader)

            train_stats.append((train_loss, train_acc, train_acc_top_k))
            val_stats.append((val_loss, val_acc, val_acc_top_k))

            self.logger.info(f'Epoch {epoch + 1}/{self.num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train Acc Topk: {train_acc_top_k:.4f}' 
                             f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Acc TopK: {val_acc_top_k:.4f}'
                             f'Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            if epoch % 2 and self.params['save_checkpoints']:
                torch.save(self.network.state_dict, self.trained_model_path)

        torch.save(self.network.state_dict, self.trained_model_path)
        return train_stats, val_stats
