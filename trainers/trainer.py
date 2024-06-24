import torch
import torch.nn as nn
import pickle
from tqdm import tqdm
from torch.optim import Adam
import torch.optim as optim
from models.network import TractoTransformer
from torch.utils.tensorboard import SummaryWriter
from data_handling import *
from utils.trainer_utils import *


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
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                              factor=params['decay_factor'],
                                                              patience=params['decay_LR_patience'],
                                                              threshold=params['threshold'],
                                                              threshold_mode='abs',
                                                              min_lr=params['min_lr'],
                                                              cooldown=2)

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
        loss = masked_loss.sum() / valid_mask.sum()

        return loss

    def train_epoch(self, data_loader):
        self.network.train()
        total_loss, total_acc_top_1, total_acc_top_k = 0, 0, 0
        with tqdm(data_loader, desc='Training', unit='batch') as progress_bar:
            for streamline_voxels_batch, labels, lengths, padding_mask in progress_bar:
                labels = labels.to(self.device)
                streamline_voxels_batch = streamline_voxels_batch.to(self.device)
                padding_mask = padding_mask.to(self.device)

                # Forward pass
                outputs = self.network(self.train_dwi_data, streamline_voxels_batch, padding_mask, self.train_causality_mask)
                loss = self.calc_loss(outputs, labels, ~padding_mask)
                acc_top_1, acc_top_k = calc_metrics(outputs, labels, ~padding_mask, self.params['k'])

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_acc_top_1 += acc_top_1
                total_acc_top_k += acc_top_k

                progress_bar.set_postfix({'loss': loss.item(),
                                          'acc': acc_top_1,
                                          f'top{self.params["k"]}': acc_top_k})


        train_loss = total_loss / len(data_loader)
        train_acc_top_1 = total_acc_top_1 / len(data_loader)
        train_acc_top_k = total_acc_top_k / len(data_loader)

        return {'loss': train_loss,
                'accuracy_top_1': train_acc_top_1,
                'accuracy_top_k': train_acc_top_k 
                }

    def validate(self, data_loader):
        self.logger.info("TractoGNNTrainer: Validation phase")
        self.network.eval()
        total_loss, total_acc_top_1, total_acc_top_k = 0, 0, 0
        with torch.no_grad():
            for streamline_voxels_batch, labels, lengths, padding_mask in data_loader:
                labels = labels.to(self.device)
                streamline_voxels_batch = streamline_voxels_batch.to(self.device)
                padding_mask = padding_mask.to(self.device)

                # Forward pass
                outputs = self.network(self.val_dwi_data, streamline_voxels_batch, padding_mask, self.val_causality_mask)
                loss = self.calc_loss(outputs, labels, ~padding_mask)
                acc_top_1, acc_top_k = calc_metrics(outputs, labels, ~padding_mask, self.params['k'])

                total_loss += loss.item()
                total_acc_top_1 += acc_top_1
                total_acc_top_k += acc_top_k


        val_loss = total_loss / len(data_loader)
        val_acc_top_1 = total_acc_top_1 / len(data_loader)
        val_acc_top_k = total_acc_top_k / len(data_loader)

        if self.params['decay_LR']:
            self.scheduler.step(val_loss)

        return {'loss': val_loss,
                'accuracy_top_1': val_acc_top_1,
                'accuracy_top_k': val_acc_top_k 
                }


    def train(self):
        # Initialize writer
        log_dir = "logs"
        stats_path = os.path.join(log_dir, 'train_val_stats.pkl')
        writer = SummaryWriter(log_dir=log_dir)

        # Log hyperparameters
        writer.add_hparams(fetch_hyper_params(self.params), {})
        writer.add_text('FODFs prediction', 'This is an experiment ONE fiber bundle', 0)

        train_stats, val_stats = [], []
        for epoch in range(self.num_epochs):
            self.logger.info("TractoGNNTrainer: Training Epoch")
            train_metrics = self.train_epoch(self.train_data_handler.data_loader)
            val_metrics = self.validate(self.val_data_handler.data_loader)

            # Print epoch message
            self.logger.info(get_epoch_message(self, train_metrics, val_metrics, epoch))

            # Log metrics
            for metric_name, metric_value in train_metrics.items():
                writer.add_scalar(f'Train/{metric_name}', metric_value, epoch)
        
            for metric_name, metric_value in val_metrics.items():
                writer.add_scalar(f'Val/{metric_name}', metric_value, epoch)

            # Log model parameters
            for name, param in self.network.named_parameters():
                writer.add_histogram(name, param, epoch)
                writer.add_histogram(f'{name}.grad', param.grad, epoch)

            # Save statistics
            train_stats.append((train_metrics['loss'], train_metrics['accuracy_top_1'], train_metrics['accuracy_top_k']))
            val_stats.append((val_metrics['loss'], val_metrics['accuracy_top_1'], val_metrics['accuracy_top_k']))
            
            # Save checkpoints
            if self.params['save_checkpoints']:
                save_checkpoints(self, stats_path, train_stats, val_stats, epoch+1)

        
        save_checkpoints(self, stats_path, train_stats, val_stats, self.num_epochs)
        writer.flush()
        writer.close()
        return train_stats, val_stats
