import torch


class Parameters(object):

    def __init__(self):
        self.params = dict()

        """ Model Parameters """

        # encoder_feature_dims - GCN encoder feature dimensions in each layer.
        # Number of layers determined by len(encoder_feature_dims)-1.
        self.params['graph_encoder_feature_dims'] = [103, 64, 32]

        # num_transformer_encoder_layers - Number of transformer encoder layers will be used for TractoGNN decoder.
        self.params['num_transformer_encoder_layers'] = 8

        # nhead - Number of heads in the Multi Head Self Attention mechanism of the TransformerEncoderLayer.
        self.params['nhead'] = 8

        # transformer_feed_forward_dim - Dimension of the feedforward network in TransformerEncoder layer.
        self.params['transformer_feed_forward_dim'] = 256

        # dropout_rate - Probability to execute a dropout
        self.params['dropout_rate'] = 0.3

        # max_streamline_len - Upper bound of an expected streamline length. Used for positional encoding.
        self.params['max_streamline_len'] = 250

        # output_size - Decoder output features size.
        self.params['output_size'] = 730

        # model_weights_save_dir - (string) Path for saving the model's files after training is done.
        self.params['model_weights_save_dir'] = ""

        """ Training Parameters """

        # learning_rate -(float) Initial learning rate in training phase.
        self.params['learning_rate'] = 7e-4

        # batch_size - (int) Data batch size for training.
        self.params['batch_size'] = 500

        # epochs - (int) Number of training epochs.
        self.params['epochs'] = 10

        # top k accuracy computation
        self.params['k'] = 7

        # decay_LR - (bool) Whether to use learning rate decay.
        self.params['decay_LR'] = True

        # decay_LR_patience - (int) Number of training epochs to wait in case validation performance does not improve
        # before learning rate decay is applied.
        self.params['decay_LR_patience'] = 2

        # decay_factor - (float [0, 1]) In an LR decay step, the existing LR will be multiplied by this factor.
        self.params['decay_factor'] = 0.6

        # early_stopping - (bool) Whether to use early stopping.
        self.params['early_stopping'] = True

        # early_stopping - (int) Number of epochs to wait before training is terminated when validation performance
        # does not improve.
        self.params['early_stopping_patience'] = 5

        # train_val_ratio - (float [0, 1]) Training/Validation split ratio for training.
        self.params['train_val_ratio'] = 0.8

        # device - Device for training, GPU if available and otherwise CPU.
        self.params['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        """ Data Parameters """

        # subject_folder - (string) Path to subject folder containing diffusion weighted image, white matter mask, 
        # spherical harmonics coefficients and reference tractograms.
        self.params['train_subject_folder'] = 'sub-1013'
        self.params['val_subject_folder'] = 'sub-1006'
       
    