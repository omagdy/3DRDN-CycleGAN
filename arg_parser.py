import argparse

def training_parser():
    parser = argparse.ArgumentParser(description='Training arguments.')
    
    parser.add_argument('-lr', '--learning_rate', action='store',
                         default=1e-4, type=float, help=('Learning Rate. Default: 0.0001'))
    parser.add_argument('-bs', '--batch_size', action='store', 
                         default=8, type=int, help='Batch Size. Default: "8"')
    parser.add_argument('-ep', '--epochs', action='store', default=1, 
                         type=int, help=('Epochs. Default: 1'))
    parser.add_argument('-eps', '--epoch_start', action='store', default=0, 
                         type=int, help=('Starting Epoch. Default: 0'))
    parser.add_argument('-lt', '--loss_type', action='store', default='l1_loss', 
                         type=str, choices=['l1_loss', 'l2_loss'], 
                         help=('Loss type, either L1 (MAE) or L2 (MSE). Default: L1'))
    parser.add_argument('-k', '--growth_rate', action='store', default=12, 
                         type=int, help=('Growth rate of the filters for each layer in the dense block. Default: 12'))
    parser.add_argument('-ub', '--utilize_bias', action='store', default=True, 
                         type=bool, help=('Whether to utilize bias in the convolutional layers or not. Default: True'))
    parser.add_argument('-ndb', '--no_of_dense_blocks', action='store', default=2, 
                         type=int, help=('Number of dense blocks in the generator. Default: 2'))
    parser.add_argument('-nub', '--no_of_units_per_block', action='store', default=3, 
                         type=int, help=('Number of BNN-LEAKYRELU-Conv layers in each dense block. Default: 3'))
    parser.add_argument('-wi', '--weights_init', action='store', default='HeUniform', 
                         type=str, choices=['HeUniform', 'HeNormal', 'GlorotUniform', 'GlorotNormal'],
                         help=('Initializer for the kernel weights matrix. Default: HeUniform'))
    parser.add_argument('-af', '--activation_func', action='store', default='LEAKYRELU', 
                         type=str, choices=['LEAKYRELU', 'SIWSH', 'ELU', 'RELU'],
                         help=('Activation Function used in the dense blocks. Default: LEAKYRELU'))

    return parser
