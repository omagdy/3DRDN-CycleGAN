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
    parser.add_argument('-ld1', '--lambda_adv', action='store', default=0.1, 
                         type=int, help=('Lambda hyperparameter for generator adversarial loss. Default: 0.1'))
    parser.add_argument('-ld2', '--lambda_grd_pen', action='store', default=10, 
                         type=int, help=('Lambda hyperparameter for discriminator gradient penalty. Default: 10'))
    parser.add_argument('-ci', '--crit_iter', action='store', default=7, 
                         type=int, help=('Iterations for training discriminator for each generator step. Default: 7'))
    parser.add_argument('-de', '--disc_only_epochs', action='store', default=10, 
                         type=int, help=('Initial number of epochs where only Discriminator is trained. Default: 10'))
    parser.add_argument('-mo', '--model', action='store', default='3DRLDSRN', 
                         type=str, choices=['3DRLDSRN', 'WGANGP-3DRLDSRN', 'CYCLE-WGANGP-3DRLDSRN'],
                         help=('Model used during training. Default: 3DRLDSRN'))
    return parser
