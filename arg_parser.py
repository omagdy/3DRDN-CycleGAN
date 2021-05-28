import argparse

def training_parser():
    parser = argparse.ArgumentParser(description='Training arguments.')
    parser.add_argument('-lr', '--learning_rate', action='store',
                         default=10e-4, type=float, help=('Learning Rate. Default: 0.001'))
    parser.add_argument('-bs', '--batch_size', action='store', 
                         default=8, type=int, help='Batch Size. Default: "8"')
    parser.add_argument('-eps', '--epoch_start', action='store', default=0, 
                         type=int, help=('Starting Epoch. Default: 0'))
    parser.add_argument('-ep', '--epochs', action='store', default=1, 
                         type=int, help=('Epochs. Default: 1'))
    parser.add_argument('-db', '--dense_blocks', action='store', default=3, 
                         type=int, help=('Number of dense blocks. Default: 3'))
    parser.add_argument('-du', '--dense_units', action='store', default=4, 
                         type=int, help=('Number of dense units. Default: 4'))
    parser.add_argument('-ld1', '--lambda_adv', action='store', default=0.01, 
                         type=float, help=('Lambda hyperparameter for generator adversarial loss. Default: 0.01'))
    parser.add_argument('-ld2', '--lambda_grd_pen', action='store', default=10, 
                         type=int, help=('Lambda hyperparameter for discriminator gradient penalty. Default: 10'))
    parser.add_argument('-ld3', '--lambda_cyc', action='store', default=0.01, 
                         type=float, help=('Lambda hyperparameter for cycle consistency loss. Default: 0.01'))
    parser.add_argument('-ld4', '--lambda_idt', action='store', default=0.005, 
                         type=float, help=('Lambda hyperparameter for the identity loss. Default: 0.005'))
    parser.add_argument('-ci', '--crit_iter', action='store', default=3, 
                         type=int, help=('Iterations for training discriminator for each generator step. Default: 3'))
    parser.add_argument('-to', '--train_only', action='store', default='', 
                         type=str, choices=['', 'GENERATORS', 'DISCRIMINATORS'],
                         help=('Select to only train either generators or discriminators.'))
    parser.add_argument('-mo', '--model', action='store', default='3DRLN', 
                         type=str, choices=['3DRLN', '3DRLN-WGAN', '3DRLN-CGAN'],
                         help=('Model used during training. Default: 3DRLN'))
    return parser
