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
    parser.add_argument('-mo', '--model', action='store', default='3DRLDSRN', 
                         type=str, choices=['3DRLDSRN', 'WGANGP-3DRLDSRN', 'CYCLE-WGANGP-3DRLDSRN'],
                         help=('Model used during training. Default: 3DRLDSRN'))
    return parser
