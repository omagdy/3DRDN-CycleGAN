from arg_parser import training_parser
from training import training_loop

def main():
	args = training_parser().parse_args()

	LR_G                  = args.learning_rate
	EPOCHS                = args.epochs
	LOSS_FUNC             = args.loss_type
	BATCH_SIZE            = args.batch_size
	EPOCH_START           = args.epoch_start
	N_TRAINING_DATA       = args.n_training_data

	K                     = args.growth_rate
	UTILIZE_BIAS          = args.utilize_bias
	NO_OF_DENSE_BLOCKS    = args.no_of_dense_blocks
	NO_OF_UNITS_PER_BLOCK = args.no_of_units_per_block

	training_loop(LR_G, EPOCHS, BATCH_SIZE, N_TRAINING_DATA, LOSS_FUNC, EPOCH_START,
	 NO_OF_DENSE_BLOCKS, K, NO_OF_UNITS_PER_BLOCK, UTILIZE_BIAS)


if __name__ == '__main__':
    main()
