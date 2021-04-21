from training import main_loop
from arg_parser import training_parser

def main():
	args = training_parser().parse_args()

	LR_G                  = args.learning_rate
	MODEL                 = args.model
	EPOCHS                = args.epochs
	BATCH_SIZE            = args.batch_size
	EPOCH_START           = args.epoch_start

	main_loop(LR_G, EPOCHS, BATCH_SIZE, EPOCH_START, MODEL)


if __name__ == '__main__':
    main()
