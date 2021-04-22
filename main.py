from training import main_loop
from arg_parser import training_parser

def main():
	args = training_parser().parse_args()

	LR                    = args.learning_rate
	MODEL                 = args.model
	EPOCHS                = args.epochs
	CRIT_ITER             = args.crit_iter
	BATCH_SIZE            = args.batch_size
	LAMBDA_ADV            = args.lambda_adv
	EPOCH_START           = args.epoch_start
	LAMBDA_GRD_PEN        = args.lambda_grd_pen
	DISC_ONLY_EPOCHS      = args.disc_only_epochs

	main_loop(LR, EPOCHS, BATCH_SIZE, EPOCH_START, LAMBDA_ADV, LAMBDA_GRD_PEN, CRIT_ITER, DISC_ONLY_EPOCHS, MODEL)


if __name__ == '__main__':
    main()
