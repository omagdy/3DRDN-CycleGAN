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
	LAMBDA_IDT            = args.lambda_idt
	LAMBDA_CYC            = args.lambda_cyc
	TRAIN_ONLY            = args.train_only
	EPOCH_START           = args.epoch_start
	LAMBDA_GRD_PEN        = args.lambda_grd_pen

	main_loop(LR, EPOCHS, BATCH_SIZE, EPOCH_START, LAMBDA_ADV, LAMBDA_GRD_PEN,
	 LAMBDA_CYC, LAMBDA_IDT, CRIT_ITER, TRAIN_ONLY, MODEL)


if __name__ == '__main__':
    main()
