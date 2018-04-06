
def getGlobalParams():
	
	global_params = dict(
		batch_size = 32,
		epochs = 50,
		img_width = 197,
		img_height = 197,
		train_folder = 'data/train',
		validation_folder = 'data/test',
		test_folder = 'data/test',
		model_prefix = 'resnet50',
		n_classes = 2
	)

	return global_params

	