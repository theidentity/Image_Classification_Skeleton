
def getGlobalParams():
	
	global_params = dict(
		batch_size = 64,
		epochs = 50,
		img_width = 139,
		img_height = 139,
		train_folder = 'data/train',
		validation_folder = 'data/valid',
		test_folder = 'data/test',
		model_prefix = 'xception_',
		n_classes = 2
	)

	return global_params

