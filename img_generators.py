from keras.preprocessing.image import ImageDataGenerator


def getTrainGenerator(folder_path):

	train_datagen = ImageDataGenerator(
		rotation_range=0.1,
		zoom_range=0.1,
		width_shift_range=0.1,
		height_shift_range=0.1,
		rescale=1/255.0,
		horizontal_flip=True,
		fill_mode='nearest')

	train_generator = train_datagen.flow_from_directory(
		folder_path,
		target_size = (img_width,img_height),
		batch_size = batch_size,
		class_mode = 'categorical'
		)

	return train_generator

def getValidationGenerator(folder_path):
	
	validation_datagen = ImageDataGenerator(
		rescale=1/255.0)

	validation_generator = validation_datagen.flow_from_directory(
		folder_path,
		target_size = (img_width,img_height),
		batch_size = batch_size,
		class_mode = 'categorical'
		)

	return validation_generator

def getTestGenerator(folder_path):

	test_datagen = ImageDataGenerator(
		rescale=1/255.0)

	test_generator = test_datagen.flow_from_directory(
		folder_path,
		target_size = (img_width,img_height),
		batch_size = batch_size,
		class_mode = None
		)

	return test_generator