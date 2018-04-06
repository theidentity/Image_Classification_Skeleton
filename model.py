import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.optimizers import SGD,Adam,rmsprop
from keras import applications
from keras import Model

import img_generators 
import GLOBAL_PARAMS
global_params = GLOBAL_PARAMS.getGlobalParams()

def getPreTrained(unfreeze_all=False):
	model = applications.Xception(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
	if unfreeze_all:
		for layer in model.layers:
			layer.trainable = True
	return model

def addBottom(base_model):
	n_classes = global_params['n_classes']
	x = base_model.output
	x = Flatten()(x)
	x = Dense(128,activation='relu')(x)
	x = Dropout(0.25)(x)
	x = Dense(64,activation='relu')(x)
	x = Dropout(0.5)(x)
	predictions = Dense(n_classes,activation='softmax')(x)
	return predictions

def getModel():
	base_model = getPreTrained()
	predictions = addBottom(base_model)
	model = Model(input=base_model.input,output=predictions)
	return model

def getOptimizer():
	# opt = rmsprop(lr=0.0001, decay=1e-6)
	# opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
	opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	return opt

def getCallbacks():
	model_prefix = global_params['model_prefix']
	early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=5, verbose=1, mode='auto')
	checkpointer = ModelCheckpoint(filepath='models/'+model_prefix+'_best.h5', verbose=1, save_best_only=True)
	return [early_stopping,checkpointer]


train_generator = img_generators.getTrainGenerator(global_params['train_folder'])
validation_generator = img_generators.getValidationGenerator(global_params['validation_folder'])
test_generator = img_generators.getTestGenerator(global_params['test_folder'])

model = getModel()
opt = getOptimizer()

model.compile(
	loss='categorical_crossentropy',
	optimizer=opt,
	metrics=['accuracy'])

print model.summary()

model.fit_generator(
	train_generator,
	epochs = global_params['epochs'],
	validation_data=validation_generator,
	callbacks = getCallbacks()
	)

model.save('models/'+global_params['model_prefix']+'.h5')
