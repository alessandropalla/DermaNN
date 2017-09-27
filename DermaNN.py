from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras import applications
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout, Input, MaxPooling2D, Conv2D
from keras import backend as K
from keras.utils import plot_model
from keras.models import load_model
from keras import optimizers
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, TensorBoard, Callback
from sklearn.metrics import roc_auc_score
import numpy as np

def ResNetPreprocessing(x):
	x = np.expand_dims(x, axis=0)
	return applications.resnet50.preprocess_input(x)

class IntervalEvaluation(Callback):
	def __init__(self, validation_generator, interval=10, steps = 60):
		super(Callback, self).__init__()

		self.interval = interval
		self.validation_generator = validation_generator
		self.steps = steps

	def on_epoch_end(self, epoch, logs={}):
		if epoch % self.interval == 0:
			y_val = np.empty((0,), int)
			y_pred = np.empty((0,1), int)
			for i in range(0, self.steps):
				x_val, y_val_tmp = self.validation_generator.next()
				y_pred_tmp = self.model.predict_on_batch(x_val)
				y_val= np.append(y_val, y_val_tmp, axis=0)
				y_pred= np.append(y_pred, y_pred_tmp, axis=0)
			score = roc_auc_score(y_val, y_pred)
			print("Epoch: {:d} - AUC: {:.6f}".format(epoch+1, score))


class DermaNN():
	def __init__(self):
		print('DermaNN')
		
	def build_new(self, name = 'ResNet', freeze_first = True, model = None, width = 224, height = 224):
		if (model == None):
			self.name = name
			# create the base pre-trained model
			if (name == 'ResNet'):
				self.width = 224
				self.height = 224
				self.pre = ResNetPreprocessing
				self.base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(self.height, self.width, 3), pooling='avg')
			elif(name== 'Inception'):
				self.width = 299
				self.height = 299
				self.pre = applications.inception_v3.preprocess_input
				self.base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(self.height, self.width, 3), pooling='avg')
			else:
				raise ValueError('Not ResNet nor Inception selected')

			# add a global spatial average pooling layer
			self.x = self.base_model.output
			self.x = Dense(100, activation='relu')(self.x)
			self.x = Dropout(0.75)(self.x)
			# and a softmax layer with 3 classes
			self.predictions = Dense(1, activation='sigmoid')(self.x)
			model.summary()

			# this is the model we will train
			self.model = Model(inputs=self.base_model.input, outputs=self.predictions)
			
			if(freeze_first):
				# first: train only the top layers (which were randomly initialized)
				for layer in self.base_model.layers:
					layer.trainable = False
			
		else:
			self.pre = applications.inception_v3.preprocess_input
			self.model = model 
			self.width = width
			self.height = height
			self.name = name
			
		self.train_datagen = image.ImageDataGenerator(
					fill_mode='wrap',
					rotation_range=20,
					width_shift_range=0.2,
					height_shift_range=0.2,
					horizontal_flip=True,
					vertical_flip=True,
					preprocessing_function=self.pre)
			
	def set_all_trainable(self):
		for layer in self.model.layers:
			layer.trainable = True
		
	def train(self, batch_size = 10, n_epoch = 150):
	
		self.test_datagen  = image.ImageDataGenerator(preprocessing_function=self.pre)

		opt = optimizers.Adam(lr=0.001)
		self.model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
		
			
		train_generator = self.train_datagen.flow_from_directory( #enhanced dataset
		#train_generator = self.test_datagen.flow_from_directory( #normal dataset
				'db/train',
				target_size=(self.height, self.width),
				batch_size=batch_size,
				class_mode='binary')
				
		validation_generator = self.test_datagen.flow_from_directory(
				'db/validation',
				target_size=(self.height, self.width),
				batch_size=batch_size,
				class_mode='binary')
				
		callbacks=[ModelCheckpoint('./model/Derma'+ self.name + '.h5', monitor='val_acc', save_best_only=True, period=1),
					CSVLogger('./model/Derma'+ self.name + '.csv'),
					ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)]
					#IntervalEvaluation(validation_generator= validation_generator, interval=1, steps = 15),
					#TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)]

		# Fit data
		self.model.fit_generator(
				train_generator,
				steps_per_epoch=200,
				epochs=n_epoch,
				validation_data=validation_generator,
				validation_steps=15,
				callbacks=callbacks)
				
	def evaluate(self, batch_size = 10, height = 224, width = 224 ):
		# Predictions
		self.test_datagen  = image.ImageDataGenerator(preprocessing_function=self.pre)
		test_generator = self.test_datagen.flow_from_directory(
				'db/test',
				target_size=(height, width),
				batch_size=batch_size,
				class_mode='binary')
		test_loss, test_accuracy = self.model.evaluate_generator( test_generator, steps=60)
		print('test_loss: %.4f - test_acc = %.4f'%(test_loss, test_accuracy))
		
		y_val = np.empty((0,), int)
		y_pred = np.empty((0,1), int)
		for i in range(0, 60):
			x_val, y_val_tmp = test_generator.next()
			y_pred_tmp = self.model.predict_on_batch(x_val)
			y_val= np.append(y_val, y_val_tmp, axis=0)
			y_pred= np.append(y_pred, y_pred_tmp, axis=0)
		score = roc_auc_score(y_val, y_pred)
		print("test AUC: {:.6f}".format(score))
		
		
	def load_model(self, filename, name = None, height = 224, width = 224):
		self.model = model = load_model(filename)
		self.name = name
		tmp =  model.input_shape
		self.height = tmp[1]
		self.width = tmp[2]
		if (name == 'ResNet'):
			self.pre = ResNetPreprocessing
		elif(name== 'Inception'):
			self.pre = applications.inception_v3.preprocess_input
		else:
			self.name = name
			self.pre = None
			self.height = height
			self.width = width
		
	
if __name__ == "__main__":

	model = Sequential()
	# input: 300x300 images with 3 channels -> (300, 300, 3) tensors.
	# this applies 4 convolution filters
	model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(300, 300, 3)))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(32, (5, 5), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	
	model.add(Conv2D(32, (5, 5), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	
	model.add(Conv2D(16, (5, 5), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	model.summary()
	
	
	nn = DermaNN()
	#nn.build_new('ConvNet', freeze_first = False, model = model, width = 300, height = 300 )
	#nn.build_new('ResNet', freeze_first = True)
	#nn.train(n_epoch = 5)
	nn.load_model( filename = './model/DermaConvNet.h5', name = 'ConvNet', width = 300, height = 300)
	#nn.set_all_trainable()
	#nn.train(n_epoch = 150)
	nn.evaluate(width = 300, height = 300)

