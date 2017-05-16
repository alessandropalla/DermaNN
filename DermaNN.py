from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras import applications
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout, Input
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
			y_val = np.empty((0,3), int)
			y_pred = np.empty((0,3), int)
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
		
	def build_new(self, name = 'ResNet', model = None, width = 224, height = 224):
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
			self.predictions = Dense(3, activation='softmax')(self.x)

			# this is the model we will train
			self.model = Model(inputs=self.base_model.input, outputs=self.predictions)
			
			self.train_datagen = image.ImageDataGenerator(
					fill_mode='wrap',
					rotation_range=20,
					width_shift_range=0.2,
					height_shift_range=0.2,
					horizontal_flip=True,
					vertical_flip=True,
					preprocessing_function=self.pre)
			
		else:
			self.pre = None
			self.model = model 
			self.width = width
			self.height = height
		
	def train(self, freeze_first = True, batch_size = 10, n_epoch = 150):
	
		self.test_datagen  = image.ImageDataGenerator(preprocessing_function=self.pre)
		
		if(freeze_first):
			# first: train only the top layers (which were randomly initialized)
			for layer in self.base_model.layers:
				layer.trainable = False

		opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
		self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
		
			
		#train_generator = self.train_datagen.flow_from_directory( #enhanced dataset
		train_generator = self.test_datagen.flow_from_directory( #normal dataset
				'db/train',
				target_size=(self.height, self.width),
				batch_size=batch_size,
				class_mode='categorical')
				
		validation_generator = self.test_datagen.flow_from_directory(
				'db/validation',
				target_size=(self.height, self.width),
				batch_size=batch_size,
				class_mode='categorical')
				
		callbacks=[ModelCheckpoint('./model/Derma'+ self.name + '.h5', monitor='val_acc', save_best_only=True, period=1),
					CSVLogger('./model/Derma'+ self.name + '.csv'),
					ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001),
					IntervalEvaluation(validation_generator= validation_generator, interval=1, steps = 15),
					TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)]

		# Fit data
		self.model.fit_generator(
				train_generator,
				steps_per_epoch=200,
				epochs=n_epoch,
				validation_data=validation_generator,
				validation_steps=15,
				callbacks=callbacks)
				
	def evaluate(self, preprocessing = 'ResNet', batch_size = 10, height = 224, width = 224 ):
		# Predictions
		test_generator = self.test_datagen.flow_from_directory(
				'db/test',
				target_size=(height, width),
				batch_size=batch_size,
				class_mode='categorical')
		print(self.model.evaluate_generator( test_generator, steps=60))
		
	def load_model(self, filename, preprocessing = None):
		self.model = model = load_model(filename)
		if (preprocessing == 'ResNet'):
			self.pre = ResNetPreprocessing
		elif(preprocessing== 'Inception'):
			self.pre = applications.inception_v3.preprocess_input
		else:
			self.pre = None
		
	
if __name__ == "__main__":
	nn = DermaNN()
	nn.build_new('ResNet')
	#nn.load_model( './model/DermaInception.h5', 'Inception')
	nn.train(freeze_first = False, n_epoch = 30)
	nn.evaluate()

