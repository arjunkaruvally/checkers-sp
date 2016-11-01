from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

import numpy as np

class NeuralNetwork:	
	def __init__(self):
		self.hidden_layers=3
		self.layer_nodes=32
		self.model = Sequential()
		self.fitness = 0.0
		self.generation = 0
		self.tag = ""

		self.model.add(Dense(42, activation='tanh', input_dim=32, bias=True))
		self.model.add(Dense(42, activation='tanh', bias=True))
		self.model.add(Dense(32, activation='tanh', bias=True))		
		self.model.add(Dense(1, activation='tanh', bias=True))

		sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
		self.model.compile(loss='mean_squared_error', optimizer=sgd)

	def set_random_seed(self,seed=5):
		self.rand_gen = np.random.RandomState(seed)

	def randomize_weights(self, scale=1):

		dum_weight = np.zeros((42,))
		model_weights = self.rand_gen.rand(32,42)
		model_weights = np.multiply(np.subtract(model_weights,0.5), scale)
		self.model.layers[0].set_weights([model_weights,dum_weight])
		
		dum_weight = np.zeros((42,))
		model_weights = self.rand_gen.rand(42,42)
		model_weights = np.multiply(np.subtract(model_weights,0.5), scale)
		self.model.layers[1].set_weights([model_weights,dum_weight])

		dum_weight = np.zeros((32,))
		model_weights = self.rand_gen.rand(42,32)
		model_weights = np.multiply(np.subtract(model_weights,0.5), scale)
		self.model.layers[2].set_weights([model_weights,dum_weight])

		dum_weight = np.zeros((1,))		
		model_weights = self.rand_gen.rand(32,1)
		model_weights = np.multiply(np.subtract(model_weights,0.5), scale)
		self.model.layers[3].set_weights([model_weights,dum_weight])

	def get_heuristic_cost(self,board_config):
		board_config = np.array(board_config)
		board_config = board_config.reshape((1,32))
		output_pre = self.model.predict(board_config, verbose=0)
		return np.sum(output_pre)

	def get_weights(self):
		for x in range(0,4):
			print "layer: "+str(x)
			print self.model.layers[x].get_weights()
			print "\n"

	def reset_fitness(self):
		self.fitness = 0
