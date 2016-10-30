from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

import numpy as np

class NeuralNetwork:	
	def __init__(self):
		self.current_gen = 0
		self.gen_limit=100
		self.hidden_layers=3
		self.layer_nodes=32
		self.model = Sequential()

		for x in range(0,self.hidden_layers-1):
			self.model.add(Dense(self.layer_nodes, activation='linear', input_dim=32, bias=True))		
		self.model.add(Dense(1, activation='linear', bias=True))

		sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
		self.model.compile(loss='mean_squared_error', optimizer=sgd)

	def set_random_seed(self,seed=5):
		self.rand_gen = np.random.RandomState(seed)

	def randomize_weights(self):
		for x in range(0,self.hidden_layers-1):
			dum_weight = np.zeros((self.layer_nodes,))
			model_weights = self.rand_gen.rand(self.layer_nodes,self.layer_nodes)
			model_weights = np.subtract(model_weights,0.5)
			self.model.layers[x].set_weights([model_weights,dum_weight])
					
		dum_weight = np.zeros((1,))		
		model_weights = self.rand_gen.rand(self.layer_nodes,1)
		self.model.layers[2].set_weights([model_weights,dum_weight])

	def get_heuristic_cost(self,board_config):
		board_config = np.array(board_config)
		board_config = board_config.reshape((1,32))
		output_pre = self.model.predict(board_config, verbose=0)
		return np.sum(output_pre)