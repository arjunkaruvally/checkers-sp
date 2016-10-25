from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import numpy as np

'''
	Game Board Legend
		5 - invalid square
		0 - no coin
		-1 - player coin
		-2 - player king coin
		1 - opponent coin
		2 - opponent king coin
'''

class Node(object):
	def __init__(self, score=0):
		self.score = score
		self.moves = []
		self.children = []

	def add_child(self, obj, moves):
		self.chilren.append=obj
		self.moves=moves

class PlayingAgent:

	def __init__(self):
		self.seed = 100
		self.current_gen = 0
		self.gen_limit=100
		self.hidden_layers=3
		self.layer_nodes=32
		self.model = Sequential()
		self.rand_gen = np.random.RandomState(self.seed)

		for x in range(0,self.hidden_layers-1):
			self.model.add(Dense(self.layer_nodes, activation='sigmoid', input_dim=32, bias=True))		
		self.model.add(Dense(1, activation='sigmoid', bias=True))

		sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
		self.model.compile(loss='mean_squared_error', optimizer=sgd)

	def randomize_weights(self,model):
		for x in range(0,self.hidden_layers-1):
			dum_weight = np.zeros((self.layer_nodes,))
			model_weights = self.rand_gen.rand(self.layer_nodes,self.layer_nodes)
			model_weights = np.subtract(model_weights,0.5)
			model.layers[x].set_weights([model_weights,dum_weight])
					
		dum_weight = np.zeros((1,))		
		model_weights = self.rand_gen.rand(self.layer_nodes,1)
		model.layers[2].set_weights([model_weights,dum_weight])
		

	def get_heuristic_cost(self,board_config):
		output_pre = self.model.predict(self, board_config, batch_size=1, verbose=0)
		return output_pre

	def train(self):
		self.randomize_weights(self.model)
		print self.model.layers[0].get_weights()
		model_opponent = Sequential()
		
		for x in range(0,self.hidden_layers-1):
			model_opponent.add(Dense(self.layer_nodes, activation='sigmoid', input_dim=32, bias=True))

		model_opponent.add(Dense(1, activation='sigmoid', bias=True))

		sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
		model_opponent.compile(loss='mean_squared_error', optimizer=sgd)

		for x in range(0,self.gen_limit):
			self.randomize_weights(model_opponent)

	def in_boundary(self,pos):
		if pos[0] < 0 or pos[0] > 8:
			return False
		if pos[1] < 0 or pos[1] > 8:
			return False
		return True

	def get_possible_moves(x,y,mover):
		sur=[]
		if x>0 and y>0 and ((mover%2!=0) or simulated_board[x][y]==2):
			sur.append(( x-1, y-1 ))
		if x<7 and y>0 and ((mover%2==0) or simulated_board[x][y]==2):
			sur.append(( x+1, y-1 ))
		if x<7 and y<7 and ((mover%2==0) or simulated_board[x][y]==2):
			sur.append(( x+1, y+1 ))
		if x>0 and y<7 and ((mover%2!=0) or simulated_board[x][y]==2):
			sur.append(( x-1, y+1 ))

	def get_jumps(x,y,mover):
		ret = []
		sur = []
		if x>0 and y>0 and ((mover%2!=0) or simulated_board[x][y]==2):
			sur.append(( x-1, y-1 ))
		if x<7 and y>0 and ((mover%2==0) or simulated_board[x][y]==2):
			sur.append(( x+1, y-1 ))
		if x<7 and y<7 and ((mover%2==0) or simulated_board[x][y]==2):
			sur.append(( x+1, y+1 ))
		if x>0 and y<7 and ((mover%2!=0) or simulated_board[x][y]==2):
			sur.append(( x-1, y+1 ))
		for x in sur:
			if simulated_board(x) < 0:
				ret.append(x)
		return [sur,ret]

	def minmax(self, simulated_board, depth=2):

		'''
			- global moves array(2D)
			- 2nd dimension is the current move
		'''

		print 'minmax'

		current_move = [];
		sim_score=0
		mover=0
		coins = 8

		Tree = Node()

		for x in range(0,8):
			if coins <= 0:
				break;
			for y in range(0,8):
				# print str(x)+" "+str(y)+" "+str(simulated_board[x][y])
				if simulated_board[x][y] != 5 and simulated_board[x][y] > 0:
					coins=coins-1
					playing_coin = (x,y)

					moves = get_jumps(x,y,mover)
					jumps = moves[1]
					moves = moves[0]


board = np.array([
	[ 1, 5, 1, 5, 1, 5, 1, 5],
	[ 5, 1, 5, 1, 5, 1, 5, 1],
	[ 1, 5, 1, 5, 1, 5, 1, 5],
	[ 5, 0, 5, 0, 5, 0, 5, 0],
	[ 0, 5, 0, 5, 0, 5, 0, 5],
	[ 5,-1, 5,-1, 5,-1, 5,-1],
	[-1, 5,-1, 5,-1, 5,-1, 5],
	[ 5,-1, 5,-1, 5,-1, 5,-1]
])

my_nn = PlayingAgent()

my_nn.minmax(board,4)