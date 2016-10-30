from NeuralNetwork import NeuralNetwork
from Tree import Node
import numpy as np

import math
import random

'''
	Game Board Legend
		5 - invalid square
		0 - no coin
		-1 - player coin
		-2 - player king coin
		1 - opponent coin
		2 - opponent king coin
'''

class PlayingAgent:
	def __init__(self,gen_limit = 5000):
		self.tinsley_net = NeuralNetwork()
		self.tinsley_net.set_random_seed(1354)
		self.tinsley_net.randomize_weights()

		self.lindus_net = NeuralNetwork()
		self.lindus_net.set_random_seed(9854)
		self.lindus_net.randomize_weights()

		self.gen_limit = gen_limit

	def in_boundary(self,pos):
		if pos[0] < 0 or pos[0] > 7:
			return False
		if pos[1] < 0 or pos[1] > 7:
			return False
		return True

	def exec_move(self, sim_board, start, end):
		sim_board = np.array(sim_board)
		moving_coin = sim_board.item(start)
		sim_board[start] = 0
		sim_board[end] = moving_coin
		
		mov_vector = np.subtract(end,start)

		if abs(mov_vector[0]) == 2:
			cut_position = tuple(np.add(start,np.divide(mov_vector,2)))
			sim_board[cut_position] = 0

		return sim_board

	def get_possible_moves(self,simulated_board,x,y,mover):
		ret = []	#ret value for jump
		sur = []	#ret value for single cell hop

		if x>0 and y>0 and ((mover%2!=0) or simulated_board[x][y]==2):
			
			if simulated_board.item(x-1, y-1)==0:
				sur.append(( x-1, y-1 ))
			
			elif simulated_board.item(x-1, y-1)<0:
				target = (x-2,y-2)

				if self.in_boundary(target) and simulated_board.item(target) == 0:
					ret.append(target)
					
					sim_temp = self.exec_move(simulated_board,(x,y),target)
					temp_ret = self.get_possible_moves(sim_temp,target[0],target[1],mover)
					
					if temp_ret[1]!=None and len(temp_ret[1]) > 0:
						ret.append(temp_ret[1])
						
		if x<7 and y>0 and ((mover%2==0) or simulated_board[x][y]==2):
			if simulated_board.item(x+1, y-1)==0:
				sur.append(( x+1, y-1 ))
			elif simulated_board.item(x+1, y-1)<0:
				target = (x+2,y-2)

				if self.in_boundary(target) and simulated_board.item(target) == 0:
					ret.append(target)
					
					sim_temp = self.exec_move(simulated_board,(x,y),target)
					temp_ret = self.get_possible_moves(sim_temp,target[0],target[1],mover)
					
					if temp_ret[1]!=None and len(temp_ret[1]) > 0:
						ret.append(temp_ret[1])
					
		if x<7 and y<7 and ((mover%2==0) or simulated_board[x][y]==2):
			if simulated_board.item(x+1, y+1)==0:
				sur.append(( x+1, y+1 ))
			elif simulated_board.item(x+1, y+1)<0:
				target = (x+2,y+2)

				if self.in_boundary(target) and simulated_board.item(target) == 0:
					ret.append(target)
					
					sim_temp = self.exec_move(simulated_board,(x,y),target)
					temp_ret = self.get_possible_moves(sim_temp,target[0],target[1],mover)

					if ret!=None and len(temp_ret[1]) > 0:
						ret.append(temp_ret[1])
						
		if x>0 and y<7 and ((mover%2!=0) or simulated_board[x][y]==2):
			if simulated_board.item(x-1, y+1)==0:
				sur.append(( x-1, y+1 ))
			elif simulated_board.item(x-1, y+1)<0:
				target = (x-2,y+2)

				if self.in_boundary(target) and simulated_board.item(target) == 0:
					ret.append(target)

					sim_temp = self.exec_move(simulated_board,(x,y),target)
					temp_ret = self.get_possible_moves(sim_temp,target[0],target[1],mover)
					
					if temp_ret[1]!=None and len(temp_ret[1]) > 0:
						ret.append(temp_ret[1])
		# print [sur,ret]
		return [sur,ret]

	def get_jump_combinations(self, jumps):
		
		a = []

		for x in range(0,len(jumps)):
			
			if str(type(jumps[x])) == "<type 'list'>":
				v = a.pop()
				b = self.get_jump_combinations(jumps[x])

				for y in range(0,len(b)):
					if str(type(v)) == "<type 'tuple'>":
						if str(type(b[y])) == "<type 'tuple'>":
							b[y] = [ v, b[y] ]
						else:
							b[y] = [ v ] + b[y]
					else:
						if str(type(b[y])) == "<type 'tuple'>":
							b[y] = v.append(b[y])
						else:
							b[y] = v + b[y]
				a = a+b
			else:
				a.append(jumps[x])
		return a

	def get_optimum_move(self,tree,depth,max_depth):

		if depth == max_depth:
			return [ 0, [] ]

		score = 0
		index = []
		max = 0

		for x in range(0,len(tree.children)):
			ret = self.get_optimum_move(tree.children[x],depth+1,max_depth)
			score = tree.children[x].score - ret[0] 

			if x == 0:
				max = score
				index = [0]
			elif score > max:
				max = score
				index = [x]
			elif score == max:
				index.append(x)

		index = random.choice(index)

		return [ score, tree.children[index].moves ]

	def minmax(self, simulated_board, mover=0, depth=2):

		'''
			- global moves array(2D)
			- 2nd dimension is the current move
		'''

		self.Tree = Node(simulated_board = simulated_board)
		# current_node = self.Tree
		# root_node = self.Tree
		# current_node.simulated_board = simulated_board
		
		stack = []
		stack.append(self.Tree)

		while len(stack) > 0:

			current_node = stack.pop()
			simulated_board = current_node.simulated_board

			for x in range(0,8):
				for y in range(0,8):
					if simulated_board[x][y] != 5 and simulated_board[x][y] > 0:
						playing_coin = (x,y)

						moves = self.get_possible_moves(simulated_board,x,y,mover)
						jumps = moves[1]
						moves = moves[0]
						
						if len(jumps) > 0:
							moves = self.get_jump_combinations(jumps)

						for move in moves:
							temp_sim = np.array(simulated_board)	
							if str(type(move)) == "<type 'tuple'>":
								temp_sim = self.exec_move(simulated_board,(x,y),move)
								move = [move]
							else:
								source = (x,y)
								for z in move:
									temp_sim = self.exec_move(temp_sim,source,z)
									source = z
							move = [(x,y)] + move

							board_config = []
							
							for r in temp_sim:
								for c in r:
									if c != 5:
										board_config.append(c)
							if mover == 0:
								heuristic_cost = self.tinsley_net.get_heuristic_cost(board_config)
							else:
								heuristic_cost = self.lindus_net.get_heuristic_cost(board_config)
							
							self.node = Node(score=heuristic_cost, moves=move, simulated_board=temp_sim)
							self.node.set_parent(current_node)
							current_node.add_child(self.node)
			
			if current_node.depth < depth-1:
				for x in current_node.children:
					stack.append(x)
		
		return self.get_optimum_move(self.Tree,0,3)

		# return self.Tree 	

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

ai_agent = PlayingAgent()

ai_agent.minmax(board,0,3)
# print my_nn.exec_move(board,(2,0),(3,1))
# print ai_agent.get_possible_moves(board,0,0,2)

# jumps = [(2, 2), [(4, 0), [(6, 2)], (4, 4), [(6, 2), (6, 6)]]]

# print ai_agent.get_jump_combinations_1(jumps)