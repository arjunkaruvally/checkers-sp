import numpy as np

# Class for tree creation
class Node(object):
	def __init__(self, score=0, current_player=True, moves=[], simulated_board=[]):
		self.score = score
		self.moves = moves
		self.children = []
		self.parent = None
		self.simulated_board = np.array(simulated_board)
		self.depth = 0
		self.optimum_score = 0
		self.current_player = current_player

	def add_child(self, obj):
		obj.depth = self.depth+1
		obj.current_player = not self.current_player
		# if obj.depth%2 == 0:
		# 	obj.score = -1*obj.score
		self.children.append(obj)

	def set_parent(self,obj):
		self.parent = obj

	def postorder(self,root):
		if len(root.children) <= 0:
			return

		for x in root.children:
			self.postorder(x)
			print "----------------------"
			print x.depth
			print x.moves
			print x.score
			print x.simulated_board

			# f = open('mini_tree')
			# f.write('\n-----------------------\n')
			# f.write(str(x.depth)+'\n')


		return