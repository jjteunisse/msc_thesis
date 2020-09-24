# Defines the stage games used in the thesis experiments.
# Joris Teunisse, 2020

import numpy as np

class Bach_or_Stravinsky:
	def __init__(self):
		self.rewards = np.array([
			[[3, 2], [0, 0]],
			[[0, 0], [2, 3]]
		])
		self.nash = [(0, 0), (1, 1), (0.4, 0.6)]
		self.pareto = [(0, 0), (1, 1)]

class Chicken:
	def __init__(self):
		self.rewards = np.array([
			[[0, 0], [-1, 1]],
			[[1, -1], [-9, -9]]
		])
		self.nash = [(0, 1), (1, 0), (1/9, 1/9)]
		self.pareto = [(0, 0), (0, 1), (1, 0)]

class Coordinate_First:
	def __init__(self):
		self.rewards = np.array([
			[[1, 1], [0, 0]],
			[[0, 0], [0, 0]]
		])
		self.nash = [(0, 0)]
		self.pareto = [(0, 0)]

class Coordination:
	def __init__(self):
		self.rewards = np.array([
			[[2, 2], [0, 0]],
			[[0, 0], [1, 1]]
		])
		self.nash = [(0, 0), (1, 1)]
		self.pareto = [(0, 0)]

class Deadlock:
	def __init__(self):
		self.rewards = np.array([
			[[1, 1], [0, 3]],
			[[3, 0], [2, 2]]
		])
		self.nash = [(1, 1)]
		self.pareto = [(0, 1), (1, 0), (1, 1)]

class Matching_Pennies:
	def __init__(self):
		self.rewards = np.array([
			[[1, -1], [-1, 1]],
			[[-1, 1], [1, -1]]
		])
		self.nash = [(0.5, 0.5)]
		self.pareto = [(0, 0), (0, 1), (1, 0), (1, 1)]

class Prisoners_Dilemma:
	def __init__(self): 
		self.rewards = np.array([
			[[3, 3], [0, 5]],
			[[5, 0], [1, 1]]
		])
		self.nash = [(1, 1)]
		self.pareto = [(0, 0), (0, 1), (1, 0)]

class Stag_Hunt:
	def __init__(self):
		self.rewards = np.array([
			[[4, 4], [1, 3]],
			[[3, 1], [2, 2]]
		])
		self.nash = [(0, 0), (1, 1), (0.5, 0.5)]
		self.pareto = [(0, 0)]
