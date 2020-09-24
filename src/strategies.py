# Defines the strategies used in the thesis experiments.
# Joris Teunisse, 2020

import numpy as np
import random as rd

class Strategy:
	# Base strategy functionality: cannot be used as-is.
	def __init__(self, p1):
		if type(self) == Strategy: raise TypeError()
		self.p1 = p1
		self.name = type(self).__name__

	def reset(self):
		pass

	def take_action(self):
		return 1 if rd.random() < self.p1 else 0

	def update(self, action_pair):
		pass

class Pure_0(Strategy):
	# Always plays action 0.
	def __init__(self):
		super().__init__(0)

class Pure_1(Strategy):
	# Always plays action 1.
	def __init__(self):
		super().__init__(1)

class Random(Strategy):
	# Plays either action with equal probabilities.
	def __init__(self):
		super().__init__(0.5)

class Nash(Strategy):
	# Consistently plays its part of a random Nash equilibrium.
	def __init__(self, _id, nash):
		super().__init__(rd.choice(nash)[_id])
		self._id = _id
		self.nash = nash

	def reset(self):
		self.p1 = rd.choice(self.nash)[self._id]

class TFT(Strategy):
	# Initially plays its part of a random Pareto optimum:
	# mimics the action of its opponent afterwards.
	def __init__(self, _id, pareto):
		super().__init__(rd.choice(pareto)[_id])
		self._id = _id
		self.pareto = pareto

	def reset(self):
		self.p1 = rd.choice(self.pareto)[self._id]

	def update(self, action_pair):
		self.p1 = action_pair[self._id ^ 1]

class FP(Strategy):
	# Base Fictitious Play functionality: cannot be used as-is.
	def __init__(self, _id, rewards, params):
		if type(self) == FP: raise TypeError()
		super().__init__(0.5)
		self._id = _id
		self.rewards = rewards
		self.params = params
		self.action_rewards = np.array([
			[rewards[(0, 0)][_id], rewards[(_id, _id ^ 1)][_id]],
			[rewards[(_id ^ 1, _id)][_id], rewards[(1, 1)][_id]]
		])
		self.history = []

	def av_to_action(self, action_values):
		pass

	def reset(self):
		self.history = []
		self.p1 = 0.5

	def update(self, action_pair):
		self.history.append(action_pair[self._id ^ 1])

		opp_p1 = sum(self.history) / len(self.history)
		opp_probs = [1.0 - opp_p1, opp_p1]
		action_values = [sum(ar * opp_probs) for ar in self.action_rewards]

		self.p1 = self.av_to_action(action_values)

class Epsilon_FP(FP):
	# Plays the best response to the opponent's history of play:
	# a random action is played if the actions have similar values.
	def av_to_action(self, action_values):
		if abs(action_values[0] - action_values[1]) < self.params["epsilon"]: return 0.5
		else: return 0 if action_values[0] > action_values[1] else 1

class Relative_FP(FP):
	# The response to the opponent's history of play is
	# proportional to the action values.
	def av_to_action(self, action_values):
		if min(action_values) < 0:
			for av in action_values: av += min(action_values)
		if sum(action_values) > 0: return action_values[1] / sum(action_values)
		else: return 0.5

class Q(Strategy):
	# Base Q-learning functionality: cannot be used as-is.
	def __init__(self, _id, rewards, params):
		if type(self) == Q: raise TypeError()
		super().__init__(0.5)
		self._id = _id
		self.rewards = rewards
		self.params = params
		self.q_table = np.array([0.0, 0.0])

	def q_to_action(self):
		pass

	def reset(self):
		self.p1 = 0.5
		self.q_table = np.array([0.0, 0.0])

	def update(self, action_pair):
		action = action_pair[self._id]
		reward = self.rewards[action_pair][self._id]
		future_reward = reward + self.params["gamma"] * max(self.q_table)

		self.q_table[action] *= 1 - self.params["alpha"]
		self.q_table[action] += self.params["alpha"] * future_reward

		self.p1 = self.q_to_action()

class Boltz_Q(Q):
	# The action probabilities are a function of their Q-values:
	# the exploration rate is guided by a decaying temperature parameter.
	def __init__(self, _id, rewards, params):
		super().__init__(_id, rewards, params)
		self.curr_temp = params["temp"]

	def q_to_action(self):
		exp_q = np.exp(self.q_table / self.curr_temp)
		self.curr_temp *= (1 - self.params["decay"])
		return exp_q[1] / sum(exp_q)

	def reset(self):
		super().reset()
		self.curr_temp = self.params["temp"]

class EGreedy_Q(Q):
	# Plays a random action with probability epsilon:
	# otherwise, plays the action with the maximum Q-value.
	def q_to_action(self):
		if self.q_table[0] == self.q_table[1]: return 0.5
		else: return int(self.q_table[0] < self.q_table[1])

	def take_action(self):
		if rd.random() < self.params["epsilon"]: return rd.randrange(2)
		else: return super().take_action()