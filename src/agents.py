# Defines the agents used in the thesis experiments.
# Joris Teunisse, 2020

from copy import deepcopy
import numpy as np

class Agent():
	# Base agent functionality: cannot be used as-is.
	def __init__(self, Candidates, Strategy, params):
		if type(self) == Agent: raise TypeError()
		self.Candidates = Candidates
		self.Strategy = Strategy
		self.params = params
		self.c_probs = np.ones(len(Candidates)) / len(Candidates)
		self.ones = np.ones_like(self.c_probs)

	def get_action_probs(self):
		pass

	def get_opp_name(self):
		# Return the name of the identified opponent, if any.
		for i, C in enumerate(self.Candidates):
			if self.c_probs[i] >= self.params["prob_threshold"]: return C.name

	def look_ahead(self, cid, depth, max_sim_iters):
		# Calculate the effects of future action pairs up until a certain depth.
		action_probs = self.get_action_probs()
		action_scores = [0.0, 0.0]

		for agent_action in range(2):
			for opp_action in range(2):
				opp_action_prob = action_probs[cid] if opp_action == 1 else 1.0 - action_probs[cid]
				if opp_action_prob != 0: # Disregard invalid branches.
					My_Copy = deepcopy(self)

					action_pair = (agent_action, opp_action)
					My_Copy.update_c_probs(action_pair, action_probs)

					if My_Copy.get_opp_name() == None:

						My_Copy.Strategy.update(action_pair)
						for C in My_Copy.Candidates: C.update(action_pair)

						if depth < self.params["max_depth"]:
							next_action_scores = My_Copy.look_ahead(cid, depth + 1, max_sim_iters - 1)
							best_action_score = min(next_action_scores) + 1
							action_scores[agent_action] += best_action_score * opp_action_prob
						else:
							max_sim_iters = min(max_sim_iters, self.params["max_sim_iters"])
							total_sim_iters = 0
							for _ in range(self.params["n_sims"]):
								total_sim_iters += My_Copy.simulate(cid, max_sim_iters)
							mean_sim_iters = total_sim_iters / self.params["n_sims"]
							action_scores[agent_action] += mean_sim_iters * opp_action_prob

		return action_scores

	def reset(self, Candidates):
		# Reset the variables which are changed over time.
		self.Candidates = Candidates
		self.c_probs = np.ones(len(Candidates)) / len(Candidates)
		self.ones = np.ones_like(self.c_probs)
		self.Strategy.reset()

	def simulate(self, cid, max_sim_iters):
		# Apply playouts until a maximum has been reached or
		# the opponent has been identified.
		My_Copy = deepcopy(self)
		n_sim_iters = 0

		while n_sim_iters < max_sim_iters:
			n_sim_iters += 1

			action_pair = (My_Copy.Strategy.take_action(),
						   My_Copy.Candidates[cid].take_action())
			action_probs = My_Copy.get_action_probs()
			My_Copy.update_c_probs(action_pair, action_probs)

			if My_Copy.get_opp_name() != None: break

			My_Copy.Strategy.update(action_pair)
			for C in My_Copy.Candidates: C.update(action_pair)

		return n_sim_iters

	def take_action(self, iters_left):
		# Take an action according to either a lookahead policy based
		# upon the agent's strategy, or the strategy itself.
		# Note: has only been tested with the Random strategy.
		if self.params["lookahead"]:
			action_score_lists = [[], []]

			for cid in range(len(self.Candidates)):
				action_scores = self.look_ahead(cid, 1, iters_left)
				action_score_lists[0].append(action_scores[0])
				action_score_lists[1].append(action_scores[1])

			return int(sum(action_score_lists[0]) > sum(action_score_lists[1]))

		else: return self.Strategy.take_action()

	def update_c_probs(self, action_pair, action_probs):
		# Update the probability of each candidate.
		# Note: assumes the opponent's _id == 1.
		if action_pair[1] == 1: self.c_probs *= action_probs
		else: self.c_probs *= self.ones - action_probs

		# Normalise the probability of each candidate.
		if sum(self.c_probs) > 0: self.c_probs /= sum(self.c_probs)

class Clairvoyant_Agent(Agent):
	# Agent which has access to the true action probabilities.
	# Note: performs suboptimally against EGreedy_Q due to the random component.
	def get_action_probs(self):
		return [C.p1 for C in self.Candidates]

class Empirical_Agent(Agent):
	# Agent which approximates the action probabilities by sampling
	# the actions of each candidate strategy.
	def get_action_probs(self):
		action_probs = []
		for C in self.Candidates:
			action_sum = 0.0
			for _ in range(self.params["n_samples"]): action_sum += C.take_action()
			action_probs.append(action_sum / self.params["n_samples"])
		return action_probs