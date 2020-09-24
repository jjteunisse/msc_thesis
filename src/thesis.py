# Main script for the thesis experiments.
# Joris Teunisse, 2020

import agents
import matplotlib.colors as cols
import matplotlib.pyplot as plt
import numpy as np
import random as rd
import stage_games as sgs
import strategies as sts

def exp_full(Game, Candidates, Agent, params):
	# Conducts an experiment in which the agent identifies
	# an opponent from the full candidate set.
	result = run_exp(Game, Candidates, Agent, params, "full")

	print("Identified {} as {} in {} iterations.".format(
		result["true_opp_name"], result["id_opp_name"], result["n_iterations"]
	))

	for hst in result["prob_hists"]: plt.plot(hst)
	plt.title("{} vs {} for {} iterations of {}".format(
		Agent.Strategy.name, result["true_opp_name"],
		result["n_iterations"], type(Game).__name__
	))
	plt.legend([C.name for C in Candidates])
	plt.xlabel("Iteration")
	plt.ylabel("Probability")
	plt.show()

def exp_pairs(Game, Candidates, Agent, params, pair_plots):
	# Conducts an experiment in which the agent attempts to 
	# distinguish every possible pair in the candidate set.
	Candidate_Pairs = []
	data = {"means": [], "medians": [], "stds": [], "accs": []}

	ranges = {}
	ranges["means"] = [1.0, params["max_iterations"]]
	ranges["medians"] = [1.0, params["max_iterations"]]
	ranges["stds"] = [0.0, params["max_iterations"] / 2]
	ranges["accs"] = [0.0, 1.0]

	titles = {}
	titles["means"] = "{} (mean)".format(type(Game).__name__ )
	titles["medians"] = "{} (median)".format(type(Game).__name__ )
	titles["stds"] = "{} (sd)".format(type(Game).__name__ )
	titles["accs"] = "{} (accuracy)".format(type(Game).__name__ )

	for i, C1 in enumerate(Candidates):
		for j, C2 in enumerate(Candidates):
			if j > i: Candidate_Pairs.append([C1, C2])

	for Pair in Candidate_Pairs:
		failures = np.zeros(2)
		iteration_list = []
		for i in range(params["n_experiments"]):
			if not params["debug"]: print("Experiment #{}\r".format(i), end="")

			for C in Pair: C.reset()
			Agent.reset(Pair)

			result = run_exp(Game, Pair, Agent, params, "pairs")

			if result["true_opp_name"] == result["id_opp_name"]:
				iteration_list.append(result["n_iterations"])
			elif result["n_iterations"] == params["max_iterations"]: failures[0] += 1
			else: failures[1] += 1

		failures /= params["n_experiments"]

		if len(iteration_list) > 0:
			mean = np.mean(iteration_list)
			median = np.median(iteration_list)
			std = np.std(iteration_list)
			acc = len(iteration_list) / params["n_experiments"]

			data["means"].append(mean)
			data["medians"].append(median)
			data["stds"].append(std)
			data["accs"].append(acc)

			print(("{:11} vs {:11}: Mean {:6.2f}, Median {:6.2f}, SD {:6.2f}, "
				   "Accuracy {:.2f} (MI {:.2f}, WC {:.2f})").format(
				Pair[0].name, Pair[1].name, mean, median, std, acc, *failures
			))
		else:
			data["means"].append(-1)
			data["medians"].append(-1)
			data["stds"].append(-1)
			data["accs"].append(0)

			print("{:11} vs {:11}: N/A (MI {:.2f}, WC {:.2f})".format(
				Pair[0].name, Pair[1].name, *failures
			))

	if pair_plots["means"]:
		plot_matrix(Candidates, [d for d in data["means"]], ranges["means"], titles["means"])
	if pair_plots["medians"]:
		plot_matrix(Candidates, [d for d in data["medians"]], ranges["medians"], titles["medians"])
	if pair_plots["stds"]:
		plot_matrix(Candidates, [d for d in data["stds"]], ranges["stds"], titles["stds"])
	if pair_plots["accs"]:
		plot_matrix(Candidates, [d for d in data["accs"]], ranges["accs"], titles["accs"])

	return data

def plot_matrix(Candidates, data_row, _range, title):
	# Plots the given data row as a matrix.
	with open('experiments/txt/' + title + '.txt', "w") as f:
		f.write(" ".join([str(d) for d in data_row]))

	matrix = -np.ones((len(Candidates), len(Candidates)))
	for i in range(len(Candidates)):
		for j in range(len(Candidates)):
			if j > i:
				matrix[i][j] = data_row.pop(0)
				if matrix[i][j] != -1:
					if title == "All games (failures)":
						plt.text(j, i, int(matrix[i][j]),
								 horizontalalignment='center',
								 verticalalignment='center')
					else:
						plt.text(j, i, "{:.2f}".format(matrix[i][j]),
								 horizontalalignment='center',
								 verticalalignment='center')

				else:
					plt.text(j, i, "N/A",
							 horizontalalignment='center',
							 verticalalignment='center')

	col_list = None
	if "accuracy" in title: col_list = ["red", "orange", "yellow", "lime", "cyan"]
	else: col_list = ["cyan", "lime", "yellow", "orange", "red"]
	cmap = cols.LinearSegmentedColormap.from_list("", col_list)
	cmap.set_under("white")

	plt.title(title)
	plt.imshow(matrix, cmap=cmap, vmin=(_range[0] - 1E-5), vmax=(_range[1] + 1E-5))
	plt.xticks(range(len(Candidates)), [C.name for C in Candidates], rotation='vertical')
	plt.yticks(range(len(Candidates)), [C.name for C in Candidates], rotation='horizontal')
	plt.savefig('experiments/pdf/' + title + '.pdf', bbox_inches='tight')
	plt.close()

def run_exp(Game, Candidates, Agent, params, exp_type):
	# Runs a single experiment with the given settings.
	np.set_printoptions(floatmode="fixed", precision=3, suppress=True)

	id_opp_name = None
	iters_left = params["max_iterations"]
	n_iterations = 0
	prob_hists = [[1.0 / len(Candidates)] for _ in Candidates]

	Opponent = Candidates[rd.randrange(len(Candidates))]

	if params["debug"]:
		print("Candidates: {}".format([C.name for C in Candidates]))
		print("Opponent: {}".format(Opponent.name))

	while n_iterations < params["max_iterations"]:
		action_pair = (Agent.take_action(iters_left), Opponent.take_action())
		action_probs = Agent.get_action_probs()
		Agent.update_c_probs(action_pair, action_probs)
		id_opp_name = Agent.get_opp_name()

		iters_left -= 1
		n_iterations += 1

		if params["debug"]:
			print("[{:03}] {} -> {}".format(n_iterations, action_pair, Agent.c_probs))

		if exp_type == "full":
			print("Iterations: {}\r".format(n_iterations), end="")
			for i, hst in enumerate(prob_hists): hst.append(Agent.c_probs[i])

		if id_opp_name != None: break

		Agent.Strategy.update(action_pair)
		for C in Candidates: C.update(action_pair)

	return {"id_opp_name": id_opp_name,
			"n_iterations": n_iterations,
			"prob_hists": prob_hists,
			"true_opp_name": Opponent.name}

if __name__ == "__main__":
	# Sets required parameters and runs the experiments.
	agent_params = {"lookahead": False, "max_depth": 1, "max_sim_iters": 10,
					"n_samples": 100, "n_sims": 100, "prob_threshold": 0.99}
	exp_params = {"debug": False, "max_iterations": 10, "n_experiments": 100}
	fp_params = {"epsilon": 0.01}
	q_params = {"alpha": 0.1, "decay": 1E-5, "epsilon": 0.2, "gamma": 0.9, "temp": 1}
	pair_plots = {"means": True, "medians": False, "stds": False, "accs": True}

	Games = [
		sgs.Bach_or_Stravinsky(),
		sgs.Chicken(),
		sgs.Coordinate_First(),
		sgs.Coordination(),
		sgs.Deadlock(),
		sgs.Matching_Pennies(),
		sgs.Prisoners_Dilemma(),
		sgs.Stag_Hunt()
	]

	Plot_Candidates = None
	mean_sum = np.array([0.0 for _ in range(36)])
	median_sum = np.array([0.0 for _ in range(36)])
	std_sum = np.array([0.0 for _ in range(36)])
	acc_sum = np.array([0.0 for _ in range(36)])
	n_na = [0 for _ in range(36)]
	avg_game_means = []
	avg_game_accs = []
	na_per_game = []

	for Game in Games:
		Candidates = [
			sts.Pure_0(), sts.Pure_1(), sts.Random(),
			sts.Nash(1, Game.nash),
			sts.TFT(1, Game.pareto),
			sts.Epsilon_FP(1, Game.rewards, fp_params),
			sts.Relative_FP(1, Game.rewards, fp_params),
			sts.Boltz_Q(1, Game.rewards, q_params),
			sts.EGreedy_Q(1, Game.rewards, q_params)
		]
		Strategy = sts.Random()
		Agent = agents.Empirical_Agent(Candidates, Strategy, agent_params)

		print("Description: {} applying {} in {}.".format(
			type(Agent).__name__, Agent.Strategy.name, type(Game).__name__
		))
		print("Legend: MI = Max Iterations, WC = Wrong Candidate.")
		print("Params:")
		print("- Agent", agent_params)
		print("- Experiment", exp_params)
		print("- FP", fp_params)
		print("- Q", q_params)
		print()

		# exp_full(Game, Candidates, Agent, exp_params)
		data = exp_pairs(Game, Candidates, Agent, exp_params, pair_plots)

		na_this_game = 0
		if Plot_Candidates == None: Plot_Candidates = [C for C in Candidates]
		for i in range(36):
			if data["means"][i] != -1: mean_sum[i] += data["means"][i]
			else:
				n_na[i] += 1
				na_this_game += 1

			if data["medians"][i] != -1: median_sum[i] += data["medians"][i]
			if data["stds"][i] != -1: std_sum[i] += data["stds"][i]

			acc_sum[i] += data["accs"][i]
		avg_game_means.append(np.mean([d for d in data["means"] if d != -1]))
		avg_game_accs.append(np.mean(data["accs"]))
		na_per_game.append(na_this_game)

	ranges = {}
	ranges["means"] = [1.0, exp_params["max_iterations"]]
	ranges["medians"] = [1.0, exp_params["max_iterations"]]
	ranges["stds"] = [0.0, exp_params["max_iterations"] / 2]
	ranges["accs"] = [0.0, 1.0]

	for i in range(36):
		mean_sum[i] /= len(Games) - n_na[i]
		median_sum[i] /= len(Games) - n_na[i]
		std_sum[i] /= len(Games) - n_na[i]
		acc_sum[i] /= len(Games)

	plot_matrix(Plot_Candidates, list(mean_sum), ranges["means"], "All games (mean)")
	plot_matrix(Plot_Candidates, list(median_sum), ranges["medians"], "All games (median)")
	plot_matrix(Plot_Candidates, list(std_sum), ranges["stds"], "All games (sd)")
	plot_matrix(Plot_Candidates, list(acc_sum), ranges["accs"], "All games (accuracy)")
	plot_matrix(Plot_Candidates, n_na, [min(n_na), max(n_na)], "All games (failures)")

	with open("experiments/txt/stats.txt", "w") as f:
		f.write("avg_game_means: " + " ".join(["{:.2f}".format(m) for m in avg_game_means]) + "\n")
		f.write("avg_game_accs: " + " ".join(["{:.2f}".format(a) for a in avg_game_accs]) + "\n")
		f.write("na_per_game: " + " ".join([str(num) for num in na_per_game]))