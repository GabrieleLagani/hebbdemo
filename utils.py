import os
import csv
import matplotlib.pyplot as plt
import torch

# Return formatted string with time information
def format_time(seconds):
	seconds = int(seconds)
	minutes, seconds = divmod(seconds, 60)
	hours, minutes = divmod(minutes, 60)
	return str(hours) + "h " + str(minutes) + "m " + str(seconds) + "s"

def update_param_stats(param_stats, new_stats):
	for n, s in new_stats.items():
		if n + '.max' not in param_stats: param_stats[n + '.max'] = []
		param_stats[n + '.max'].append(torch.max(s.abs()).item())
		if n + '.nrm' not in param_stats: param_stats[n + '.nrm'] = []
		param_stats[n + '.nrm'].append(torch.sum(s ** 2).item())
	return param_stats

# Save data to csv file
def update_csv(results, path):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, mode='w', newline='') as csv_file:
		writer = csv.writer(csv_file)
		for name, entries in results.items():
			writer.writerow([name + '_step'] + list(entries.keys() if isinstance(entries, dict) else range(1, len(entries) + 1)))
			writer.writerow([name] + list(entries.values() if isinstance(entries, dict) else entries))

# Save a figure showing time series plots in the specified file
def save_plot(data_dicts, path, xlabel='step', ylabel='result'):
	graph = plt.axes(xlabel=xlabel, ylabel=ylabel)
	for key, data_dict in data_dicts.items():
		graph.plot(list(data_dict.keys()) if isinstance(data_dict, dict) else range(1, len(data_dict) + 1),
		           list(data_dict.values()) if isinstance(data_dict, dict) else data_dict,
		           label=str(key))
	graph.grid(True)
	graph.legend()
	os.makedirs(os.path.dirname(path), exist_ok=True)
	fig = graph.get_figure()
	fig.savefig(path, bbox_inches='tight')
	plt.close(fig)

# Save a grid of figures showing time series plots in the specified file
def save_grid_plot(data_dicts, path, rows=8, cols=8, xlabel='step', ylabel='result'):
	fig, graphs = plt.subplots(rows, cols, figsize=(48, 12))
	i, j = 0, 0
	for key, data_dict in data_dicts.items():
		graph = graphs[i][j]
		graph.plot(list(data_dict.keys()) if isinstance(data_dict, dict) else range(1, len(data_dict) + 1),
		           list(data_dict.values()) if isinstance(data_dict, dict) else data_dict,
		           label=str(key))
		graph.set_xlabel(xlabel)
		graph.set_ylabel(ylabel)
		graph.grid(True)
		graph.legend()
		i += 1
		if i == rows:
			i = 0
			j += 1
	os.makedirs(os.path.dirname(path), exist_ok=True)
	fig.savefig(path, bbox_inches='tight')
	plt.close(fig)

# Save state dictionary file to specified path
def save_dict(state_dict, path):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	torch.save(state_dict, path)
	
# Load state dictionary file from specified path
def load_dict(path, device='cpu'):
	return torch.load(path, map_location=device)

