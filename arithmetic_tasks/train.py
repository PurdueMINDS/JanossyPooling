import torch
import torch.nn as nn
from models import TextModels
import argparse
import sys
import tqdm
import numpy as np
from itertools import combinations
import time
import random
from torchvision.datasets import MNIST
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json

NUM_TRAINING_EXAMPLES = 100000
NUM_TEST_EXAMPLES = 10000
NUM_VALIDATION_EXAMPLES = 10000
NUM_EPOCHS_JANOSSY = 1000
NUM_EPOCHS_RNN = 1000
BASE_EMBEDDING_DIMENSION = 100
INFERENCE_PERMUTATIONS = 20

supported_tasks = {'range': {'vocab_size': 100, 'sequence_length': 5},
				   'sum': {'vocab_size': 100, 'sequence_length': 5},
				   'unique_sum': {'vocab_size': 10, 'sequence_length': 10},
				   'unique_count': {'vocab_size': 10, 'sequence_length': 10},
				   'variance': {'vocab_size': 100, 'sequence_length': 10},'stddev': {'vocab_size': 100, 'sequence_length': 10}}
supported_nature = ['image', 'text']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
output_dict = {"accuracy":[],"mae":[],"mse":[],"1_inf_accuracy":[],"1_inf_mae":[],"1_inf_mse":[]}

# Based on task selection create train and test data
def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-t', '--task', help='Specify the arthimetic task', required=True)
	parser.add_argument('-m', '--model', help='Model to get accuracy scores for', default='janossy_2ary')
	parser.add_argument('-i', '--iterations',
						help='Number of iterations to run the task and model for - Confidence Interval tasks',
						default=1, type=int)
	parser.add_argument('-l', '--hidden_layers', help='Number of hidden layers in rho MLP', default=1, type=int)
	parser.add_argument('-n', '--neurons', help='Number of neurons in each hidden layer in the rho MLP', default=100,
						type=int)
	parser.add_argument('--type', help='Specify if the task involves text or images', default='text')
	parser.add_argument('-lr', '--learning_rate', help='Specify learning rate', default=0.0001, type=float)
	parser.add_argument('-b', '--batch_size', help='Specify batch size', default=128, type=int)
	# Will enable sweeps using master script
	# parser.add_argument('-hl',help='Enable Parameter Sweep over learning rate',action='store_true')
	# parser.add_argument('-hn',help='Enable Parameter Sweep over number of neurons',action='store_true')
	# parser.add_argument('-hh',help='Enable Parameter Sweep over number of hidden layers in rho',action='store_true')
	args = parser.parse_args()
	return args


def valid_argument_check(task, nature, model):
	if task not in list(supported_tasks.keys()):
		print("Specified Task %s not supported" % task)
		sys.exit()

	if nature not in supported_nature:
		print("Specified Type %s not supported" % nature)
		sys.exit()

	janossy_k = 1
	if model not in ['deepsets', 'lstm', 'gru']:
		if 'janossy' == model[0:7]:
			try:
				janossy_k = int(model.split('_')[1].split('ary')[0])
			except:
				print(
					"Specified format incorrect for Janossy model. Model format is janossy_kary where  k in an integer")
				sys.exit()
		else:
			print(
				"Model specified not available. Models should be selected from 'deepsets','lstm','gru','janossy_kary'")
			sys.exit()
	return janossy_k


def construct_task_specific_output(task, input_sequence):
	if task == 'range':
		return np.max(input_sequence) - np.min(input_sequence)
	if task == 'sum':
		return np.sum(input_sequence)
	if task == 'unique_sum':
		return np.sum(np.unique(input_sequence))
	if task == 'unique_count':
		return np.size(np.unique(input_sequence))
	if task == 'variance':
		return np.var(input_sequence)
	if task == 'stddev':
		return np.std(input_sequence)


def janossy_text_input_construction(X, janossy_k):
	X_janossy = []
	for index in range(len(X)):
		temp = list(X[index])
		temp = [int(x) for x in temp]
		temp.sort()
		temp = list(combinations(temp, janossy_k))
		temp = [list(x) for x in temp]
		X_janossy.append(temp)
	return np.array(X_janossy)


def text_dataset_construction(train_or_test, janossy_k, task):
	""" Data Generation """
	if train_or_test:
		num_examples = NUM_TRAINING_EXAMPLES
	else:
		num_examples = NUM_TEST_EXAMPLES

	train_length = int(supported_tasks[task]['sequence_length'])
	vocab_size = int(supported_tasks[task]['vocab_size'])
	X = np.zeros((num_examples, train_length))
	output_X = np.zeros((num_examples, 1))
	for i in tqdm.tqdm(range(num_examples), desc='Generating Training / Validation / Test Examples: '):
		for j in range(train_length):
			X[i, j] = np.random.randint(0, vocab_size)
			output_X[i, 0] = construct_task_specific_output(task, X[i])
	if janossy_k == 1:
		return X, output_X
	else:
		# Create Janossy Input
		X_janossy = janossy_text_input_construction(X, janossy_k)
		return X_janossy, output_X


def image_dataset_construction(train_or_test, janossy_k):
	"""Image Data Generation"""
	pass


def determine_vocab_size(task):
	return supported_tasks[task]['vocab_size']


def permute(x):
	return np.random.permutation(x)


def func(x):
	unique, counts = np.unique(ar=x, return_counts=True)
	return unique[np.argmax(counts)]


def unison_shuffled(a, b):
	assert len(a) == len(b)
	p = np.random.permutation(len(a))
	return a[p], b[p]


def train_text(vocab_size, input_dim, task, model, num_layers, num_neurons, janossy_k, learning_rate,batch_size,iteration):
	# Construct vocab size base on model
	janossy_model = TextModels(vocab_size, input_dim, model, num_layers, num_neurons, janossy_k, device)
	janossy_model.to(device)
	X, output_X = text_dataset_construction(1, janossy_k, task)
	V, output_V = text_dataset_construction(0, janossy_k, task)
	Y, output_Y = text_dataset_construction(0, janossy_k, task)
	# model.train
	if model in ['lstm', 'gru']:
		num_epochs = NUM_EPOCHS_RNN
	else:
		num_epochs = NUM_EPOCHS_JANOSSY
	checkpoint_file_name = str(model) + "_" + str(task) +"_" + str(num_layers) + "_" + "iteration" + str(iteration) + "_" + "learning_rate_" + str(learning_rate) + "_batch_size_" + str(batch_size) + "_checkpoint.pth.tar"
	# Use Adam Optimizer on all parameters with requires grad as true
	optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, janossy_model.parameters()), lr=learning_rate)
	# Train over multiple epochs
	start_time = time.time()
	num_batches = int(NUM_TRAINING_EXAMPLES / batch_size)
	best_val_accuracy = 0.0
	for epoch in range(num_epochs):
		# Do seed and random shuffle of the input
		X, output_X = unison_shuffled(X, output_X)
		#Performing pi-SGD for RNN's
		if model in ['lstm','gru']:
			X = np.apply_along_axis(permute, 1, X)
		for batch in range(num_batches):
			batch_seq = torch.LongTensor(X[batch_size * batch:batch_size * batch + batch_size]).to(device)
			optimizer.zero_grad()
			loss = janossy_model.loss(batch_seq, torch.FloatTensor(output_X[np.array(range(batch_size * batch, batch_size * batch + batch_size))]).to(device))
			loss.backward()
			optimizer.step()
		with torch.no_grad():
			val_output = np.round(janossy_model.forward(torch.LongTensor(V).to(device)).data.cpu().numpy())
			val_loss = janossy_model.loss(torch.LongTensor(V).to(device),torch.FloatTensor(output_V).to(device))
			val_correct = 0
			for j in range(len(output_V)):
				if output_V[j,0] == val_output[j,0]:
					val_correct+=1
			val_accuracy = (1.0*val_correct)/len(output_V)
			if val_accuracy >= best_val_accuracy:
				best_val_accuracy = val_accuracy
				#Save Weights
				torch.save(janossy_model.state_dict(),checkpoint_file_name)	
		print(epoch, loss.data[0],val_loss.data[0])
	end_time = time.time()
	total_training_time = end_time - start_time
	print("Total Training Time: ", total_training_time)

	# model.eval
	inference_output = np.zeros((NUM_TEST_EXAMPLES, 1))
	with torch.no_grad():
		test_output = np.round(janossy_model.forward(torch.LongTensor(Y).to(device)).data.cpu().numpy())
		inference_output = test_output
		if model == 'lstm' or model == 'gru':
			#Print 1 inference output as well
			print(model," Model 1 Inference Scores")
			correct = 0
			for j in range(len(output_Y)):
				if output_Y[j, 0] == inference_output[j, 0]:
					correct += 1
			acc =  1.0 * correct / len(output_Y)
			mae = mean_absolute_error(output_Y,inference_output)
			mse = mean_squared_error(output_Y,inference_output)
			print("1 inf Accuracy :", acc)
			print("1 inf Mean Absolute Error: ",mae)
			print("1 inf Mean Squared Error: ",mse)
			output_dict["1_inf_accuracy"].append(acc)
			output_dict["1_inf_mae"].append(mae)
			output_dict["1_inf_mse"].append(mse)
			# Inference is performed since we do not sample all permutations
			for inference_step in range(INFERENCE_PERMUTATIONS - 1):
				temp_Y = np.apply_along_axis(permute, 1, Y)
				temp_pred = janossy_model.forward(torch.LongTensor(temp_Y).to(device)).data.cpu().numpy()
				inference_output = np.column_stack((inference_output, np.round(temp_pred)))
			inference_output = np.apply_along_axis(func, 1, inference_output)
			inference_output = np.reshape(inference_output,(-1,1))
	correct = 0
	for j in range(len(output_Y)):
		if output_Y[j, 0] == inference_output[j, 0]:
			correct += 1
	acc =  1.0 * correct / len(output_Y)
	mae = mean_absolute_error(output_Y,inference_output)
	mse = mean_squared_error(output_Y,inference_output)
	print("Accuracy :", acc)
	print("Mean Absolute Error: ",mae)
	print("Mean Squared Error: ",mse)
	output_dict["accuracy"].append(acc)
	output_dict["mae"].append(mae)
	output_dict["mse"].append(mse)




def main():
	# if iteration more than 1 present with stddev as well over the runs
	args = parse_args()
	batch_size = args.batch_size
	task = str(args.task).lower()
	nature = str(args.type).lower()
	model = str(args.model).lower()
	num_iterations = int(args.iterations)
	num_neurons = args.neurons
	num_layers = args.hidden_layers
	learning_rate = args.learning_rate
	janossy_k = valid_argument_check(task, nature, model)
	vocabulary_size = determine_vocab_size(task)
	output_file_name = str(model) + "_" + str(task) + "_" + str(num_layers) + "_" + str(args.learning_rate) + "_" + str(batch_size) + ".txt"
	for iteration in range(num_iterations) :
		train_text(vocabulary_size, BASE_EMBEDDING_DIMENSION, task, model, num_layers, num_neurons, janossy_k, learning_rate,batch_size,iteration)
		with open(output_file_name,'w') as file :
			file.write(json.dumps(output_dict))

if __name__ == '__main__':
	main()
