from grader import EssayGrader
from essay import Essay
from setup import setup_env
import argparse
import os
from utils import load_index, store_results, evaluate_accuracy, write_essays_tocsv
import pickle as pkl
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression



# Function to label examples in input directory and store results in output directory
def label_eval_test(input_dir, output_dir, model_dir, evaluate):
	test_essays_dir = os.path.join(input_dir, TEST_DIR, ESSAY_DIR)
	test_idx_filepath = os.path.join(input_dir, TEST_DIR, IDX_FILE)

	print("Loading index...")
	index = load_index(test_idx_filepath)

	# Annotate essays
	print("Labelling essays...")
	essays = []
	ground_truth = []
	for idx in tqdm(range(len(index))):
		record = index[idx]
		filename = record['filename']
		prompt = record['prompt']
		grade = record['grade'] if 'grade' in record else None
		
		# Get Annotations
		filepath = os.path.join(test_essays_dir, filename)
		e = Essay(filepath, prompt, grade)
		essays.append(e)

		if evaluate:
			ground_truth.append(record['grade'])

	print("Grading essays...")
	results = []
	grader = EssayGrader(model_dir)
	for idx in tqdm(range(len(index))):
		e = essays[idx]
		# Grade, augment and store
		result = grader.grade(e)
		result['filename'] = index[idx]['filename']
		results.append(result)

	out_filepath = os.path.join(output_dir, OUT_FILE)
	store_results(results, out_filepath)

	if evaluate:
		predictions = [r['grade'] for r in results]
		accuracy = evaluate_accuracy(ground_truth, predictions)
		print("Accuracy: {0:.2f}%".format(accuracy * 100))


def train(input_dir, output_dir, model_dir):
	# Housekeeping
	train_essays_dir = os.path.join(input_dir, TRAIN_DIR, ESSAY_DIR)
	train_idx_filepath = os.path.join(input_dir, TRAIN_DIR, IDX_FILE)

	# Load the index file
	print("Loading index...")
	index = load_index(train_idx_filepath)

	# Annotate essays
	print("Labelling essays...")
	essays = []
	for idx in tqdm(range(len(index))):
		record = index[idx]
		filename = record['filename']
		prompt = record['prompt']
		grade = record['grade']

		# Get Annotations
		filepath = os.path.join(train_essays_dir, filename)
		e = Essay(filepath, prompt, grade)
		essays.append(e)

	with open('essays.pkl', 'wb') as fout:
		pkl.dump(essays, fout)


	# Grade essays
	print("Grading essays...")
	results = []
	grader = EssayGrader(model_dir)
	for idx in tqdm(range(len(essays))):
		e = essays[idx]
		result = grader.grade(e)
		result['filename'] = index[idx]['filename']
		results.append(result)

	# Saving subject verb agreement probabilities
	print("Computing subject verb agreement probabilities...")
	probs = EssayGrader.get_sub_verb_probs(essays)
	prob_filename = "sub_verb_probs.pkl"
	prob_filepath = os.path.join(model_dir, prob_filename)
	with open(prob_filepath, 'wb') as fout:
		pkl.dump(probs, fout)

	# Saving verb context probabilities
	print("Computing verb context probabilities....")
	probs = EssayGrader.get_verb_context_probs(essays)
	prob_filename = "verb_ctx_probs.pkl"
	prob_filepath = os.path.join(model_dir, prob_filename)
	with open(prob_filepath, 'wb') as fout:
		pkl.dump(probs, fout)


	# Store results in the output folder
	print("Storing results...")
	res_filepath = os.path.join(output_dir, RES_FILE)
	store_results(results, res_filepath)


	# Preparing featueres and labels
	feat = []
	lab = []
	for idx, r in enumerate(results):
		vec = [r['length'], r['spell'], r['sv_agr'], r['verb'], r['form'], r['cohr'], r['topic']]
		feat.append(vec)
		l = 1 if index[idx]['grade'] == 'high' else 0
		lab.append(l)

	feat = np.array(feat)
	lab = np.array(lab)

	# Training the Logistic regression classifier
	clf = LogisticRegression()
	clf.fit(feat, lab)
	class_filename = 'classifier.pkl'
	classifier_path = os.path.join(model_dir, class_filename)
	with open(classifier_path, 'wb') as fout:
		pkl.dump(clf, fout)


	# Write the csv file containing essays
	# csv_filepath = os.path.join(output_dir, ESS_FILE)
	# write_essays_tocsv(essays, csv_filepath)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_dir', help='input directory containing training and test data', default='../input')
	parser.add_argument('-o', '--output_dir', help='output directory to store results in', default='../output')
	parser.add_argument('-m', '--model_dir', help='directory to store all models', default='../model')
	parser.add_argument('-f', '--function', help='whether to train or test', default='test')
	parser.add_argument('-e', '--evaluate', help='whether or not to perform evaluation', action='store_true')
	args = parser.parse_args()


	setup_env()
	if args.function == 'test':
		label_eval_test(args.input_dir, args.output_dir, args.model_dir, args.evaluate)	
	else:
		train(args.input_dir, args.output_dir, args.model_dir)

if __name__ == '__main__':

	# Some global constants
	ESSAY_DIR = 'essays'
	IDX_FILE = 'index.csv'
	RES_FILE = 'results.txt'
	TRAIN_DIR = "training"
	TEST_DIR = "testing"
	OUT_FILE = 'results.txt'

	exit(main())