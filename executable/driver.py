from grader import EssayGrader
from essay import Essay
from setup import setup_env
import argparse
import os
from utils import load_index, store_results, evaluate_accuracy, write_essays_tocsv
import pickle as pkl



# Function to label examples in input directory and store results in output directory
def label_eval_test(input_dir, output_dir, model_dir, evaluate):
	ESSAY_DIR = 'essays'
	IDX_FILE = 'index.csv'
	OUT_FILE = 'results.txt'
	TEST_DIR = "testing"

	test_essays_dir = os.path.join(input_dir, TEST_DIR, ESSAY_DIR)
	test_idx_filepath = os.path.join(input_dir, TEST_DIR, IDX_FILE)
	index = load_index(test_idx_filepath)

	results = []
	ground_truth = []

	grader = EssayGrader(model_dir)
	for record in index:
		filename = record['filename']
		prompt = record['prompt']
		grade = record['grade'] if 'grade' in record else None

		filepath = os.path.join(test_essays_dir, filename)
		
		# Grade, augment and store
		e = Essay(filepath, prompt, grade)
		result = grader.grade(e)
		result['filename'] = filename
		results.append(result)

		if evaluate:
			ground_truth.append(record['grade'])

	out_filepath = os.path.join(output_dir, OUT_FILE)
	store_results(results, out_filepath)

	if evaluate:
		predictions = [r['grade'] for r in results]
		accuracy = evaluate_accuracy(ground_truth, predictions)
		print("Accuracy: {0:.2f}%".format(accuracy * 100))


def train(input_dir, output_dir, model_dir):
	ESSAY_DIR = 'essays'
	IDX_FILE = 'index.csv'
	RES_FILE = 'results.txt'
	ESS_FILE = 'essays.csv'
	TRAIN_DIR = "training"
	
	train_essays_dir = os.path.join(input_dir, TRAIN_DIR, ESSAY_DIR)
	train_idx_filepath = os.path.join(input_dir, TRAIN_DIR, IDX_FILE)

	index = load_index(train_idx_filepath)
	grader = EssayGrader(model_dir)

	
	# load essays
	with open('essay.pkl', 'rb') as fin:
		essays = pkl.load(fin)

	# essays = []
	results = []
	for idx, record in enumerate(index):
		print('---' * 10 )
		filename = record['filename']
		prompt = record['prompt']
		grade = record['grade']

		# Grade, augment and store
		filepath = os.path.join(train_essays_dir, filename)
		# e = Essay(filepath, prompt, grade)
		# essays.append(e)

		e = essays[idx]
		result = grader.grade(e)
		result['filename'] = filename
		results.append(result)

	probs = EssayGrader.get_sub_verb_probs(essays)
	prob_filename = "sub_verb_probs.pkl"
	prob_filepath = os.path.join(model_dir, prob_filename)
	with open(prob_filepath, 'wb') as fout:
		pkl.dump(probs, fout)

	with open('essay.pkl', 'wb') as fout:
		pkl.dump(essays, fout)

	res_filepath = os.path.join(output_dir, RES_FILE)
	store_results(results, res_filepath)
	csv_filepath = os.path.join(output_dir, ESS_FILE)
	write_essays_tocsv(essays, csv_filepath)



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
		label_eval_test(args.input_dir, args.output_dir, args.evaluate)	
	else:
		train(args.input_dir, args.output_dir, args.model_dir)

if __name__ == '__main__':
	exit(main())