from grader import EssayGrader
from essay import Essay
from setup import setup_env
import argparse
import os
from utils import load_index, store_results, evaluate_accuracy



# Function to label examples in input directory and store results in output directory
def label_eval_test(input_dir, output_dir, evaluate):
	ESSAY_DIR = 'essays'
	IDX_FILE = 'index.csv'
	OUT_FILE = 'results.txt'
	TEST_DIR = "testing"

	test_essays_dir = os.path.join(input_dir, TEST_DIR, ESSAY_DIR)
	test_idx_filepath = os.path.join(input_dir, TEST_DIR, IDX_FILE)
	index = load_index(test_idx_filepath)

	results = []
	ground_truth = []

	grader = EssayGrader()
	for record in index:
		filename = record['filename']
		prompt = record['prompt']
		filepath = os.path.join(test_essays_dir, filename)
		
		# Grade, augment and store
		e = Essay(filepath, prompt)
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


def train(input_dir, model_dir):
	pass


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_dir', help='input directory containing training and test data', default='input')
	parser.add_argument('-o', '--output_dir', help='output directory to store results in', default='output')
	parser.add_argument('-m', '--model_dir', help='directory to store all models', default='model')
	parser.add_argument('-f', '--function', help='whether to train or test', default='test')
	parser.add_argument('-e', '--evaluate', help='whether or not to perform evaluation', action='store_true')
	args = parser.parse_args()

	setup_env()
	if args.mode == 'test':
		label_eval_test(args.input_dir, args.output_dir, args.evaluate)	
	else:
		train(args.input_dir, args.model_dir)

if __name__ == '__main__':
	exit(main())