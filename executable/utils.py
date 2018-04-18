import csv
from nltk.corpus import wordnet
import essay

# Function to load records from csv file
def load_index(filepath):
	records = []
	with open(filepath) as fin:
		reader = csv.DictReader(fin, delimiter=';')
		for row in reader:
			records.append(row)
	return records


# Stores results in a text file
def store_results(results, filepath):
	order = ['filename', 'length', 'spell', 'sv_agr', 'verb', 'form', 'cohr', 'topic', 'final', 'grade']
	with open(filepath, 'w') as fout:
		lines = []
		for result in results:
			line = ';'.join([str(result[k]) for k in order])
			lines.append(line)
		fout.write('\n'.join(lines))


# Evaluate accuracy
def evaluate_accuracy(gts, predictions):
	correct = 0
	for idx, gt in enumerate(gts):
		if gt == predictions[idx]:
			correct += 1
	return correct / len(gts)


# Read text file
def read_txt_file(filepath):
	with open(filepath) as fin:
		return fin.read()


# Check if a word belongs to english dictionary
def is_english_word(word):
	return wordnet.synsets(word)


# Write essays to the CSV file
def write_essays_tocsv(essays, filepath):
	with open(filepath, 'w') as fout:
		writer = csv.DictWriter(fout, fieldnames=essay.Essay.get_fields())
		writer.writeheader()
		for e in essays:
			writer.writerow(e.to_dict())
