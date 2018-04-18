import argparse
import os
from utils import load_index
from nltk.tokenize import sent_tokenize
from csv import DictWriter
from stanfordcorenlp import StanfordCoreNLP



def get_pos_tags(sentences):
	global snlp
	tag_list = []
	for s in sentences:
		tags = snlp.pos_tag(s)
		strs = ["{0}/{1}".format(t[0], t[1]) for t in tags]
		str_tags = " ".join(strs)
		tag_list.append(str_tags)
	return tag_list


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("input", help="input directory containing the index and essays")
	parser.add_argument("output", help="output file containing all data")

	args = parser.parse_args()
	index_file = os.path.join(args.input, 'index.csv')
	essay_dir = os.path.join(args.input, 'essays')

	index = load_index(index_file)
	augmented_recs = []
	for rec in index:
		filepath = os.path.join(essay_dir, rec['filename'])
		with open(filepath) as fin:
			rec['text'] = fin.read()
		sentences = [s.strip() for s in sent_tokenize(rec['text'])]
		rec['sentences'] = "\n\n".join(sentences)
		pos_tags = "\n\n".join(get_pos_tags(sentences))
		rec['tags'] = pos_tags
		augmented_recs.append(rec)

	with open(args.output, 'w') as fout:
		writer = DictWriter(fout, fieldnames=['filename', 'grade', 'text', 'sentences', 'tags', 'prompt'])
		writer.writeheader()
		for rec in augmented_recs:
			writer.writerow(rec)


if __name__ == '__main__':
	snlp = StanfordCoreNLP('./resources/stanford-corenlp-full-2018-02-27')
	exit(main())