import utils
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re
from stanfordcorenlp import StanfordCoreNLP
from nltk.tree import Tree


class Essay():
	def _split_sentence(self, s, split_on):
		all_sent = []
		start = 0
		for token in split_on:
			offset = len(token.split('.')[0])
			start_idx = [m.start() for m in re.finditer(token, s)]
			for idx in start_idx:
				new_idx = idx + offset
				end = new_idx + 1
				all_sent.append(s[start:end])
				start = end
		return all_sent


	def _get_sentences(self):
		en_stopwords = stopwords.words('english')
		en_stopwords.append('etc')

		tok_sentences = sent_tokenize(self.text)
		sentences = []
		for tsen in tok_sentences:
			sentences += list(filter(len, [s.strip() for s in tsen.split('\n')]))

		new_sentences = []
		for s in sentences:
			tokens = re.findall("[A-Za-z0-9]+\.[A-Za-z0-9]+", s)
			if len(tokens) > 0:
				split_on = []
				for token in tokens:
					word_list = token.split('.')
					one_valid = False
					for w in word_list:
						if len(w) > 1 and (w.lower() in en_stopwords or utils.is_english_word(w.lower())):
							one_valid = True
					if one_valid:
						split_on.append(token)
				if len(split_on) > 0:
					new_sentences += self._split_sentence(s, split_on)
				else:
					new_sentences.append(s)
			else:
				new_sentences.append(s)
		return new_sentences


	def _get_words(self):
		words = []
		for s in self.sentences:
			words.append(self.snlp.word_tokenize(s))
		return words


	def _get_pos(self):
		pos_tags = []
		for s_idx, s in enumerate(self.sentences):
			pos = [t[1] for t in self.snlp.pos_tag(s)]
			pos_tags.append(pos)
		return pos_tags

	def _get_synparse(self):
		synparse = []
		for s in self.sentences:
			synparse.append(Tree.fromstring(self.snlp.parse(s)))
		return synparse

	def _get_depparse(self):
		dep_parse = []
		for s in self.sentences:
			dep_parse.append(self.snlp.dependency_parse(s))
		return dep_parse

	def get_sub_verb_tups(self):
		valid_sub = {'PRP', 'NNS', 'NN', 'NNP', 'WP', 'WDT', 'CD', 'NNPS'}
		tups = []
		for idx, s in enumerate(self.sentences):
			for element in self.dep_parse[idx]:
				if (element[0] == 'nsubj' or element[0] == 'nsubjpass'):
					verb_idx = element[1] - 1
					subject_idx = element[2] - 1
					verb_tag = self.pos_tags[idx][verb_idx]
					subject_tag = self.pos_tags[idx][subject_idx]
					subject = self.words[idx][subject_idx]
					verb = self.words[idx][verb_idx]
					if re.match("VB*", verb_tag) and subject_tag in valid_sub:
						tups.append((subject, subject_tag, verb, verb_tag))
		return tups


	def get_verb_contexts(self):
		contexts = []
		for idx, s in enumerate(self.sentences):
			verb_idx = []
			for v_idx, tag in enumerate(self.pos_tags[idx]):
				if re.match("VB*", tag) and v_idx != 0:
					verb_idx.append(v_idx)
			
			intervals = []
			for i in verb_idx:
			 	start = i - 1
			 	if start < 0:
			 		start = 0
			 	intervals.append((start, i))

			if len(intervals) > 0:
				new_intervals = [intervals[0]]
				for i in range(1, len(intervals)):
					prev, current = new_intervals[-1], intervals[i]
					if current[0] <= prev[1]: 
						new_intervals[-1] = (new_intervals[-1][0], max(prev[1], current[1]))
					else:
						new_intervals.append(current[:])

				for interval in new_intervals:
					start, end = interval
					context = [(self.words[idx][i], self.pos_tags[idx][i])for i in range(start, end + 1)]
					contexts.append(context)
		return contexts




	def to_dict(self):
		data = {}
		data['filepath'] = self.filepath
		data['prompt'] = self.prompt
		data['grade'] = self.grade
		data['text'] = self.text
		data['sentences'] = '\n\n'.join(self.sentences)
		pos_strs = []
		for i in range(len(self.words)):
			pos_strs.append(" ".join(["{0}/{1}".format(self.words[i][j], self.pos_tags[i][j]) for j in range(len(self.words[i]))]))
		data['words'] = '\n\n'.join([word_str for word_str in [' '.join(w) for w in self.words]])
		data['pos'] = '\n\n'.join(pos_strs)
		data['syn'] = '\n\n'.join(self.syn_parse)
		data['dep'] = '\n\n'.join([str(d) for d in self.dep_parse])
		return data

	@staticmethod
	def get_fields():
		return ['filepath', 'prompt', 'grade', 'text', 'sentences', 'words', 'syn', 'pos', 'dep']


	def __init__(self, filepath, prompt, grade=None):
		self.snlp = StanfordCoreNLP('http://localhost', port=8080)
		self.filepath = filepath
		self.prompt = prompt
		self.grade = grade
		self.text = utils.read_txt_file(self.filepath).strip()
		self.sentences = self._get_sentences()
		self.words = self._get_words()
		self.pos_tags = self._get_pos()
		self.syn_parse = self._get_synparse()
		self.dep_parse = self._get_depparse()


