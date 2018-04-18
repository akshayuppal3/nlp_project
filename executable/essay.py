import utils
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re
from stanfordcorenlp import StanfordCoreNLP


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
			synparse.append(self.snlp.parse(s))
		return synparse

	def _get_depparse(self):
		dep_parse = []
		for s in self.sentences:
			dep_parse.append(self.snlp.dependency_parse(s))
		return dep_parse


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
		data['pos'] = '\n\n'.join(pos_strs)
		return data

	@staticmethod
	def get_fields():
		return ['filepath', 'prompt', 'grade', 'text', 'sentences', 'pos']


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


