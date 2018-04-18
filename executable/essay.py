import utils
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import re


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

		sentences = sent_tokenize(self.text)
		sentences = list(filter(len, [s.strip() for s in sentences]))
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

	def to_dict(self):
		data = {}
		data['filepath'] = self.filepath
		data['prompt'] = self.prompt
		data['grade'] = self.grade
		data['text'] = self.text
		data['sentences'] = '\n\n'.join(self.sentences)
		return data

	@staticmethod
	def get_fields():
		return ['filepath', 'prompt', 'grade', 'text', 'sentences']


	def __init__(self, filepath, prompt, grade=None):
		self.filepath = filepath
		self.prompt = prompt
		self.grade = grade
		self.text = utils.read_txt_file(self.filepath).strip()
		self.sentences = self._get_sentences()
		print(len(self.sentences))
		print(self.grade)