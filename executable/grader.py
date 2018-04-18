import nltk
from essay import Essay
import re
from utils import is_english_word
from nltk.corpus import stopwords
import pickle as pkl
import os
import math


class EssayGrader:
	def __init__(self, models_dir):
		self.models_dir = models_dir

		with open(os.path.join(models_dir, 'sub_verb_probs.pkl'), 'rb') as fin:
			self.sub_verb_probs = pkl.load(fin)


	# A function to compute probabilities for subject verb agreement pairs
	@staticmethod
	def get_sub_verb_probs(essays):
		specific = {'PRP', 'WP', 'WDT'}

		sub_counts = {}
		comb_counts = {}

		for e in essays:
			if e.grade == 'high':
				for sub, stag, vb, vtag in e.get_sub_verb_tups():
					smod_tag = stag
					if stag in specific:
						smod_tag = "{0}/{1}".format(sub.lower(), stag)

					if not smod_tag in sub_counts:
						sub_counts[smod_tag]  = 0

					sub_counts[smod_tag] += 1

					key = "{0}_{1}".format(smod_tag, vtag)
					if not key in comb_counts:
						comb_counts[key] = 0
					comb_counts[key] += 1

		probs = {}
		for key in comb_counts:
			probs[key] = comb_counts[key] / sub_counts[key.split('_')[0]]
		return probs



	# A value between 1-5, 1 being the lowest and 5 the highest
	def length_score(self, e):
		length = len(e.sentences)
		score = 1 + (4 * (length / 30))
		if score > 5:
			score = 5
		return round(score, 2)


	# A value between 0-4, 0 means no error and 4 means all wo
	def spell_score(self, e):
		valid_words = 0
		mistakes = 0

		en_stopwords = stopwords.words('english')

		for w_list in e.words:
			for w in w_list:
				if re.match("^[A-Za-z]+$", w) and not w.lower() in en_stopwords:
					if not is_english_word(w):
						mistakes += 1
					valid_words += 1
		score = mistakes / valid_words
		score *= 4
		return round(score, 2)


	# A value between 1-5, 1 being the lowest and 5 the highest
	def sv_agr_score(self, e):
		PRP_sing = ['he/PRP', 'she/PRP','it/PRP','this/DT','that/WDT', 'which/WDT']
		PRP_non_sing = ['you/PRP', 'we/PRP','they/PRP', 'these/DT', 'those/DT', 'us/PRP']

		specific = {'PRP', 'WP', 'WDT', 'CD'}
		score = 0
		tups = e.get_sub_verb_tups()
		for sub, stag, vb, vtag in tups:
			smod_tag = stag
			if stag in specific:
				smod_tag = "{0}/{1}".format(sub.lower(), stag)
			key = "{0}_{1}".format(smod_tag, vtag)
			prob = self.sub_verb_probs[key] if key in self.sub_verb_probs else 0.000001
			if prob > 0.1:
				score += 1
		score = score / len(tups)
		score = 1 + (score * 4)
		return round(score, 2)

	# A value between 1-5, 1 being the lowest and 5 the highest
	def verb_score(self, e):
		return 1

	# A value between 1-5, 1 being the lowest and 5 the highest
	def form_score(self, e):
		return 1

	# A value between 1-5, 1 being the lowest and 5 the highest
	def cohr_score(self, e):
		return 1

	# A value between 1-5, 1 being the lowest and 5 the highest
	def topic_score(self, e):
		return 1

	# A linear combination of all scores
	def final_score(self, result):
		weights = {
						'length': 2,
						'spell': -1,
						'sv_agr': 1,
						'verb': 1,
						'form': 2,
						'cohr': 0,		# leaving it 0, change for part 2
						'topic': 0,		# leaving it 0, change for part 2

				}
		score = 0
		for key in result:
			score += result[key] * weights[key]
		return round(score, 2)

	# Either 'low' or 'high' based on 
	def label(self, score):
		return 'unknown'


	def grade(self, e):
		result = dict()

		# Part a
		result['length'] = self.length_score(e)
		
		# Part b
		result['spell'] = self.spell_score(e)

		# Part c(i), c(ii), c(iii)
		result['sv_agr'] = self.sv_agr_score(e)
		result['verb'] = self.verb_score(e)
		result['form'] = self.form_score(e)
		
		# Part d(i), d(ii)
		result['cohr'] = self.cohr_score(e)
		result['topic'] = self.topic_score(e)

		# Aggregates
		result['final'] = self.final_score(result)
		result['grade'] = self.label(result['final'])

		# TODO: Remove this
		result['grade'] = e.grade

		return result



