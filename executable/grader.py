import nltk
from essay import Essay
import re
from utils import is_english_word
from nltk.corpus import stopwords, wordnet, wordnet_ic
import pickle as pkl
import os
import math
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus.reader.wordnet import WordNetError
from utils import third_pers_plural
from utils import third_sing_plural


class EssayGrader:
	def __init__(self, models_dir):
		self.models_dir = models_dir

		with open(os.path.join(models_dir, 'sub_verb_probs.pkl'), 'rb') as fin:
			self.sub_verb_probs = pkl.load(fin)

		with open(os.path.join(models_dir, 'verb_ctx_probs.pkl'), 'rb') as fin:
			self.verb_ctx_probs = pkl.load(fin)

		with open(os.path.join(models_dir, 'conj_vb_probs.pkl'), 'rb') as fin:
			self.conj_vb_probs = pkl.load(fin)

		with open(os.path.join(models_dir, 'classifier.pkl'), 'rb') as fin:
			self.classifier = pkl.load(fin)

		with open(os.path.join(models_dir, 'scaler.pkl'), 'rb') as fin:
			self.scaler = pkl.load(fin)

		self.lemmatizer = WordNetLemmatizer()
		self.semcor_ic = wordnet_ic.ic('ic-semcor.dat')
		self.stopwords = stopwords.words('english')

		# with open('resources/sumo_graph.pkl', 'rb') as fin:
		# 	self.sumo_graph = pkl.load(fin)

		# @TODO Remove these in the end
		# self.high_scores = []
		# self.low_scores = []


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


	@staticmethod
	def _get_key(tup):
		word, pos = tup
		cwords = {'not', 'i', 'we', 'you', 'they', 'he', 'she', 'it', 'is' 'am', 'are', 'has', 'have', 'be', 'been', 'will', 'shall', 'had'}
		if word.lower() in cwords:
			return "{0}/{1}".format(word.lower(), pos)
		return pos


	# A function to compute bigram probabilities in the context of a verb
	@staticmethod
	def get_verb_context_probs(essays):
		sub_counts = {}
		comb_counts = {}

		for e in essays:
			if e.grade == 'high':
				contexts = e.get_verb_contexts()
				for context in contexts:
					for i in range(len(context) - 1):
						first, second = context[i], context[i + 1]
						first_key = EssayGrader._get_key(first)
						second_key = EssayGrader._get_key(second)

						for key in (first_key, second_key):
							if not key in sub_counts:
								sub_counts[key] = 0
							sub_counts[key] += 1

						comp_key = "{0}_{1}".format(first_key, second_key)
						if not comp_key in comb_counts:
							comb_counts[comp_key] = 0
						comb_counts[comp_key] += 1
		
		probs = {}
		for key in comb_counts:
			probs[key] = comb_counts[key] / sub_counts[key.split("_")[0]]
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

		en_stopwords = self.stopwords
		en_stopwords += ['could', 'would', 'since', 'without']

		for idx, w_list in enumerate(e.words):
			for w in w_list:
				if re.match("^[A-Za-z]+$", w) and not w.lower() in en_stopwords and e.pos_tags[idx] != 'NNP':
					if not is_english_word(w):
						mistakes += 1
					valid_words += 1
		score = (mistakes / float(valid_words)) if valid_words > 0 else 0
		score *= 4.0
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
			if prob > 0.08:
				score += 1
		score = (score / len(tups)) if len(tups) > 0 else 1
		score = 1 + (score * 4)
		return round(score, 2)

	# A value between 1-5, 1 being the lowest and 5 the highest
	def verb_score(self, e):
		num_bigrams = 0
		score = 0
		contexts = e.get_verb_contexts()
		for context in contexts:
			for i in range(len(context) - 1):
				first, second = context[i], context[i + 1]
				first_key = EssayGrader._get_key(first)
				second_key = EssayGrader._get_key(second)
				comp_key = "{0}_{1}".format(first_key, second_key)
				prob = self.verb_ctx_probs[comp_key] if comp_key in self.verb_ctx_probs else 0.000001
				if prob > 0.04:
					score += 1
				num_bigrams += 1

		score = (score / num_bigrams) if num_bigrams > 0 else 1
		score = 1 + (score * 4)
		return round(score, 2)
	
	# Check for the presence of fragments
	def _frag_present(self, tree):
		for t in tree.subtrees():
			if t.label() == 'FRAG':
				return True
		return False


	# Helper function to check for incomplete sentences
	def _incomplete_sentence(self, tree):
		labels = []
		for t in tree.subtrees():
			labels.append(t.label())
		return labels[1] not in ['S', 'SINV', 'SQ', 'SBARQ']

	# Helper functin to compute SBAR scores
	def _sbar_incorrect(self, tree):
		for t in tree.subtrees():
			if t.label() == 'SBAR':
				labels = [n.label() for n in t]
				if labels[0] == 'S':
					return True
		return False

	# Get conjunction scores
	def _incorrect_conj(self, tree, grade):
		for t in tree.subtrees():
			if t.label() == 'SBAR':
				children = [n for n in t]
				c_labels = [c.label() for c in children]
				try:
					in_idx = [idx for idx, lab in enumerate(c_labels) if lab == 'IN']
					s_idx = [idx for idx, lab in enumerate(c_labels) if lab == 'S']
					if in_idx and s_idx:
						inj = '_'.join([children[idx][0].lower() for idx in in_idx])
						s = children[s_idx[0]]
						s_labels = [n.label() for n in s]
						vp = s[s_labels.index('VP')]
						vp_labels = [n.label() for n in vp.subtrees()]
						verb_labels = [l for l in vp_labels if re.match("VB+", l)]
						if len(verb_labels) == 0:
							return True
						conj_tup_key = "{0}_{1}".format(inj, verb_labels[0])
						prob = self.conj_vb_probs[conj_tup_key] if conj_tup_key in self.conj_vb_probs else 0.000001
						if prob < 0.125:
							return True
				except ValueError:
					pass
		return False


	# Helper function to score a tree between 0 and 1 based on syntactic well formedness
	def _tree_score(self, tree, grade):
		frag_score = int(not self._frag_present(tree))
		sent_score = int(not self._incomplete_sentence(tree))
		sbar_score = int(not self._sbar_incorrect(tree))
		conj_score = int(not self._incorrect_conj(tree, grade))
		return np.average([frag_score, sent_score, sbar_score, conj_score])



	# A value between 1-5, 1 being the lowest and 5 the highest
	def form_score(self, e):
		total_score = 0
		for tree in e.syn_parse:
			total_score += self._tree_score(tree, e.grade)
		score = (total_score / len(e.syn_parse)) if len(e.syn_parse) > 0 else 1
		score = 1 + (score * 4)
		return round(score, 2)

	# A value between 1-5, 1 being the lowest and 5 the highest
	def cohr_score(self, e):
		sing_score, plural_score = third_pers_plural(e.text) #result returnded by JSON
		score1 = (sing_score + plural_score) / 2
		norm1 = score1
		score2 = third_sing_plural(e.sentences) #rule based system for noun choerence
		norm2 = score2
		norm = 0
		if ((norm1 + norm2) == 0):
			return(0)
		elif(norm1 == 0 or norm2 == 0):
			norm = 1
		else:
			norm = 2	
		score = ((score1 + score2) / norm)	
		score = 1 - score  		#making it a psotive value
		score = 1 + (score * 4)
		return np.round(score, 2)


	def _get_pos(self, words, pos_tags, regex):
		return [w for idx, w in enumerate(words) if re.match(regex, pos_tags[idx])]


	def _get_wn_sim(self, plist, elist, pos):
		all_sim = [0.1]
		for p in plist:
			p_sim = [0]
			for e in elist:
				if not p in self.stopwords and not e in self.stopwords:
					try:
						w1 = wordnet.synsets(p, pos=pos)[0]
						w2 = wordnet.synsets(e, pos=pos)[0]
						p_sim.append(w1.lin_similarity(w2, self.semcor_ic))
					except WordNetError:
						pass
					except IndexError:
						pass

			all_sim.append(np.max(p_sim))

		all_sim = np.array(all_sim)
		return np.average(all_sim[all_sim > 0])



	def _get_wordnet_score(self, e):
		all_es_words = [w for sent in e.words for w in sent]
		all_es_pos = [t for sent in e.pos_tags for t in sent]
		all_pt_words = [w for sent in e.pt_words for w in sent]
		all_pt_pos = [t for sent in e.pt_pos_tags for t in sent]

		sims = []
		es_nn = self._get_pos(all_es_words, all_es_pos, 'NNS+')
		pt_nn = self._get_pos(all_pt_words, all_pt_pos, 'NNS+')

		if len(es_nn) > 0 and len(pt_nn) > 0:
			noun_sim = self._get_wn_sim(pt_nn, es_nn, wordnet.NOUN)
			sims.append(noun_sim)

		es_vb = self._get_pos(all_es_words, all_es_pos, 'VB+')
		pt_vb = self._get_pos(all_pt_words, all_pt_pos, 'VB+')
		if len(es_vb) > 0 and len(pt_vb) > 0:
			verb_sim = self._get_wn_sim(pt_vb, es_vb, wordnet.VERB)
			sims.append(verb_sim)

		return np.average(sims, weights=[0.7, 0.3])

	# def _get_sumo_entities(self, nlist):	
	# 	ent_urls = [e for e, _, _ in self.sumo_graph]
	# 	ent_urls = set(ent_urls)
	# 	for e in ent_urls:
	# 		print()




	# def _get_sumo_score(self, e):
	# 	all_es_words = [w for sent in e.words for w in sent]
	# 	all_es_pos = [t for sent in e.pos_tags for t in sent]
	# 	all_pt_words = [w for sent in e.pt_words for w in sent]
	# 	all_pt_pos = [t for sent in e.pt_pos_tags for t in sent]

	# 	es_nn = self._get_pos(all_es_words, all_es_pos, 'NN.*')
	# 	pt_nn = self._get_pos(all_pt_words, all_pt_pos, 'NN.*')

	# 	es_ents = self._get_sumo_entities(es_nn)
	# 	pt_ents = self._get_sumo_entities(pt_nn)
	# 	return score


	# A value between 1-5, 1 being the lowest and 5 the highest
	def topic_score(self, e):
		score = self._get_wordnet_score(e)
		# score = self._get_sumo_score(e)
		score = 1 + (score * 4)
		return round(score, 2)

	# A linear combination of all scores
	def final_score(self, result):
		weights = {
						'length': 2,
						'spell': -1,
						'sv_agr': 1,
						'verb': 1,
						'form': 2,
						'cohr': 2,
						'topic': 3

				}
		score = 0
		for key in result:
			score += result[key] * weights[key]
		return round(score, 2)

	# A classifier based final score assignment
	def final_score_class(self, r):
		vec = np.array([r['length'], r['spell'], r['sv_agr'], r['verb'], r['form'], r['cohr'], r['topic']]).reshape(1, -1)
		vec = self.scaler.transform(vec)
		high_prob = self.classifier.predict_proba(vec)[0][1]
		return round(high_prob * 55, 2)


	# Either 'low' or 'high' based on the final score
	def label(self, score):
		return 'high' if (score / 55) > 0.5 else 'low'


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
		# result['final'] = self.final_score(result)
		result['final'] = self.final_score_class(result)
		result['grade'] = self.label(result['final'])

		# score = result['final']
		# if e.grade == 'low':
		# 	self.low_scores.append(score)
		# else:
		# 	self.high_scores.append(score)

		return result



