import nltk
from essay import Essay
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tree import Tree
from nltk.tree import ParentedTree
from nltk import pos_tag
from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://corenlp.run', port=80)

PRP_sing = ['He/PRP', 'he/PRP', 'She/PRP', 'she/PRP','It/PRP','it/PRP','That/DT','This/DT','this/DT','That/WDT']
PRP_non_sing = ['You/PRP', 'you/PRP', 'We/PRP','we/PRP', 'They/PRP' ,'they/PRP', 'These/DT' , 'these/DT' ,'Those/DT', 'those/DT', 'us/PRP']


class EssayGrader:
	def __init__(self):
		pass

	# A value between 1-5, 1 being the lowest and 5 the highest
	def length_score(self, e):
		return 1

	# A value between 0-4
	def spell_score(self, e):
		count_words = 0
		spelling_error = 0
		score = 0
		sentences = sent_tokenize(e.data)
		for sentence in sentences:
			words= word_tokenize(sentence)
			count_words += len(words)
			for word in words:
				if(not(wn.synsets(word))):
					spelling_error += 1
		score = spelling_error/count_words
		score *= 4
		return score

	# A value between 1-5, 1 being the lowest and 5 the highest
	def sv_agr_score(self, e):
		score = 0
		sentences = sent_tokenize(e.data)
		for sentence in sentences:
			a = nlp.dependency_parse(sentence)
			nsubj_flag = 0
			for element in a:
				if (element[0] == 'nsubj'):
					verb_idx = element[1]
					subject_idx = element[2]
					nsubj_flag = 1
			if (nsubj_flag == 1 ):        
				#tokenize the sentence    
				words = word_tokenize(sentence)
				tagged_words = pos_tag(words)
				#check the tag of the words
				verb, verb_tag = tagged_words[verb_idx - 1]
				subject, subject_tag = tagged_words[subject_idx - 1]
				if (subject_tag == 'NN' or subject_tag == 'NNS'):
					if (verb_tag == 'VBZ' and subject_tag == 'NNS'):          #cant have NNS-> VBZ
						score += 1 
					if (verb_tag == 'VBP' and subject_tag == 'NN'):           #cant have NN-> VBP
						score += 1
				elif (subject_tag == 'PRP'):
					composite = subject + '/' + subject_tag                   #concatenate so that we can query PRP_SING 
					if (verb_tag == 'VBZ'):
						if composite in PRP_non_sing:                                #cant have they,you -> VBZ 
							score += 1
					elif (verb_tag == 'VBP'):                                 #cant have he,she -> VBP
						if composite in PRP_sing:                                #cant have they,you -> VBZ 
							score += 1
			score = score/len(sentences)
			score *= 5                  
			return score

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
		return score

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

		return result



