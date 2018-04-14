import nltk
from essay import Essay



class EssayGrader:
	def __init__(self):
		pass

	# A value between 1-5, 1 being the lowest and 5 the highest
	def length_score(self, e):
		return 1

	# A value between 0-4
	def spell_score(self, e):
		return 1

	# A value between 1-5, 1 being the lowest and 5 the highest
	def sv_agr_score(self, e):
		return 1

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



