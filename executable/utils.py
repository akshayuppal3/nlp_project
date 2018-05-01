import csv
from nltk.corpus import wordnet
import essay
import json
from stanfordcorenlp import StanfordCoreNLP
from nltk.classify import apply_features
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import nltk
import random

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

# Function that returns the features of the word to the naive bayes classifier
def gender_features(word):
	name = word.lower()
	return{
	'last_char' : name[-1],
	'first_char': name[0],
	'last_two' : name[-2:],
	'first_2' : name[:1],
	'last_three' : name[-3:]
	}
# @return - the most probable gender of the word (based on male an female corpora of nltk)
# @params - an individual word
def gender_label(word):

	f_names = nltk.corpus.names.words('female.txt')
	m_names = nltk.corpus.names.words('male.txt')

	all_names = [(i,'m') for i in m_names] + [(i,'f') for i in f_names]
	random.shuffle(all_names)

	fet_set = [(gender_features(n), label) for n,label in all_names]
	train_set, test_set = fet_set[500:], fet_set[:500]

	classifier = nltk.NaiveBayesClassifier.train(train_set)

	#return either male or female with a cofidence(probaility) of more that 65%
	dist = classifier.prob_classify(gender_features(word))
	if(dist.prob('m') > dist.prob('f')):
		if(dist.prob('m') > 0.65):
			return('m')
		else:
			return('u')	
	if(dist.prob('f') > dist.prob('m')):
		if(dist.prob('f') > 0.65):
			return('f')
		else:
			return('u')		

#@return no of male and female entities @params: text
def named_entities(text):
	nlp = StanfordCoreNLP('http://corenlp.run', port=80) #To be changes afterwards
	feminine_words = ['she', 'her', 'hers']
	masculine_words = ['he', 'him', 'his']
	props = {'annotators': 'ner,pos'} # tokenize,ssplit,pos,lemma,

	try: 
		jsonresponse = json.loads(nlp.annotate(text, properties=props))
		sentences = jsonresponse['sentences']
		count_m = 0
		count_f = 0

		for idx,sentence in enumerate(sentences):
			entity = sentence['entitymentions']
			
			for element in entity:
				word = element['text']
				ner = element['ner']
				begin = element['tokenBegin']
				#print(word,ner)
				if( word.lower() not in feminine_words and word.lower() not in masculine_words):  #not including she or he as an entity
					if(ner == 'PERSON'):
						#print(word)
						if(len(word_tokenize(word)) > 1):
							word = word_tokenize(word)[0]      #taking only the first name in case of surnames
						label = gender_label(word)
						if(label == 'm'):
							print("male: ",word)
							count_m += 1
						elif(label == 'f'):
							print("female: ", word)
							count_f += 1

		if (count_m > 0 or count_f > 0):			
			d = {"m" : count_m , "f" : count_f}
			return(d)
		else:
			return({'m' : 0 , 'f' :0})		
	except ValueError:
		return({'m' : 0 , 'f' :0})

#getting the named entities as an alternative to result returned by Json
def named_entity_pos(text):
	count_m = 0
	count_f = 0
	words = get_words(text)
	tagged_words = pos_tag(words)
	for token in tagged_words:
		tag = token[1]
		word = token[0]
		if(tag == 'NNP'):
			if(is_english_word(word)):
				label = gender_label(word)
				if(label == 'm'):
					print("male: ",word)
					count_m += 1
				elif(label == 'f'):
					print("female: ", word)
					count_f += 1
	if (count_m > 0 or count_f > 0):			
		d = {"m" : count_m , "f" : count_f}
	else:
		d = {'m' : 0 , 'f' :0}
	return(d)

def get_words(text):
	return (word_tokenize(text))				

#get the count of third person singular feminine words
def get_feminine_poss(words):
	feminine_words = ['she', 'her', 'hers']
	count = 0
	for word in words:
		if word.lower() in feminine_words:
			print("Feminine",word)
			count += 1
	return (count)

#get the count of third person singular feminine words
def get_masculine_poss(words):
	masculine_words = ['he', 'him', 'his']
	count = 0
	for word in words:
		if word.lower() in masculine_words:
			print("Masculine",word)
			count += 1
	return(count)		

# @returns a score between 0 - 1 based on incorrect reference in antecedent
def third_pers_sing(text):
	score = 0
	m_f_ent = named_entities(text)     #count of male and female entities
	#@Alternate approach of pos tag not seem to work
	#m_f_ent_pos = named_entity_pos(text)
	# if (m_f_ent['m'] == 0 and m_f_ent['f'] == 0):	#taking the pos tag as an alternative
	# 	if(m_f_ent_pos['m'] > 0 or m_f_ent_pos['f'] > 0):
	# 		m_f_ent['m'] = m_f_ent_pos['m']
	# 		m_f_ent['f'] = m_f_ent_pos['f']
 	#print(m_f_ent)
	m_ref = get_masculine_poss(get_words(text))
	f_ref = get_feminine_poss(get_words(text))

	#Check for m/f antecendant
	#Check if there is mention of male entity then there exits one
	if (m_ref > 0):
		# print(m_f_ent['m'])
		if(m_f_ent['m'] == 0):
			score += 1
	#Check if there is mention of male entity then there exits one
	if (f_ref > 0):	
		if(m_f_ent['f'] == 0):
			score += 1
	#Max score that can be would be 2 when there is no mention of either male or female entity but 
	#there is occurence of masculine and feminine words
	#Normailze
	if (score == 2):
		score = 1
	return (score)

#@return third_person_sing score between 0-1 @params text
def third_pers_plural(text):
	nlp = StanfordCoreNLP('http://corenlp.run', port=80) #To be changes afterwards
	third_plural_words = ['they', 'themselves','their','them']
	masculine_words = ['he', 'him', 'his']
	feminine_words = ['she', 'her', 'hers']
	third_person_score = third_pers_sing(text)
	third_person_normalize = 0
	props = {'annotators': "coref"} #tokenize,ssplit,,
	try:							#handling exception by Json
		jsonresponse = json.loads(nlp.annotate(text, properties=props))
		jsonresponse = (jsonresponse['corefs'])
		#print(jsonresponse)
		vaild_count = 0
		score = 0
		score_th_sing = 0
		for key,value in jsonresponse.items():  #dict
			ident = key
			gender = value[0]['gender']   #referring to the first element(main entity) in coref chain 
			number = value[0]['number']
			animacy = value[0]['animacy']
			text = value[0]['text']
			for element in value:  #list
				if(element['text'] in third_plural_words): 
				# or element['text'] in masculine_words
				# or element['text'] in feminine_words):  #Check only in case we encounter third person plural(they)
					vaild_count += 1
					#if(element['gender'] != 'UNKNOWN'): #Not checking with unknown gender
					if (element['gender'] != gender or element['number'] != number):
						# print (gender, number, number, animacy)
						# print(element['text'],element['gender'], element['number'], element['animacy'])
						score += 1
				#checking the reference of third person singular
				if (element ['text'] in masculine_words or element['text'] in feminine_words):
					third_person_normalize += 1
					if (element['gender'] != gender or element['number'] != number):
						score_th_sing += 1
						print(text)
						print(text,gender, number, animacy)
						print(element['text'],element['gender'], element['number'], element['animacy'])
		#Update the third person singular scores if update found by coref
		if(third_person_score == 0):
			if(score_th_sing > 0):
				if (third_person_normalize >0):  #check for div by zero
					third_person_score = score_th_sing/ third_person_normalize
			else:  #Keep it zero if nothing caught by coref
				third_person_score = 0

		#normalize the score by no of occurence of third person plural words in the essay
		if (vaild_count != 0):   #Handle div by zero error
			score = score/ vaild_count
		else:
			score = score		
	except ValueError:
		score = 0		
	return(third_person_score, score)	