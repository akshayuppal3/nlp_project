import nltk


def setup_env():
	nltk.download('wordnet')
	nltk.download('averaged_perceptron_tagger')
	nltk.download('punkt')
	nltk.download('wordnet_ic')
	nltk.download('stopwords')
	nltk.download('names')