import nltk


def setup_env():
	nltk.download('wordnet')
	nltk.download('averaged_perceptron_tagger')
	nltk.download('punkt')