
import random
import json
import pickle

import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

from flask import Flask, render_template, request

import numpy as np
import time

app = Flask(__name__)

lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intents.json").read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')


def clean_up_sentence(sentence):
	"""
	The `clean_up_sentence` function tokenizes and lemmatizes the words in a given sentence using NLTK.
	
	:param sentence: The `clean_up_sentence` function takes a sentence as input, tokenizes the words in
	the sentence using the NLTK library, and then lemmatizes each word in the tokenized list. Finally,
	it returns the lemmatized words as a list
	:return: The function `clean_up_sentence` returns a list of lemmatized words extracted from the
	input sentence.
	"""
	sentence_words = nltk.word_tokenize(sentence)
	sentence_words = [lemmatizer.lemmatize(word)
					for word in sentence_words]

	return sentence_words


def bag_of_words(sentence):
	"""
	The `bag_of_words` function takes a sentence as input, cleans it up, and creates a bag of words
	representation of the sentence using a binary vector.
	
	:param sentence: The `bag_of_words` function seems to be creating a bag of words representation for
	a given sentence. However, the `clean_up_sentence` function and the `words` variable are not defined
	in the provided code snippet
	:return: The function `bag_of_words(sentence)` returns a numpy array representing a bag of words for
	the input sentence. Each element in the array corresponds to a word in the vocabulary `words`, and
	the value is 1 if the word is present in the input sentence and 0 otherwise.
	"""
	sentence_words = clean_up_sentence(sentence)
	bag = [0] * len(words)

	for w in sentence_words:
		for i, word in enumerate(words):
			if word == w:
				bag[i] = 1
	return np.array(bag)


def predict_class(sentence):
	"""
	The function `predict_class` takes a sentence as input, predicts the class of the sentence using a
	pre-trained model, and returns a list of intents along with their probabilities based on the
	prediction results.
	
	:param sentence: The `predict_class` function takes a sentence as input and predicts the class or
	intent of that sentence based on a pre-trained model. The function uses a bag of words
	representation of the sentence, then predicts the class using the model. It filters out predictions
	below a certain threshold and returns a list of
	:return: The `predict_class` function returns a list of dictionaries, where each dictionary contains
	the predicted intent and its corresponding probability for the input sentence.
	"""
	bow = bag_of_words(sentence)
	res = model.predict(np.array([bow]))[0]

	ERROR_THRESHOLD = 0.25

	results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

	results.sort(key=lambda x: x[1], reverse=True)

	return_list = []

	for r in results:
		return_list.append({'intent': classes[r[0]],
							'probability': str(r[1])})
	return return_list


def get_response(intents_list, intents_json):
	"""
	This Python function retrieves a response based on the provided intent from a list of intents and
	corresponding JSON data.
	
	:param intents_list: A list containing dictionaries with information about different intents. Each
	dictionary includes the 'intent' key, which represents the intent of the user input
	:param intents_json: Intents_json is a JSON object that contains a list of intents with their
	corresponding tags and responses. Each intent object in the list has keys like 'tag' for the intent
	tag and 'responses' for a list of possible responses associated with that intent
	:return: a response based on the intent provided in the `intents_list`. It looks for the matching
	intent in the `intents_json` and returns a random response associated with that intent.
	"""
	tag = intents_list[0]['intent']
	list_of_intents = intents_json['intents']

	result = ''

	for i in list_of_intents:
		if i['tag'] == tag:
			result = random.choice(i['responses'])
			break
	return result


def get_input():
  text = input()
  return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get', methods=['POST'])
def get_bot_response():
    user_input = request.form['msg']
    predict = predict_class(user_input)
    response = get_response(predict, intents)
    return response

if __name__ == '__main__':
    app.run(debug=True)



