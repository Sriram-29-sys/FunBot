#import necessary libraries
import io
import random
import string  # to process standard python strings
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')

import nltk
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (first run will take a bit of time)
nltk.download('popular', quiet=True)
nltk.download('punkt')
nltk.download('wordnet')

# Reading in the corpus
with open('chatbot.txt', 'r', encoding='utf8', errors='ignore') as fin:
    raw = fin.read().lower()

# Tokenization
sent_tokens = nltk.sent_tokenize(raw)   # list of sentences
word_tokens = nltk.word_tokenize(raw)   # list of words

# Preprocessing
lemmer = WordNetLemmatizer()


def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# Keyword Matching
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
GREETING_RESPONSES = [
    "hi",
    "hey",
    "*nods*",
    "hi there",
    "hello",
    "I am glad you are talking to me.",
]


def greeting(sentence):
    """if user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# Extra, custom responses that make this YOUR bot
ADDITIONAL_RESPONSES = {
    "how are you": "I'm just a chatbot, but thanks for asking!",
    "who created you": "I was created and customized by Devi Sriram from Hyderabad, India, as the FunBot project.",
    "tell me a joke": "Why don't scientists trust atoms? Because they make up everything!",
    "bye": "Goodbye! Feel free to come back if you have more questions.",
}


def response(user_response):
    """Return a response based on corpus similarity and additional rules."""
    robo_response = ''
    sent_tokens.append(user_response)

    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]

    if req_tfidf == 0:
        robo_response = "I am sorry, I don't understand you."
    else:
        robo_response = sent_tokens[idx]

    # Exact rule-based overrides
    if user_response in ADDITIONAL_RESPONSES:
        robo_response = ADDITIONAL_RESPONSES[user_response]

    # remove the user input from sentence list
    sent_tokens.pop()

    return robo_response


# Chatbot conversation loop
if __name__ == "__main__":
    flag = True
    print("FunBot: Hi, I'm Devi's Chatbot (FunBot)! I will answer your queries about chatbots and general topics. If you want to exit, type 'bye'.")

    while flag:
        user_response = input("You: ")
        user_response = user_response.lower()

        if user_response != 'bye':
            if user_response in ('thanks', 'thank you'):
                flag = False
                print("FunBot: You're welcome!")
            else:
                greet = greeting(user_response)
                if greet is not None:
                    print("FunBot:", greet)
                else:
                    print("FunBot:", response(user_response))
        else:
            flag = False
            print("FunBot: Goodbye! Feel free to come back if you have more questions.")
