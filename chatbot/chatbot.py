import nltk 
import random 
import string
import warnings 
import pandas as pd
from cleaner import clean_corpus

##ChatterBot creates a SQLite databbase file where it stores all your inputs and connect them with possible reponses
##Improves responses manually 
#1 Create instance of chatbot
#2 Train Chatbot on industry-specific data 

from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer #custom training dataset 
from chatterbot.trainers import ChatterBotCorpusTrainer #training library dataset
CORPUS_FILE = "chat.txt"

manual_chatbot = ChatBot("manual_chatbot") #flowerpot chatbot

trainer = ListTrainer(manual_chatbot)
trainer.train([
    "Hi",
    "Welcome, friend 🤗",
])
trainer.train([
    "Are you a plant?",
    "No, I'm the pot below the plant!",
])
cleaned_corpus = clean_corpus(CORPUS_FILE)
trainer.train(cleaned_corpus) 

# library_chatbot = ChatBot('Example Bot')
# trainer = ChatterBotCorpusTrainer(library_chatbot)
# trainer.train("chatterbot.corpus.english")
# trainer.train(
#     "chatterbot.corpus.english.greetings",
#     "chatterbot.corpus.english.conversations"
# )

##Change based on which chatbot
exit_conditions = (":q", "quit", "exit")
while True:
    query = input("> ")
    if query in exit_conditions:
        break
    else:
        print(f"🪴 {manual_chatbot.get_response(query)}") #library_chatbot
    



