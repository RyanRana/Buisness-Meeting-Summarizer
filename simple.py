import sounddevice as sd
import soundfile as sf
import time
import speech_recognition as sr 
import os 
from pydub import AudioSegment
from pydub.silence import split_on_silence
import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from nltk.tokenize import sent_tokenize
import numpy as np
import networkx as nx
import re

def sync_record(filename, duration, fs, channels):
   print('recording')
   myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=channels)
   sd.wait()
   sf.write(filename, myrecording, fs)
   print('done recording')
x = int(input("How long is your meeting (enter in minutes):"))
x= 60*x
sync_record('sync_record.wav', 10, x, 1)
r = sr.Recognizer()
path = "sync_record.wav"
sound = AudioSegment.from_wav(path)  
chunks = split_on_silence(sound,min_silence_len = 500,silence_thresh = sound.dBFS-14,keep_silence=500,)
folder_name = "audio-chunks"
if not os.path.isdir(folder_name):
    os.mkdir(folder_name)
whole_text = ""
for i, audio_chunk in enumerate(chunks, start=1):

    chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
    audio_chunk.export(chunk_filename, format="wav")
    with sr.AudioFile(chunk_filename) as source:
        audio_listened = r.record(source)
        try:
            text = r.recognize_google(audio_listened)
        except sr.UnknownValueError as e:
            print("Error:", str(e))
        else:
            text = f"{text.capitalize()}. "
            print(chunk_filename, ":", text)
            whole_text += text


# Read the text and tokenize into sentences
def read_article(text):
    
    sentences =[]
    
    sentences = sent_tokenize(text)
    for sentence in sentences:
        sentence.replace("[^a-zA-Z0-9]"," ")

    return sentences
    

# Create vectors and calculate cosine similarity b/w two sentences
def sentence_similarity(sent1,sent2,stopwords=None):
    if stopwords is None:
        stopwords = []
    
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    
    all_words = list(set(sent1 + sent2))
    
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    
    #build the vector for the first sentence
    for w in sent1:
        if not w in stopwords:
            vector1[all_words.index(w)]+=1
    
    #build the vector for the second sentence
    for w in sent2:
        if not w in stopwords:
            vector2[all_words.index(w)]+=1
            
    return 1-cosine_distance(vector1,vector2)

# Create similarity matrix among all sentences
def build_similarity_matrix(sentences,stop_words):
    #create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences),len(sentences)))
    
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1!=idx2:
                similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1],sentences[idx2],stop_words)
                
    return similarity_matrix

    
nltk.download('stopwords')
nltk.download('punkt')
stop_words = stopwords.words('english')
summarize_text = []
y = int(input("How many sentances do you want in summarization"))
sentences = read_article(wordtext)
sentence_similarity_matrix = build_similarity_matrix(sentences,stop_words)
sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
scores = nx.pagerank(sentence_similarity_graph)
ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)),reverse=True)
for i in range(y):
    summarize_text.append(ranked_sentences[i][1])
return " ".join(summarize_text),len(sentences)

