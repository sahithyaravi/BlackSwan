import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download the necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

remove_list = ['fail', 'fails', 'funny', 'best', 'compilation', 'failfactory', 'month', 'failarmy', 
               'throwback', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 
               'october', 'november', 'december', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 
               'saturday', 'sunday', 'week', 'get', 'got', 'enough', 'going', 'year', 'ok', 'let', 'guess', 'gone', 'wrong']

def preprocess_text(text):
    # Tokenize the text into words
    words = word_tokenize(text)
    
    # Convert to lower case
    words = [word.lower() for word in words]
    
    # Remove punctuation
    words = [word for word in words if word.isalnum()]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Remove numbers
    words = [word for word in words if not word.isdigit()]

    # Remove words in remove_list
    words = [word for word in words if word not in remove_list]

    words = [word for word in words if not word.startswith('failarmy')]
    
    return words

def get_word_frequency(words):
    # Calculate word frequency
    freq_dist = nltk.FreqDist(words)
    return freq_dist

with open("VAR_Data.json") as f:
    data = json.load(f) 

words = []
for d in data:
    text = d['original']
    words = words + preprocess_text(text)

freq_dist = get_word_frequency(words)
print(freq_dist.most_common(50))

wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(freq_dist)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.savefig('wordcloud.png')
    