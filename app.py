import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
#nltk.download('vader_lexicon')
from textblob import TextBlob
from transformers import pipeline
import boto3

#transformers
classifier = pipeline("sentiment-analysis")
print(classifier("I am so happy right now!"))


#NLTK
sent = SentimentIntensityAnalyzer()
print(sent.polarity_scores("I am so happy right now!"))


#Textblob
blob = TextBlob("I am so happy right now!")
print(blob.sentiment)


#Comprehend
comprehend_client = boto3.client('comprehend')
print(comprehend_client.detect_sentiment(Text = "I am so happy right now!", LanguageCode = 'en'))

