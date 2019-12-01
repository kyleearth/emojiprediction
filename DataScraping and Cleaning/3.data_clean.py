import os
import re
import pandas as pd
import tweepy
import string
from textblob import TextBlob
import preprocessor as p
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from datetime import datetime
import fasttext


class data_clean:

    def __init__(self, INPUT_PATH, OUTPUT_PATH):

        # initail path, dataframe and emoji pattern
        self.INPUT_PATH = INPUT_PATH
        self.OUTPUT_PATH = OUTPUT_PATH
        self.df = pd.read_csv(INPUT_PATH, header=0)
        self.emoji_pattern = re.compile("["
                                        u"\U000000A0-\U0001FA90"
                                        "]+", flags=re.UNICODE)
        # output list
        self.new_entry = []

    def rm_punc(self, text):
        return ' '.join([w for w in word_tokenize(text) if w not in string.punctuation + '“”’‘-——…'])

    def rm_stops(self, text):
        return ' '.join([w for w in word_tokenize(text) if w not in set(stopwords.words('english'))])

    def date_trans(self, date):
        datetime_str = str(date[:-6])
        return datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')

    def clean(self):

        # cleaning loop
        for i in range(self.df.shape[0]):

            # tweepy preprocessing called for basic preprocessing
            clean_text = p.clean(self.df['text'][i])
            # remove emojis from tweet
            filtered_tweet = self.emoji_pattern.sub(r'', clean_text)
            # convert to lowercase
            filtered_tweet = filtered_tweet.lower()
            # remove punctuation
            filtered_tweet = filtered_tweet.translate(str.maketrans(
                '', '', string.punctuation))
            filtered_tweet = self.rm_punc(filtered_tweet)
            word_count = len(word_tokenize(filtered_tweet))
            # remove stop words
            filtered_tweet = self.rm_stops(filtered_tweet)

            # tokenizing
            # word_count = len(word_tokenize(filtered_tweet))
            # len(filtered_tweet.strip().split())
            if word_count > 5:  # must have 5 more words
                blob = TextBlob(filtered_tweet)
                Sentiment = blob.sentiment

                # seperate polarity and subjectivity in to two variables
                polarity = Sentiment.polarity
                subjectivity = Sentiment.subjectivity

                self.new_entry.append([self.date_trans(
                    self.df['created_at'][i]), filtered_tweet, polarity, subjectivity, word_count, self.df['label'][i]])
        # output as csv
        self.output()

    # output method
    def output(self):
        newdataset = pd.DataFrame(self.new_entry, columns=[
                                  'time', "filtered_tweet", "polarity", "subjectivity", 'word_count', 'label'])
        print("after cleaned - Flie Shape:", newdataset.shape)
        newdataset.to_csv(OUTPUT_PATH)


if __name__ == '__main__':

    ct = 0
    for file in range(10):
        print(">>>.Cleaning file: ", file)
        INPUT_PATH = './Data/Labelled/_Labelled' + str(ct) + "DATA.csv"
        OUTPUT_PATH = './Data/Labelled/Cleaned' + str(ct) + ".csv"
        clean = data_clean(INPUT_PATH, OUTPUT_PATH)
        clean.clean()
        ct += 1
        print("Done!")

