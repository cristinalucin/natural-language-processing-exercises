import pandas as pd
import numpy as np

#Parsing text functions
import unicodedata
import nltk
from nltk.corpus import stopwords



def basic_clean(string):
    '''This function applies basic text cleaning to a given
    string input, and returns the string after the methods have
    been applied
    '''
    #Make string lowercase
    string = string.lower()
    #Normalize unicode characters
    string = unicodedata.normalize('NFKD', string).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    #Replace everything not a letter, number, whitespace or a single quote
    string = re.sub('[^a-z0-9\'\s]', '', string)
    
    return string


def tokenize(string, return_str=True):
    '''This function tokenizes a given string
    value which is provided
    '''
    #Initiate the object
    tokenize = nltk.tokenize.ToktokTokenizer()
    #Tokenize the string, default of return string true
    string_tokenized =tokenize.tokenize(string, return_str)
    
    return string_tokenized

def stem(string):
    '''This function accepts a string and returns a string
    which has been stemmed
    '''
    #Create the stem object
    ps = nltk.porter.PorterStemmer()
    #Stem the string input
    stems = [ps.stem(word) for word in string.split()]
    #Rejoin the stems to reform string
    string_stemmed = ' '.join(stems)
    #Return stemmed string
    return string_stemmed


def lemmatized(string):
    '''This function accepts a string input and applies lemmatization
    to the string, returning the lemmatized string
    '''
    #Initialize the object
    wnl = nltk.stem.WordNetLemmatizer()
    #Use the object to lemmatize the words
    lemmas = [wnl.lemmatize(word) for word in string.split()]
    #Rejoin the lemmatized words in the string to reform the string
    string_lemmatized = ' '.join(lemmas)
    
    return string_lemmatized


def remove_stopwords(string, extra_words= [], exclude_words= [], language='english'):
    #define the language used, creating a list
    stopword_language = stopwords.words(language)
    #utilizing set casting to remove excluded stopwords
    stopword_language = set(stopword_language) - set(exclude_words)
    #add in extra words to my stopwods set using a union
    stopword_language = stopword_language.union(set(extra_words))
    #Turn string into a list
    words = string.split()
    #Apply filter to string
    filtered_words = [w for w in words if w not in stopword_language]
    #Place string back together
    string_without_stopwords = ' '.join(filtered_words)
    
    return string_without_stopwords


#Function to rule them all
def organize_blog_content(df):
    '''This function takes all functions from the prepare lesson and aggregates
    them into one, providing a dataframe with columns of individualized applied
    functions for further use
    '''
    #Rename original column to content
    df = df.rename(columns={'content' : 'original'})
    #Use basic clean function to create a clean and tokenized column
    df['clean'] = df.original.apply(basic_clean).apply(tokenize, return_str=True).apply(remove_stopwords)
    #Create Stemmed Column
    df['stemmed'] = df.clean.apply(stem)
    #Create Lemmatized column
    df['lemmatized'] = df.clean.apply(lemmatized)
    
    return df