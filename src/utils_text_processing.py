from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import re
import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from spacy.attrs import ORTH
from matplotlib import pyplot as plt
from collections import defaultdict


def combine_strings(string_list):
    return " ".join(string_list)

##### TEXT PROCESSING #############
class Text_Processing:
    def __init__(self, column_to_process):
        self.column = column_to_process

    def create_processed_column(self,df):
        df['processed_text'] = df[self.column]
        return df

    def lowercasing(self,df):
        df.loc[:,'processed_text'] = df['processed_text'].apply(lambda x: str.lower(x))
        return df

    def remove_punctuation(self,df):
        df.loc[:,'processed_text'] = df['processed_text'].apply(lambda x: " ".join(re.findall(r"[\w']+",x)))
        return df

    ####### TOKENIZATION ######
    def nltk_Regexp_tokenizer(self,df):
        regexp=RegexpTokenizer(r'\w+')
        df.loc[:,'processed_text'] = df['processed_text'].apply(lambda row: regexp.tokenize(str(row)))
        return df

    def spacy_tokenizer(self,df):
        nlp = English()
        tokenizer = nlp.tokenizer
        df['processed_text'] = df['processed_text'].apply(lambda x: tokenizer(x))
        df['processed_text'] = df['processed_text'].apply(lambda row: [token.text for token in row])
        return df

    def spacy_tokenizer_custom(self,df):
        # DO not splitting "don't","can't","couldn't"
        nlp = English()
        tokenizer = nlp.tokenizer
        special_cases = {"don't": [{ORTH: "don't"}],
                         "can't": [{ORTH: "can't"}],
                         "doesn't": [{ORTH: "doesn't"}],
                         "isn't": [{ORTH: "isn't"}],
                         "haven't": [{ORTH: "haven't"}],
                         "wasn't": [{ORTH: "wasn't"}],
                         "weren't": [{ORTH: "weren't"}],
                         "wouldn't": [{ORTH: "wouldn't"}],}
        for case, token in special_cases.items():
            tokenizer.add_special_case(case, token)
        df['processed_text'] = df['processed_text'].apply(lambda x: nlp(x))
        df['processed_text'] = df['processed_text'].apply(lambda row: [token.text for token in row])
        return df


    ##### FILTERING ######
    def stop_words_removal(self, df,ss=False):
        stop_words = stopwords.words('english')
        negation_words = {"don't",'ain', 'aren', "aren't", 'couldn',"couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'mightn', "mightn't", 'mustn', "mustn't", 'needn',"I", "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", "won't", 'wouldn', "now","wouldn't","not","no","n't","can't"} #"than","before" put the nfs to 0.85 nut lower the patients
        additional = {"i","it","than"}
        #additional = {}
        # negations are kept
        filtered_stop_words = (set(stop_words))- negation_words #| additional
        if ss:
            filtered_stop_words = filtered_stop_words - additional
        # negations are kept

        df['processed_text'] = df['processed_text'].apply(lambda row: ([token for token in row if token not in (filtered_stop_words)]))
        df['processed_text']=df['processed_text'].apply(combine_strings)
        return df

    ##### STEMMING #######
    def nltk_Snowball_stemmer(self,df):
        stemmer = SnowballStemmer('english')
        df.loc[:,'processed_text'] = df['processed_text'].apply(lambda x: [stemmer.stem(word) for word in x])
        return df

    def nltk_porter_stemmer(self,df):
        stemmer = PorterStemmer()
        df.loc[:,'processed_text'] = df['processed_text'].apply(lambda x: [stemmer.stem(word) for word in x])
        return df

    ##### LEMMATIZER ####

    def nltk_lemmatize_text(self, df):
        ''' Given a list of tokens, apply WordNet lemmatization to each token.
        '''
        lemmatizer = WordNetLemmatizer() # verb tenses are kept
        df['processed_text'] = df['processed_text'].apply(lambda row: [lemmatizer.lemmatize(token) for token in row])
        return df

