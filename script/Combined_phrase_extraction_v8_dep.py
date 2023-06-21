# # -*- coding: utf-8 -*-
import spacy
from spacy.lang.en.stop_words import STOP_WORDS


from heapq import nlargest
import shutil
import en_core_web_lg
nlp = spacy.load('en_core_web_lg')

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import pytextrank
from spacytextblob.spacytextblob import SpacyTextBlob
import re
from scipy import stats
import numpy as np
import csv



def rating_gen(FILE,ORG_NAME,TYPE):  

    df = FILE
    #os.chdir(INPUT_PATH)
    
    #path = INPUT_PATH +"//*.csv"
    
    #files = glob.glob(path)
    
    #df = pd.read_csv(files[0])
    #df = pd.read_csv(file)
    df.dropna(how = "all",axis =1,inplace =True)
    
    df.fillna("Not Applicable",inplace =True)
    df["Phrases"] = 1
    df["Sentiment"] = 1
    df["Score"] = 1
    df["Rating"] = 1
    df["Subjectivity"] = 1
    #df["Emotion"] = 1
    # load the model into spaCy
    nlp = spacy.load('en_core_web_lg')
    
    nlp.add_pipe("textrank")
    nlp.add_pipe('spacytextblob')
    
    
    #Custom punctuation
    from string import punctuation
    punctuation = punctuation +"“"+"”"
    
    for i in df.index:
        text = df["Answer"].values[i]
        
        # pass the text into the nlp function
        doc= nlp(text)
        
        ## The score of each word is kept in a frequency table
        #tokens=[token.text for token in doc]
        #freq_of_word=dict()
        lst =[]
        lst_phrase = []
    
        for word in doc:
                if word.text.lower() not in list(STOP_WORDS):
                    if word.text.lower() not in punctuation:
                        lst.append(word.text)
                        
        text2 = " ".join(lst)
    
    
        doc2 = nlp(text2)
        # examine the top-ranked phrases in the document
        for phrase in doc2._.phrases:

            if len(set(phrase.text.split(" "))) >=2:
                filtered = re.sub("[^a-zA-Z' ]+", '', phrase.text)
                lst_phrase.append(filtered)

                #print(lst_phrase)
        if len(lst_phrase) !=0:                 
            df["Phrases"].loc[i] = lst_phrase
        
        p = doc2._.blob.polarity                            
        s = doc2._.blob.subjectivity 
        
        if p>0 and p<=0.25:
            rating = "Mildly Positive"
            score = 3
            rating2 = "Investigating"
        elif p>0.25 and p<=0.5:
            rating = "Positive"
            score = 4
            rating2 = "Experimenting"
        elif p>0.5 and p<=1:
            rating = "Strongly Positive"
            score = 5
            rating2 = "Engaging"
        elif p<0 and p>=-0.25:
            rating = "Mildly Negative"
            score = 3
            rating2 = "Investigating"
        elif p<-0.25 and p>=-0.5:
            rating = "Negative"
            score = 2
            rating2 = "Discomfort"
        elif p<-0.5 and p>=-1:
            rating = "Strongly Negative"
            score = 1
            rating2 = "Hesitant"
        elif p==0:
            rating = "Neutral"
            score = 3
            rating2 = "Investigating"
        else:
            print("polarity issue")
            break
        
        df["Sentiment"].loc[i] = rating
        df["Score"].loc[i] = score
        df["Rating"].loc[i] = rating2
        
        if s>0 and s<=0.25:
            s_rating = "Mildly Subjective"
        elif s>0.25 and s<=0.5:
            s_rating = "Subjective"
        elif s>0.5 and s<=1:
            s_rating = "Strongly Subjective"
        elif s==0:
            s_rating = "Neutral"
        else:
            print("subjectivity issue")
            break
        
            
        df["Subjectivity"].loc[i] = s_rating
    
    #text = str(df["Phrases"])
    #text = re.sub("[^a-zA-Z' ]+", '', text)
    #text2 =text.split(" ")
    # text3 = []
    # for i in text2:
    #     if len(i)>2:
    #         text3.append(i)
            
    # text3 = " ".join(text3)
   
    # # Generate word cloud
    # word_cloud2 = WordCloud(
    #     width=6000,
    #     height=4000,
    #     random_state=123,
    #     background_color="purple",
    #     colormap="Set2",
    #     collocations=False,
    #     stopwords=STOPWORDS,
    # ).generate(text3)    
    
    #plt.imshow(word_cloud2)
    #plt.show()
    
    #Summarising resutls
    mode1 = stats.mode(df["Rating"])
    score_avg = np.mean(df["Score"])
    
    #Creating a dictionary
    d1 ={}
    
    #Storing values
    d1["Average_Score"]= round(score_avg,2)
    d1["Rating"] = mode1[0][0]
    
    #os.chdir("..")
    # INPUT_PATH = os.getcwd()
    # OUTPUT_PATH = INPUT_PATH+"//"+ORG_NAME+"//"+TYPE+"//"+"Analysed_file"
    # if os.path.exists(OUTPUT_PATH):
    #     shutil.rmtree(OUTPUT_PATH)
    # if not os.path.exists(OUTPUT_PATH):
    #     os.makedirs(OUTPUT_PATH)
    # os.chdir(OUTPUT_PATH)
    
    #Writing the dictionary as a CSV file
    # with open('Summary.csv', 'w') as f:  
    #     w = csv.DictWriter(f, d1.keys())
    #     w.writeheader()
    #     w.writerow(d1)
    
    #Saving the word cloud
    #word_cloud2.to_file(ORG_NAME+"_"+TYPE+"_interviews.jpg")
    #Saving the results
    #df.to_csv(ORG_NAME+"_"+TYPE+"_interviews_sentiment_analysis2222.csv",index = False)
    #z = os.getcwd()
    #os.chdir("..")
    x = "all good"
    lst = str(df["Phrases"])
    #lst = "lst"
    return df,d1,lst
