import os
import pandas as pd
import shutil


import spacy
from spacy import *
from spacy.lang.en.stop_words import STOP_WORDS
from heapq import nlargest
import glob
from wordcloud import WordCloud, STOPWORDS
import pytextrank
from spacytextblob.spacytextblob import SpacyTextBlob
import re
import scipy
from scipy import stats
import numpy as np
import csv
import urllib
import fitz
from PIL import Image
import io

# import requests
# url ="https://flaskdepapp.blob.core.windows.net/ttt/analysis_HR/HR/HR_Doc/JPE20_final.pdf?se=2023-06-17T12%3A37%3A37Z&sp=r&sv=2022-11-02&sr=b&sig=nU3zJam48k6G9/U0VsbNZh94z5nab3qycUp8YLHhz4c%3D"
# mem_area =requests.get(url)
# x = mem_area.content

# doc = fitz.open(stream=x, filetype="pdf")


# x  = fitz.open(stage_file.content)


def master_func(FILE,KEYWORD_LIST):

        

    def extraction(df):
        df.dropna(how = "all",axis =1,inplace =True)
        col_lst11 = ["Answer","Page Number"]
        df.fillna("Not Applicable",inplace =True)
        df.columns = col_lst11
        
        df["Phrases"] = 1
        df["Sentiment"] = 1.0
        df["Score"] = 1
        df["Rating"] = 1
        df["Subjectivity"] = 1.0
    
    
        #nlp = spacy.load(r"C:\Users\aditya.shah\AppData\Local\Programs\Python\Python39\Lib\site-packages\en_core_web_lg\en_core_web_lg-3.5.0")
        nlp = spacy.load('en_core_web_lg')
    
        nlp.add_pipe('spacytextblob')
        nlp.add_pipe("textrank")
        
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
            
            doc2 = nlp(str(text2))
            # examine the top-ranked phrases in the document
            for phrase in doc2._.phrases:
                if len(set(phrase.text.split(" "))) >=2:
                    filtered = re.sub("[^a-zA-Z' ]+", '', phrase.text)
                    lst_phrase.append(filtered)
    
                    #print(lst_phrase)
            if len(lst_phrase) !=0:                 
                df["Phrases"].loc[i] = lst_phrase
                
    
            
    
            p = float(doc2._.blob.polarity)                        
            s = float(doc2._.blob.subjectivity) 
            
            df["Sentiment"].loc[i] = p
            df["Score"].loc[i] =1
            df["Rating"].loc[i] = s
            
    
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
            
        
        text = str(df["Phrases"])
        text = re.sub("[^a-zA-Z' ]+", '', text)
        text2 =text.split(" ")
        text3 = []
        for i in text2:
            if len(i)>2:
                text3.append(i)
                
        text3 = " ".join(text3)
       
        # Generate word cloud
        word_cloud2 = WordCloud(
            width=6000,
            height=4000,
            random_state=123,
            background_color="purple",
            colormap="Set2",
            collocations=False,
            stopwords=STOPWORDS,
        ).generate(text3)
        return (df,word_cloud2)
    
    def search_analyse(FILE,KEYWORD):
        
        #os.chdir(PATH)
        
        #file = os.listdir()[0]
        
        #doc = fitz.open(file)
        #doc = fitz.open(FILE)
        doc = FILE 
        
        blocks = []
        for page in doc:
        
            output = page.get_text("blocks")                   
        
            previous_block_id = 0 # Set a variable to mark the block id
        
            for block in output:
        
                if block[6] == 0: # We only take the text
        
                    if previous_block_id != block[5]: # Compare the block number 
                        pass
                        #print("\n")
                        
                    #print(block[4])
                    blocks.append((block[4],page.number))
        
        filtered_blocks = []
        filtered_blocks_page_number = []
        
        for i in blocks:
            if KEYWORD in i[0]:
                filtered_blocks.append(i[0])
                filtered_blocks_page_number.append(i[1])
        
        df = pd.DataFrame([filtered_blocks,filtered_blocks_page_number])
        df = df.T
        
        values = extraction(df)
        df1 = values[0]
        op = df1.to_csv(encoding="utf-8")
        word_cloud2 = values[1]
        z = word_cloud2.to_array()
        im = Image.fromarray(z)
        img_byte_arr = io.BytesIO()
        im.save(img_byte_arr, format='jpeg')
        img_byte_arr = img_byte_arr.getvalue()
    
    
        
        return (KEYWORD,op,img_byte_arr)
    
    OP_LST = []
    
    FILE = FILE
    for i in range(len(KEYWORD_LIST)):
        KEYWORD = KEYWORD_LIST[i]
        OP = search_analyse(FILE,KEYWORD)
        OP_LST.append(OP)
        
    return OP_LST
            
lst =[(1,2,3),(34,56,78)]

for i in range(len(lst)):
    # print(lst[i][0])
    # print(lst[i][1])
    # print(lst[i][2])
    lst2 = [lst[i][0],lst[i][1],lst[i][2]]
    lst2 = lst2[1:]
    for i in range(len(lst2)):
        print(lst2[i])
