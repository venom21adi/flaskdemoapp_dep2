# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 11:27:08 2023

@author: aditya.shah
"""
import re
import os
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import io

def wordcloud(PHRASES,ORG_NAME,TYPE):
    text = str(PHRASES)
    #text = str(df["Phrases"])
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

    #z = word_cloud2.to_file(ORG_NAME+"_"+TYPE+"_interviews.jpg")
    z = word_cloud2.to_array()
    im = Image.fromarray(z)
    img_byte_arr = io.BytesIO()
    im.save(img_byte_arr, format='jpeg')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr 
