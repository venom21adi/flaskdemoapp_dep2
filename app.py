
from werkzeug.utils import secure_filename
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions

import os
from datetime import datetime, timedelta
import sys
sys.path.append("..//script")
#-----------------------------------

from flask import *  


import pandas as pd
import openpyxl
import numpy as np
import fitz
import requests
import re

pd.options.display.max_colwidth = 120
pd.set_option('display.max_colwidth', None)

MASTER_USER = "test"
MASTER_PASSWORD = "test"
USERS = {"test":"test",
        "user1@jpe.com":"user1",
        "user2@jpe.com":"user2",
        "user3@jpe.com":"user3",
        "user4@jpe.com":"user4",
        "user5@jpe.com":"user5",
        }   

#-----------------------------------
app = Flask(__name__)
app.secret_key = os.urandom(24)

app.config.from_pyfile('config.py')
account = app.config['ACCOUNT_NAME']   # Azure account name
key = app.config['ACCOUNT_KEY']      # Azure Storage account access key  
connect_str = app.config['CONNECTION_STRING']
#container = app.config['CONTAINER'] # Container name
allowed_ext = app.config['ALLOWED_EXTENSIONS'] # List of accepted extensions
max_length = app.config['MAX_CONTENT_LENGTH'] # Maximum size of the uploaded file



blob_service_client = BlobServiceClient.from_connection_string(connect_str)

#----------------------------------Functions-----------------------------------

#------------------Importing survey key-------------------
#------------------Importing survey key-------------------
def survey_key(Survey_key_file,Analysis_file):
    lst = []
    file = pd.read_excel(Survey_key_file)
    cols = file.columns
    for i in range(len(cols)):
        if "Unnamed" in cols[i]:
            file.columns = [file.iloc[0,0]]*file.shape[1]
            file = file.iloc[1:,:]
            file.index = file.iloc[:,0]
            break
        else:
            file = file
    lst.append(file)

    #------------------------------------
    
    """
    The following part of the script takes the data frame having more than 3 columns and 
    removes the first column.
    
    This has to be done as the first column is actually same as the index.
    
    This is essential in reducing the redundancy.
    
    """
    
    for i in range(len(lst)):
        if lst[i].shape[1]>3:
            col = lst[i].shape[1]
            lst[i] = lst[i].iloc[:,1:]
        else:
            pass
    
    
    #------------------Importing the excel file----------------
    #os.chdir(DATA_FILE_PATH)
    
    #filename1 = os.listdir()
    #data_raw = pd.ExcelFile(Analysis_file)
    # data_raw = Analysis_file
    # sheet_names1 = data_raw.sheet_names
    
    file1 = pd.read_excel(Analysis_file)
    
    df1 = file1
    
    
    
    #Extracting the column names from the survey results file
    cols2 = df1.columns
    
    #Making the respondent column as the index    
    for i in cols2:
        if "Respondant" or "Respondent"in i:
            df1.index = df1.loc[:,i]
            break
        
    
    #Removing the Unnamed columns from the survey results file
    
    
    cols2 = list(cols2)
    
    for i in range(len(cols2)):
        if "Unnamed" in cols2[i]:
            cols2[i] = cols2[i-1]
    
    
    #Assigning the fiktered column names
    df1.columns = cols2
    
    #Dropping the columns with all NaN values
    df1.dropna(how = "all",axis=1, inplace =True) 
    """
    Extracting the unique column names from the dataframe
    
    These column names are the questions from the survey key
    
    """
    col_lst = []
    
    for i in lst:
        col_lst.append(set(i.columns))
    
        
    """
    Extracting the unique column names from the dataframe
    
    These column names are the questions from the survey key
    
    """
    d1_keys = []
    for i in range(len(lst)):
        d1_keys.append(lst[i].columns[0])
    
    """
    Extracting column names from the survey file and check whether it matches
    with the survey key list.
    
    If it matches we put it into the cols_filtered list otherwise cols_unfiltered
    list.
    
    The cols_unfiltered will help us identify any non-matching columns.
    """
    cols_filtered = []
    cols_unfiltered = []
    for i in cols2:
        if i in d1_keys:
            cols_filtered.append(i)
        else:
            cols_unfiltered.append(i)
    
    y = list(df1.columns)
    for i in y:
        if "describe" in i:
            t = y.index(i)
            break
    #Creating a new dataframe with the filtered columns
    df2 = df1.loc[:,set(cols_filtered)] 
    
    demographic_cols = df1.iloc[:,t:]
    #Creating a new dataframe with the filtered columns
    df2 = df1.loc[:,set(cols_filtered)] 
    df2 = pd.concat([df2,demographic_cols],axis = 1)
    
    # This is a manual extraction of the columns from the survey file
    # This needs to be removed from the production version
    cols_not_matching1 = set(df1.iloc[:,5:-9].columns)
    
    #Extracting the column names from the df2 
    cols_not_matching2 = set(df2.columns)
    
    
    # A check to see whethere there is any mismatch between column names and reporting
    # This is manual effort and should be removed in the production version
    if cols_not_matching1 != cols_not_matching2:
        print("Something is wrong")
    else:
        print("No issues detected")
    
    #------------------Scored Data---------------
    #Creating another copy of the dataframe
    df4 = df2  
    
    """
    Creating a dictionary of with list of questions and tying them up with
    the dataframes so that we can pull the dataframes based on the keys
    """
    
    d1 = dict(zip(d1_keys, lst))
    
    
    #------------------Scoring Approach---------------
    #Extracting the actual column names from the df4 (survey results)
    lst_new = list(set(df4.columns))
    
    #Calculating and storing a questionwise count of the questions appeared in the df4 
    count_f = {}
    for i in list(set(cols_filtered)):
        count_f[i] = list(df4.columns).count(i)
    
    """
    We are creating a multi-index version of the survey file
    
    So we are extracting the columns and the first row. 
    This will be combined and made into a multi-level index
    
    """
    index1 = list(df4.columns)
    index2 = list(df4.iloc[0,:])
    index3 = list(zip(index1,index2))
    
    index4 = pd.MultiIndex.from_tuples(index3,names=["Question","Category"])
    
    #Creating a copy of the dataframe
    df5 = df4
    #Transposing the dataframe
    df6 = df5.T
    
    #Resetting the index and dropping the exsiting one
    df6 = df6.reset_index(drop = True)
    
    #Transposing the dataframe
    df6 = df6.T
    
    #Assignging the new multi-level index
    df7 = pd.DataFrame(data = df6.values,columns =index4)
    
    #Ensuring the indices are the same
    df7.index = df5.index
    
    #Dropping the first row
    df8 = df7.iloc[1:,:]
    
    #Dropping all the rows and columns with all the values with NaN value
    df8.dropna(how = "all",axis =0,inplace= True)
    df8.dropna(how = "all",axis =1,inplace= True)
    # df8.dropna(how = "any",axis =0,inplace= True)
    
           
    #Extracting the actual column names from the df4 (survey results)
    lst_new = list(set(df8.columns))
    
    #Calculating and storing a questionwise count of the questions appeared in the df8 
    count_f = {}
    count1 = []
    
    for i in range(len(df8.columns)):
                  count1.append(df8.columns[i][0])
                  
    for i in set(count1):
        count_f[i] = count1.count(i)
    
    
    """
    Creating a new dictionary.
    
    We are going column by column in df8 (both levels), taking the question from the level
    then checking whether that is in the d1_keys then use that and check what is the shape of 
    that dataframe in the survey key dictionary d1.
    
    If it is 3 
    
    We pull the rating of the same in t4 
    take the numerical values in t5
    Zip it and make the rating and the score into a dictionary
    assign the same to the d4_lst.
    
    So now, we have questions their sub-category and its rating and score.
    
    If it is 5 
    
    We transpose and do a similar procedure.
    
    
    """
    d4_lst = {}
    for i in df8.columns:
        if i[0] in d1_keys and d1[i[0]].shape[1]==3:
            # t1 = i
            # t2 = df8[i[0]].iloc[0]
            # t3 = (t1,t2)
            t4 = d1[i[0]][i[0]].iloc[:,0].values
            t5 = d1[i[0]].iloc[:,-1].values
            t6 = dict(zip(t4,t5))
            d4_lst[i] = t6
        elif i[0] in d1_keys and d1[i[0]].shape[1]==5:
            t1 = i[0]
            t2 = d1[i[0]].T[i[1]].values
            t3 = d1[i[0]].T["Score"].values
            t4 = dict(zip(t2,t3))
            d4_lst[i] = t4
    
    """
    We are splitting the respones by ( and removing everything after this.
                                      
    
                                      
    """
    for i in df8.columns:
        df8[i] = df8[i].str.split("(").str[0]
    
    # Creating a copy of the dataframe
    df9 = df8
    
    for i in df9.columns:
        df9[i] = df9[i].str.strip()
        
    for j in df9:
        values = list(set(df9.loc[:,j]))
        if j in d4_lst:
            values_key = d4_lst[j]
            df9[j].replace(d4_lst[j],inplace=True)      
    g = "Analysis is completed"
    return df9
#---------------------------------------------------------------------------
import spacy
import matplotlib.pyplot as plt
import pytextrank
from spacytextblob.spacytextblob import SpacyTextBlob
from scipy import stats
from spacy.lang.en.stop_words import STOP_WORDS

def rating_gen(FILE,ORG_NAME,TYPE):   
    import en_core_web_lg
    
    df = FILE
   
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
   
   
    
    #Summarising resutls
    mode1 = stats.mode(df["Rating"])
    score_avg = np.mean(df["Score"])
    
    #Creating a dictionary
    d1 ={}
    
    #Storing values
    d1["Average_Score"]= round(score_avg,2)
    d1["Rating"] = mode1[0][0]
    

    
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
    """
    return "test"
    """
#------------------------------------------------------------------
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
#----------------------------------------------------------------------------
def main_table(dataframe_file):
    df = dataframe_file
    
    #---------------------------------------------------------------------
    df  = df.iloc[:,1:]
    
    #---------------------------------------------------------------------
    l1 = list(df.columns)
    l2 = []
    for i in range(len(l1)):
        if i != len(l1)-1:
            if l1[i] in l1[i+1]:
                ctr = True
                q = l1[i]
                l2.append(q)
            else:
                ctr = False
    #---------------------------------------------------------------------
    df = df.dropna(how ="all",axis = 1)
    df = df.dropna(axis = 0)
    #---------------------------------------------------------------------
    df_count = df.iloc[1:,:]
    #---------------------------------------------------------------------
    lst = df.columns
    
    d1 = {
          "Gender":"Gender",
          "Ethnic":"Ethnicity",
          "Generation":"Generation",
          "region":"Region",
          "working professional":"Experience (Total)",
          "describes my position":"Designation",
          "my current position":"Experience (Current Position)",
          "have been with Populous":"Experience (Current Organisation)"
          }
    
    lst1 = list(d1.keys())
    
    new_col = []
    
    for i in lst1:
        for j in lst:
            if i in j:
                new_col
    
    for i in lst1:
        for j in lst:
            if i in j:
                df.loc[0][j] = d1[i]
    
    lst2 = list(lst)  
    
    
    #-----------------------------------------------------------------------          
    
    for i in lst1:
        for j in lst2:
            if i in j:
                split_column = df.columns.get_loc(j)
                break
        break
    
    df_scored = df.iloc[:,:split_column]
    
    lst_Q_DIBS_BIG9 = []
    
    for i in l2:
        for j in range(len(list(set(df_scored.iloc[0,:])))):
            lst_Q_DIBS_BIG9.append((i,df_scored.iloc[0,j]))
            
            
    df_demographic = df.iloc[:,split_column:]
    
    df_scored.columns = lst_Q_DIBS_BIG9
    
    lst_questions = df_scored.columns
    
    lst_demo_questions = df_demographic.columns
    
    df_combined = pd.concat([df_scored,df_demographic],axis = 1)
    #---------------------------Leadership----------------------------------
    lst_leadership = []
    
    for i in df_combined.columns:
        if "Designation" in df_combined[i].iloc[0]:
            ind_leadership = df_combined.columns.get_loc(i)
            col_leadership = i
            lst_leadership.append(set(list(df_combined[i].values)[1:]))
            #df.drop(i,inplace =True,axis =1)
            break
    
    # for i in lst_leadership[0]:
    #     print(i)
    
    
    master_lst =[]
    
    #row_count = []
    
    for i in lst_leadership[0]:
        designation =i
        dfx12 = df_combined[df_combined[col_leadership]==i]
        #row_count.append((i,dfx12.shape[0]))
    
        #---------------------------------------Reassignment-------------------------
        df_scored = dfx12.iloc[:,:split_column]
        
        df_demographic = dfx12.iloc[:,split_column:]
        
        for i in df_demographic.columns:
            if "describes my position" in i:
                ind_des = df_demographic.columns.get_loc(i)
                df_demographic.drop(i,inplace = True, axis =1)
                break
                
        #----------------------------------BIG9---------------------------
    
    
        d2 = {}
        
        dfx = df_demographic
        
        for i in dfx.columns:
            d2[i] = list(pd.unique(dfx[i]))
        
        lst1 = list(d2.keys())
        
        dfx2 = dfx12
        
        lst_col = list(lst_questions) + list(lst_demo_questions)
        
        dfx2.columns = lst_col
    
    
    
    
        #--------------------------------Combined--------------------------
        import numpy as np
        
        d3 = {}
        
        for k in df_scored.columns:
            d4 = {}
            for i in lst1:
                d5 ={}
                for j in d2[i]:
                    d6 = {}
                    #print(j)
                    a = np.array(dfx2[k][dfx2[i]==j],dtype =float).mean()
                    # #print(a)
                    b = max(np.array(dfx2[k][dfx2[i]==j],dtype =float))
                    # #print(b)
                    c = min(np.array(dfx2[k][dfx2[i]==j],dtype =float))
                    # #print(c)
                    d = len(np.array(dfx2[k][dfx2[i]==j]))
                    d6["mean"] = a
                    d6["max"] = b
                    d6["min"] = c
                    d6["Count"] = d
                    d5[j] = d6
                d4[i]=d5
            d3[k] = d4         
                    
        lstx1 = []
        lstx2 = [] 
        lstx3 = [] 
        lstx4 = []
        lstx5 = []
        lstx6 = []
        lstx7 = []
        dummy = []
        ctr = 0
        for i in list(d3.keys()):
            for j,k in zip(range(len(list(d3[i].keys()))),range(len(list(d3[i].values())))):
                #for k in range(len(list(d3[i].values()))):
                dictionary = list(d3[i].values())[k]
                keys = dictionary.keys()
                for l in keys:
                    lstx1.append(i)
                    lstx2.append(list(d3[i].keys())[j])
                    lstx3.append(l)
                    dummy.append(l)
                    value = dictionary[l]
                    lstx4.append(dictionary[l]['mean'])
                    lstx5.append(dictionary[l]['min'])
                    lstx6.append(dictionary[l]['max'])
                    lstx7.append(dictionary[l]['Count'])
        
        dfx3 = pd.DataFrame(data = {"Question":lstx1,"Demographic Categories":lstx2,"Categories":lstx3,"Mean":lstx4,"Max":lstx5,"Min":lstx6,"Count":lstx7})
        
        #------------------Main Table--------------------------
        
        questions = list(dfx3.iloc[:,0])
        col1 = []
        col2 = []
        for i in questions:
            col1.append(i[0])
            col2.append(i[1])
        
        dfx3.iloc[:,0] = col1
        #dfx3["DIBS BIG 9 Category"] = col2
        dfx3.insert(1, "DIBS BIG 9 Category", col2)
        dfx3 = dfx3.iloc[:,1:]
        dfx3.insert(0, "Designation", designation)
        
        # df3
        
        dfx4 = dfx3.groupby(by = ["Designation","DIBS BIG 9 Category","Demographic Categories","Categories"]).mean()
        
        master_lst.append(dfx4)
    
    
    master_df = pd.concat(master_lst)
    return master_df
    
def persona(dataframe_file):
    # os.chdir(SOURCE_PATH)

    
    # file = os.listdir(SOURCE_PATH)[0]
    
    # data = pd.read_csv(file)
    
    # df = data
    
    df = dataframe_file


    #---------------------------------------------------------------------
    df  = df.iloc[:,1:]
    
    #---------------------------------------------------------------------
    l1 = list(df.columns)
    l2 = []
    for i in range(len(l1)):
        if i != len(l1)-1:
            if l1[i] in l1[i+1]:
                ctr = True
                q = l1[i]
                l2.append(q)
            else:
                ctr = False
    #---------------------------------------------------------------------
    df = df.dropna(how ="all",axis = 1)
    df = df.dropna(axis = 0)
    #---------------------------------------------------------------------
    df_count = df.iloc[1:,:]
    
    lst = df.columns
    
    
    d1 = {
          "Gender":"Gender","Ethnic":"Ethnicity",
          "Generation":"Generation",
          "region":"Region",
          "working professional":"Experience (Total)",
          "describes my position":"Designation",
          "my current position":"Experience (Current Position)",
          "have been with Populous":"Experience (Current Organisation)"
          }
    
    lst1 = list(d1.keys())
    
    for i in lst1:
        for j in lst:
            if i in j:
                df.loc[0][j] = d1[i]
    
    lst2 = list(lst)            
    
    for i in lst1:
        for j in lst2:
            if i in j:
                split_column = df.columns.get_loc(j)
                break
        break
    
    df_scored = df.iloc[:,:split_column]
    
    lst_Q_DIBS_BIG9 = []
    
    for i in l2:
        for j in range(len(list(set(df_scored.iloc[0,:])))):
            lst_Q_DIBS_BIG9.append((i,df_scored.iloc[0,j]))
            
            
    df_demographic = df.iloc[:,split_column:]
    
    df_scored.columns = lst_Q_DIBS_BIG9
    
    lst_questions = df_scored.columns
    
    lst_demo_questions = df_demographic.columns
    
    #----------------------------------BIG9---------------------------
    
    
    d2 = {}
    
    dfx = df_demographic.iloc[1:,:]
    
    for i in dfx.columns:
        d2[i] = list(pd.unique(dfx[i]))
    
    lst1 = list(d2.keys())
    
    dfx2 = df.iloc[1:,:]
    
    
    #------------------------------------------------------------------------------
    
    lst_col = list(lst_questions) + list(lst_demo_questions)
    
    dfx2.columns = lst_col
    
    
    #--------------------------------Combined--------------------------

    
    d3 = {}
    
    for k in df_scored.columns:
        d4 = {}
        for i in lst1:
            d5 ={}
            for j in d2[i]:
                d6 = {}
                #print(j)
                a = np.array(dfx2[k][dfx2[i]==j],dtype =float).mean()
                # #print(a)
                b = max(np.array(dfx2[k][dfx2[i]==j],dtype =float))
                # #print(b)
                c = min(np.array(dfx2[k][dfx2[i]==j],dtype =float))
                # #print(c)
                d = len(np.array(dfx2[k][dfx2[i]==j]))
                d6["mean"] = a
                d6["max"] = b
                d6["min"] = c
                d6["Count"] = d
                d5[j] = d6
            d4[i]=d5
        d3[k] = d4         
                
    lstx1 = []
    lstx2 = [] 
    lstx3 = [] 
    lstx4 = []
    lstx5 = []
    lstx6 = []
    lstx7 = []
    dummy = []
    ctr = 0
    for i in list(d3.keys()):
        for j,k in zip(range(len(list(d3[i].keys()))),range(len(list(d3[i].values())))):
            #for k in range(len(list(d3[i].values()))):
            dictionary = list(d3[i].values())[k]
            keys = dictionary.keys()
            for l in keys:
                lstx1.append(i)
                lstx2.append(list(d3[i].keys())[j])
                lstx3.append(l)
                dummy.append(l)
                value = dictionary[l]
                lstx4.append(dictionary[l]['mean'])
                lstx5.append(dictionary[l]['min'])
                lstx6.append(dictionary[l]['max'])
                lstx7.append(dictionary[l]['Count'])
    
                    
              
    dfx3 = pd.DataFrame(data = {"Question":lstx1,"Demographic Categories":lstx2,"Categories":lstx3,"Mean":lstx4,"Max":lstx5,"Min":lstx6,"Count":lstx7})
    
    
    #------------------Main Table--------------------------
    
    questions = list(dfx3.iloc[:,0])
    col1 = []
    col2 = []
    for i in questions:
        col1.append(i[0])
        col2.append(i[1])
    
    dfx3.iloc[:,0] = col1
    #dfx3["DIBS BIG 9 Category"] = col2
    dfx3.insert(1, "DIBS BIG 9 Category", col2)
    
    dfx3 = dfx3.iloc[:,1:]
    
    dfx4 = dfx3.groupby(by = ["DIBS BIG 9 Category","Demographic Categories","Categories"]).mean()
    
    dfx_database_level = dfx4

    
    
    #-----------------------------------Persona------------------------
    dfx4 = dfx3
    
    dfx4_mean_groupby = dfx4.groupby(["Demographic Categories","Categories"]).mean()["Mean"]
    
    dfx4_mean_groupby = pd.DataFrame(dfx4_mean_groupby)
    
    
    x = dfx4_mean_groupby.reset_index("Categories")
    
    most_inclusive_params = x.groupby("Demographic Categories").max()
    
    
    
    least_inclusive_params = x.groupby("Demographic Categories").min()
    
    
    
    
    #----------------------------------------------------------
    
    dfx_d = df_demographic
    
    dfx_d = dfx_d.iloc[1:,:]
    
    des = dfx_d.describe()
    
    most_represented_params = des
    
    lst_dem_min1 = []
    lst_dem_min2 = []
    lst_dem_max1 = []
    lst_dem_max2 = []
    for i in range(len(dfx_d.columns)):
        lst_min1 = []
        lst_min2 = []
        lst_max1 = []
        lst_max2 = []
        dfx_l1 = dfx_d.iloc[:,i].value_counts()
        dfx_l1_op = dfx_l1.sort_values()
        index_length_min = len(dfx_l1[dfx_l1==dfx_l1_op[0]].index)
        index_length_max = len(dfx_l1[dfx_l1==dfx_l1_op[-1]].index)
        for j in range(index_length_min):
            lst_min1.append(dfx_l1[dfx_l1==dfx_l1_op[0]].index[j])
        lst_min2.append(dfx_l1_op[0].item())
        for k in range(index_length_max):
            lst_max1.append(dfx_l1[dfx_l1==dfx_l1_op[-1]].index[k])
        lst_max2.append(dfx_l1_op[-1].item())
        lst_dem_min1.append(tuple(lst_min1))
        lst_dem_min2.append(dfx_l1_op[0].item())
        lst_dem_max1.append(tuple(lst_max1))
        lst_dem_max2.append(dfx_l1_op[-1].item())
        
    df_dem_presentation = pd.DataFrame({"Categories":list(dfx_d.columns),"Most Represented Categories":lst_dem_max1,"Count (Most Represented)":lst_dem_max2,"Least Represented Categories":lst_dem_min1,"Count (Least Represented)":lst_dem_min2})
    
    return (df_count,most_inclusive_params,least_inclusive_params,df_dem_presentation,dfx_database_level)

def dashboard_creation2(dataframe_file):
    x = dataframe_file

    master_df= main_table(dataframe_file)
    persona_dfs = persona(dataframe_file)
    

    #---------------------------------Extracting the dataframes-----------------------------
    df_count = persona_dfs[0]
    most_inclusive_params = persona_dfs[1]
    least_inclusive_params = persona_dfs[2]
    df_dem_presentation = persona_dfs[3]
    dfx_database_level= persona_dfs[4]
    
    
    
    #------------------------Renaming--------------------------------
    
    d1 = {
          "Gender":"Gender",
          "Ethnic":"Ethnicity",
          "Generation":"Generation",
          "region":"Region",
          "working professional":"Experience (Total)",
          "describes my position":"Job Title",
          "my current position":"Experience (Current Position)",
          "have been with Populous":"Experience (Current Organisation)"
          }
    
    most_inclusive_resetted_index = most_inclusive_params.reset_index()
    
    cols = list(most_inclusive_resetted_index.iloc[:,0])
    
    d2 = {}
    
    d1_keys = list(d1.keys())
    
    for i in range(len(d1_keys)):
        for j in cols:
            if d1_keys[i] in j:
                d2[j] = d1[d1_keys[i]]
                
    master_df_resetted_index = master_df.reset_index()
    
    master_df_resetted_index2 = master_df_resetted_index.replace(d2)
    
    master_df_final = master_df_resetted_index2
    
    most_inclusive_params = most_inclusive_params.reset_index()
    
    most_inclusive_params = most_inclusive_params.replace(d2)
    
    least_inclusive_params = least_inclusive_params.reset_index()
    
    least_inclusive_params = least_inclusive_params.replace(d2)
    
    df_dem_presentation = df_dem_presentation.reset_index()
    
    df_dem_presentation = df_dem_presentation.replace().replace(d2)
    
    dfx_database_level = dfx_database_level.reset_index()
    
    dfx_database_level = dfx_database_level.replace().replace(d2)
    
    #---------------------------------Changing the Path-----------------------------
    # global OUTPUT_PATH
    # OUTPUT_PATH = DESTINATION_PATH.get()
    # OUTPUT_PATH = OUTPUT_PATH +"\\"+ORG_NAME.get()
    # if os.path.exists(OUTPUT_PATH):
    #     shutil.rmtree(OUTPUT_PATH)
    # if not os.path.exists(OUTPUT_PATH):
    #     os.makedirs(OUTPUT_PATH)
    # os.chdir(OUTPUT_PATH)
    
    #----------------------------------Writing the files----------------------------
    op4 = df_count.to_csv(encoding= "utf-8")
    op5 = master_df_final.to_csv(encoding= "utf-8")
    op6 = dfx_database_level.to_csv(encoding= "utf-8")
    op7 =  most_inclusive_params.to_csv(encoding= "utf-8")
    op8 = least_inclusive_params.to_csv(encoding= "utf-8")
    op9 = df_dem_presentation.to_csv(encoding= "utf-8")
    # os.chdir("..")
    #op4 = df_count
    #op5 = master_df_final
    #op6 = dfx_database_level
    #op7 =  most_inclusive_params
    #op8 = least_inclusive_params
    #op9 = df_dem_presentation
    #lst_master =[op4,op5,op6,op7,op8,op9]
    
    lst_master =[op4,op5,op6,op7,op8,op9]
    return lst_master

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
#----------------------------------------------------------------------------    



def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1] in allowed_ext

def delete_container(blob_service_client, client):
    container_client = blob_service_client.get_container_client(container=client)
    container_client.delete_container()

def create_blob_container(blob_service_client,client):
    if session["flag"] =="survey":
        try:
            container_client = blob_service_client.create_container(name=client)
        except:
            pass
            # delete_container(blob_service_client, client)
            # container_client = blob_service_client.create_container(name=client)
    elif session["flag"] =="fi":
        try:
            container_client = blob_service_client.create_container(name=client)
        except:
            pass
            # delete_container(blob_service_client, client)
            # container_client = blob_service_client.create_container(name=client)
    elif session["flag"] =="HR":
        try:
            container_client = blob_service_client.create_container(name=client)
        except:
            pass
            # delete_container(blob_service_client, client)
            # container_client = blob_service_client.create_container(name=client)
    return "Container Created"

def delete_blob(blob_service_client):
    container = session["client"]
    container_client_analysis = blob_service_client.get_container_client(container=container)
    x = "test"
    blob_lst_delete =[]
    if session["flag"] == "survey" and session["type_i"] == "inclusivity": 
        for blob_i in container_client_analysis.list_blobs():
            blob_lst_delete.append(blob_i['name'])
        TYPE = "analysed_file"
        blob_lst_delete = [k for k in blob_lst_delete if TYPE in k]
        if len(blob_lst_delete)!=0:
            blob_client = blob_service_client.get_blob_client(container=container,blob =blob_lst_delete[0])
            blob_client.delete_blob()
    elif session["flag"] == "fi" and session["type_i"] == "focus":
        for blob_i in container_client_analysis.list_blobs():
            blob_lst_delete.append(blob_i['name'])
        TYPE = "analysed_file"
        blob_lst_delete = [k for k in blob_lst_delete if TYPE in k]
        for blob_i in blob_lst_delete:
                blob_client = blob_service_client.get_blob_client(container=container,blob =blob_i)
                blob_client.delete_blob()
    return None
                

def upload_file():
    x = delete_blob(blob_service_client)
    client = session["client"]
    if request.method == 'POST':
        img = request.files['file']
        if img and allowed_file(img.filename):
            filename = secure_filename(img.filename)
            img.save(filename)
            if session["flag"] == "survey":
                blob_client = blob_service_client.get_blob_client(container = client, blob = "inclusivity_analysis_survey/"+filename)
            elif session["flag"] == "fi" and session["type_i"]== "focus":
                blob_client = blob_service_client.get_blob_client(container = client, blob = "analysis_fi/focus/"+filename)
            elif session["flag"] == "fi" and session["type_i"]== "individual":
                blob_client = blob_service_client.get_blob_client(container = client, blob = "analysis_fi/individual/"+filename)
            elif session["flag"] == "HR" and session["type_i"]== "HR_Doc":
                blob_client = blob_service_client.get_blob_client(container = client, blob = "analysis_HR/HR/HR_Doc/"+filename)
            session["Analysis_file"] =  img.filename
            with open(filename, "rb") as data:
                try:
                    blob_client.upload_blob(data, overwrite=True)
                    msg = "Upload Done ! "
                except:
                    pass
            os.remove(filename)     
    return msg

def upload_survey_key2():

    client = session["client"]
    if request.method == 'POST':
        img = request.files['file1']
        if img and allowed_file(img.filename):
            filename = secure_filename(img.filename)
            img.save(filename)
            blob_client = blob_service_client.get_blob_client(container = client, blob = "inclusivity_surveykey/"+filename)
            session["Survey_key"] =  img.filename
            with open(filename, "rb") as data:
                try:
                    blob_client.upload_blob(data, overwrite=True)
                    msg = "Upload Done ! "
                except:
                    msg ="Effort failed!"
            os.remove(filename)
    return msg


def generate_SAS():
    blob_list = []
    df_lst = []
    container = session["client"]
    # analysis = "analysis"
    # survey_key = "surveykey"
    container_client_analysis = blob_service_client.get_container_client(container=container)
    #container_client_survey_key = blob_service_client.get_container_client(container=container)
    
    blob_list = []
    for blob_i in container_client_analysis.list_blobs():
        blob_list.append(blob_i.name)
    
    TYPE = session["type_i"]
    blob_list = [k for k in blob_list if TYPE in k]
    
    sas_url_lst = []
    
    for j in range(len(blob_list)):
        sas_analysis = generate_blob_sas(account_name = account,
                        container_name = container,
                        blob_name = blob_list[j],
                        account_key=key,
                        permission=BlobSasPermissions(read=True),
                        expiry=datetime.utcnow() + timedelta(hours=1))
        sas_url = 'https://' + account+'.blob.core.windows.net/' + container + '/' + blob_list[j] + '?' + sas_analysis
        sas_url_lst.append(sas_url)
        if session["flag"] == "survey":
            df = pd.ExcelFile(sas_url)
        elif session["flag"] == "HR":
            mem_area =requests.get(sas_url)
            stage_file = mem_area.content
            df = fitz.open(stream=stage_file, filetype="pdf")
            #stage_file= urllib.request.urlopen(sas_url)
            #df = fitz.open(stage_file)
            #df =stage_file
        else:
            df = pd.read_csv(sas_url,encoding = "utf-8")
        
        df_lst.append(df)    
    return df_lst
    #return sas_url_lst
    #return blob_list

def write_files(lst_write):
    container = session["client"]
    if session["flag"]== "survey":
        blob_client = blob_service_client.get_blob_client(container=container, blob="analysed/survey/inclusivity_analysedfile/"+container+"_analysed_file.csv")
        blob_client.upload_blob(lst_write, blob_type="BlockBlob", overwrite=True)
    elif session["flag"] == "fi" and session["type_i"]== "focus":
        lst_write2 = ["focus","summary","wordcloud.jpg"]
        for i in range(len(lst_write)):
            if i<2:
                x = ".csv"
                blob_client = blob_service_client.get_blob_client(container=container, blob="analysed/focus/focus_analysedfile/"+container+"_focus_analysed_file"+lst_write2[i]+x)
                blob_client.upload_blob(lst_write[i], blob_type="BlockBlob", overwrite=True)
            else:
                blob_client = blob_service_client.get_blob_client(container=container, blob="analysed/focus/focus_analysedfile/"+container+"_focus_analysed_file"+lst_write2[i])
                blob_client.upload_blob(lst_write[i], blob_type="BlockBlob", overwrite=True)
                
            # if i<2:
            #     blob_client = blob_service_client.get_blob_client(container=container, blob="analysedfile/"+container+"_focus_analysed_file"+lst_write2[i])
            #     blob_client.upload_blob(lst_write[i], blob_type="BlockBlob", overwrite=True)
            # else:
            #     blob_client = blob_service_client.get_blob_client(container=container, blob="analysedfile/"+container+"_focus_analysed_file"+lst_write2[i])
            #     blob_client.upload_blob(lst_write[i], blob_type="BlockBlob", overwrite=True)
    elif session["flag"] == "fi" and session["type_i"]== "individual":
        lst_write2 = ["focus","summary","wordcloud.jpg"]
        for i in range(len(lst_write)):
            # blob_client = blob_service_client.get_blob_client(container=container, blob="individual_analysedfile/"+container+"_focus_analysed_file"+lst_write2[i])
            # blob_client.upload_blob(lst_write[i], blob_type="BlockBlob", overwrite=True)
            if i<2:
                x = ".csv"
                blob_client = blob_service_client.get_blob_client(container=container, blob="analysed/individual/individual_analysedfile/"+container+"_individual_analysed_file"+lst_write2[i]+x)
                blob_client.upload_blob(lst_write[i], blob_type="BlockBlob", overwrite=True)
            else:
                blob_client = blob_service_client.get_blob_client(container=container, blob="analysed/individual/individual_analysedfile/"+container+"_individual_analysed_file"+lst_write2[i])
                blob_client.upload_blob(lst_write[i], blob_type="BlockBlob", overwrite=True)
            # if i<2:
            #     blob_client = blob_service_client.get_blob_client(container=container, blob="analysedfile/"+container+"_focus_analysed_file"+lst_write2[i])
            #     blob_client.upload_blob(lst_write[i], blob_type="BlockBlob", overwrite=True)
            # else:
            #     blob_client =  blob_service_client.create_blob_from_bytes(container, blob = "analysedfile/"+container+"_focus_analysed_file"+lst_write2[i])
            #     blob_client.upload_blob(lst_write[i], blob_type="BlockBlob", overwrite=True)
    elif session["flag"] == "not_survey" and session["type_stage"]== "dashboard":
        #lst_write2 = ["focus","summary","wordcloud.jpg"]
        for i in range(len(lst_write)):
            lst_write2 = ["Complete Scored Table.csv","Main_Table_Leadership_Level.csv",
                          "Main_Table_Organisational_level.csv","most_inclusive_params.csv",
                          "least_inclusive_params.csv","most_and_least_represented.csv"]
            blob_client = blob_service_client.get_blob_client(container=container, blob="analysed/dashboard_inclusivity_analysedfile/"+lst_write2[i])
            blob_client.upload_blob(lst_write[i], blob_type="BlockBlob", overwrite=True)
            # if i<2:
            #     blob_client = blob_service_client.get_blob_client(container=container, blob="analysedfile/"+container+"_focus_analysed_file"+lst_write2[i])
            #     blob_client.upload_blob(lst_write[i], blob_type="BlockBlob", overwrite=True)
            # else:
            #     blob_client =  blob_service_client.create_blob_from_bytes(container, blob = "analysedfile/"+container+"_focus_analysed_file"+lst_write2[i])
            #     blob_client.upload_blob(lst_write[i], blob_type="BlockBlob", overwrite=True)
    elif session["flag"] == "HR" and session["type_i"]== "HR_Doc":
        lst_write2 = [lst_write[0],lst_write[0]+"_analysed",lst_write[0]+"_wordcloud.jpg"]
        keyword = lst_write[0]
        lst_write = lst_write[1:]
        for i in range(len(lst_write)):
            # blob_client = blob_service_client.get_blob_client(container=container, blob="analysed_HR"+"//"+keyword+"//"+lst_write2[i]+"_"+container++"_HR_Doc_analysed_file")
            # blob_client.upload_blob(lst_write[i], blob_type="BlockBlob", overwrite=True)
            if i<2:
                x = ".csv"
                blob_client = blob_service_client.get_blob_client(container=container, blob="analysed/HR_analysedfile/"+container+"_HR_analysed_file"+lst_write2[i]+x)
                blob_client.upload_blob(lst_write[i], blob_type="BlockBlob", overwrite=True)
            else:
                blob_client = blob_service_client.get_blob_client(container=container, blob="analysed/HR_analysedfile/"+container+"_HR_analysed_file"+lst_write2[i])
                blob_client.upload_blob(lst_write[i], blob_type="BlockBlob", overwrite=True)
            # if i<2:
            #     blob_client = blob_service_client.get_blob_client(container=container, blob="analysedfile/"+container+"_focus_analysed_file"+lst_write2[i])
            #     blob_client.upload_blob(lst_write[i], blob_type="BlockBlob", overwrite=True)
            # else:
            #     blob_client =  blob_service_client.create_blob_from_bytes(container, blob = "analysedfile/"+container+"_focus_analysed_file"+lst_write2[i])
            #     blob_client.upload_blob(lst_write[i], blob_type="BlockBlob", overwrite=True)

    return None
    
def download_blob_to_file():
    container = session["client"]
    blob_list_download = []
    container_client_analysis = blob_service_client.get_container_client(container=container)
    
    for blob_i in container_client_analysis.list_blobs():
        blob_list_download.append(blob_i.name)
    TYPE = session["type_i"]  

    blob_list_download = [k for k in blob_list_download if TYPE in k]

    if session["flag"]== "survey":
        OP_PARAM = "analysed"
        blob_list_download = [k for k in blob_list_download if OP_PARAM in k]
    elif session["flag"] == "not_survey" and session["type_i"] =="dashboard":
        OP_PARAM = "dashboard_inclusivity_analysedfile"
        blob_list_download = blob_list_download
    elif session["flag"] == "fi" and session["type_i"] =="focus":
        OP_PARAM = "focus_analysedfile"
        blob_list_download = [k for k in blob_list_download if OP_PARAM in k]
    elif session["flag"] == "fi" and session["type_i"] =="individual":
        OP_PARAM = "individual_analysedfile"
        blob_list_download = [k for k in blob_list_download if OP_PARAM in k]
    elif session["flag"] == "HR" and session["type_i"] =="HR_Doc":
        OP_PARAM = "HR_Doc"
        blob_list_download = [k for k in blob_list_download if TYPE in k]
        
    blob_list_download2 = []

    for i in blob_list_download:
        blob_list_download2.append(i.split("/")[-1])
        
    sas_url_lst = []
    for i in range(len(blob_list_download)):
        sas_analysis = generate_blob_sas(account_name = account,
                        container_name = container,
                        blob_name = blob_list_download[i],
                        account_key=key,
                        permission=BlobSasPermissions(read=True),
                        expiry=datetime.utcnow() + timedelta(hours=1))
        sas_url = 'https://' + account+'.blob.core.windows.net/' + container + '/' + blob_list_download[i] + '?' + sas_analysis
        sas_url_lst.append(sas_url)
    sas_url_lst2 = []
    for i in range(len(blob_list_download2)):
        sas_url_lst2.append((blob_list_download2[i],sas_url_lst[i]))
        
    return sas_url_lst2
"""
def download_blob_to_file(self, blob_service_client: BlobServiceClient, container_name):
    blob_client = blob_service_client.get_blob_client(container=container_name, blob="sample-blob.txt")
    with open(file=os.path.join(r'filepath', 'filename'), mode="wb") as sample_blob:
        download_stream = blob_client.download_blob()
        sample_blob.write(blob_client.download_blob().readall())
"""
    
@app.route('/', methods=['GET'])
def index():
    if session.get('logged_in'):
        return render_template('path.html')
    elif request.method == "GET":
        return redirect("/login")



@app.route('/login/', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    else:
        u = request.form['username']
        p = request.form['password']
        if u == MASTER_USER and p == MASTER_PASSWORD:
            session['logged_in'] = "Value"
            return redirect(url_for('index'))
        return render_template('index.html', message="Incorrect Login Credentials")
    
    
@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session['logged_in'] = False
    return redirect(url_for('index'))



#-------------------------------------------------------------------
@app.route('/path')
def path_options():
    option = request.args.get("options2")
    if option == "Perform Analysis":
        return redirect("/choice_pre")  
    elif option == "View Dashboard":
        return redirect('https://app.powerbi.com/')
#-------------------------------------------------------------------

#-------------------------------------------------------------------
@app.route('/dashboard')
def dashboard():
    return render_template("dashboard.html")

@app.route('/choice_pre')
def choice_pre():
    return render_template("choice.html") 

@app.route('/choice')
def options():
  client = request.args.get("client")
  #PATH = request.args.get("path")
  #session["PATH"] = PATH
    
  session["client"] = client

  option = request.args.get("options")
  if option == "Inclusivity Survey Assessment":
    session["flag"] = "survey"
    session["type_i"]= "inclusivity"
    container_flag = create_blob_container(blob_service_client,client)
    if container_flag == "Container Created":
        return redirect("/upload")  
  elif option == "Focus Group Interview Analysis":
    session["flag"] = "fi"
    session["type_i"]= "focus"
    container_flag = create_blob_container(blob_service_client,client)
    if container_flag == "Container Created":
        return redirect("/upload_fi")  
  elif option == "Individual Interview Analysis":
    session["flag"] = "fi"
    session["type_i"]= "individual"
    container_flag = create_blob_container(blob_service_client,client)
    if container_flag == "Container Created":
        return redirect("/upload_fi")
  elif option == "HR Document Analysis":
    session["flag"] = "HR"
    session["type_i"]= "HR_Doc"
    container_flag = create_blob_container(blob_service_client,client)
    if container_flag == "Container Created":
        return redirect("/upload_fi_HR") 
  else:
      return "Something is wrong"

#-------------------------------------------------------------------
@app.route('/upload')  
def upload():
    return render_template("upload.html",client = session["client"])  

@app.route('/upload_survey_key',methods = ['POST'])  
def upload_survey_key():
    message = upload_file()
    return render_template("upload_survey_key.html")  

@app.route('/upload_fi')  
def upload_fi():
    return render_template("upload_fi.html",client = session["client"]) 

@app.route('/upload_fi_HR')  
def upload_fi_HR():
    return render_template("upload_fi_HR.html") 
#-------------------------------------------------------------------
#-------------------------------------------------------------------


@app.route('/success',methods = ['POST'])  
def success():
    message2 = upload_survey_key2()
    analysis_file_name = session["Analysis_file"]
    survey_key_file_name = session["Survey_key"] 
    client_name = session["client"]
    return render_template("success.html", name = analysis_file_name,survey_key = survey_key_file_name,clientname = client_name)

@app.route('/success_fi',methods = ['POST'])  
def success_fi():
    message = upload_file()
    analysis_file_name = session["Analysis_file"]
    return render_template("success_fi.html", name = analysis_file_name) 

@app.route('/success_fi_HR',methods = ['POST'])  
def success_fi_HR():
    message = upload_file()
    analysis_file_name = session["Analysis_file"]
    return render_template("success_fi_HR.html", name = analysis_file_name) 

#-------------------------------------------------------------------

@app.route('/analysis',methods = ['POST'])  
def analysis():
    analysis_file_name = session["Analysis_file"]
    survey_key_file_name = session["Survey_key"] 
    client_name = session["client"]
    return render_template("analysis.html", analysis_file_name = analysis_file_name,survey_key_file_name = survey_key_file_name,client_name = client_name) 

@app.route('/analysis_fi',methods = ['POST'])  
def analysis_fi():
    client_name = session["client"]
    analysis_file_name = session["Analysis_file"]
    return render_template("analysis_fi.html",client_name=client_name,analysis_file_name=analysis_file_name) 

@app.route('/analysis_HR',methods = ['POST'])  
def analysis_HR():
    client_name = session["client"]
    analysis_file_name = session["Analysis_file"]
    return render_template("analysis_HR.html",client_name=client_name,analysis_file_name=analysis_file_name) 

#-------------------------------------------------------------------


@app.route('/analysis_process',methods = ['GET','POST'])  
def analysis_p():
    x = generate_SAS()
    z = survey_key(x[1],x[0])
    op = z.to_csv(encoding= "utf-8")
    write_files(op)
    return render_template("analysis_process.html")
    # return x
    


    
@app.route('/analysis_process_fi',methods = ['POST'])  
def analysis_p_fi():
    x = generate_SAS()

    FILE = x[0]
    ORG_NAME = session["client"]
    TYPE = session["type_i"]
    #x = rating_gen(FILE,ORG_NAME,TYPE)
    df,summary,PHRASES = rating_gen(FILE,ORG_NAME,TYPE)
    op1 = df.to_csv(encoding= "utf-8")
    summary = pd.DataFrame(summary, index=[0])
    op2 = summary.to_csv(encoding= "utf-8")
    op3 = wordcloud(PHRASES,ORG_NAME,TYPE)
    lst_write = [op1,op2,op3]
    write_files(lst_write)
    y = "Analysis is complete. You can download the file"

    return render_template("analysis_process_fi.html",data = y)
    #return x

@app.route('/analysis_fi_HR1',methods = ['POST'])  
def analysis_fi_HR1():
    x = generate_SAS()
    #x = [x]

    FILE = x[0]
    ORG_NAME = session["client"]
    TYPE = session["type_i"]
    KEYWORD_LIST = ["diversity","equity"]
    OP_LIST = master_func(FILE,KEYWORD_LIST)
    for j in range(len(OP_LIST)):
        op1 = OP_LIST[j][0]
        op2 = OP_LIST[j][1]
        op3 = OP_LIST[j][2]
        lst_write = [op1,op2,op3]
        #write_files(lst_write)
        
    # op1 = df.to_csv(encoding= "utf-8")
    # summary = pd.DataFrame(summary, index=[0])
    # op2 = summary.to_csv(encoding= "utf-8")

    # op3 = wordcloud(PHRASES,ORG_NAME,TYPE)
    # lst_write = [op1,op2,op3]
    # write_files(lst_write)
    # y = "Analysis is complete. You can download the file"

    # return render_template("analysis_process_fi.html",data = y)
    # return x
    return "test"

@app.route('/dashboard_creation',methods = ['POST'])  
def dashboard_creation():
    blob_list = []
    container = session["client"]
    session["flag"]= "not_survey"
    session["type_stage"]= "dashboard"
    container_client_analysis = blob_service_client.get_container_client(container=container)
    
    blob_list = []
    for blob_i in container_client_analysis.list_blobs():
        blob_list.append(blob_i.name)
    
    TYPE = "analysed_file"
    blob_list = [k for k in blob_list if TYPE in k]
    
    sas_url_lst = []
    
    sas_analysis = generate_blob_sas(account_name = account,
                    container_name = container,
                    blob_name = blob_list[0],
                    account_key=key,
                    permission=BlobSasPermissions(read=True),
                    expiry=datetime.utcnow() + timedelta(hours=1))
    sas_url = 'https://' + account+'.blob.core.windows.net/' + container + '/' + blob_list[0] + '?' + sas_analysis
    sas_url_lst.append(sas_url)
    df = pd.read_csv(sas_url,encoding = "utf-8")
    
    #-----------------------------

    op_master = dashboard_creation2(df)
    lst_write = op_master
    write_files(lst_write)

    y = "Analysis is complete. You can download the file"

    return render_template("analysis_process_dashboard.html",data = y)
    #return render_template("analysis_process_fi.html",data = y)
    #return "test"

#-------------------------------------------------------------------
#-------------------------------------------------------------------
@app.route('/download',methods = ['POST'])
def downloadFile ():
    #For windows you need to use drive name [ex: F:/Example.pdf]
    x  = download_blob_to_file()
    if len(x) ==1:
        x = x
    else:
        x = x
    return render_template("download.html",links2= x)

@app.route('/download_fi',methods = ['POST'])
def downloadFile_fi ():
    #For windows you need to use drive name [ex: F:/Example.pdf]
    x  = download_blob_to_file()
    if len(x) ==1:
        x = x
    else:
        x = x
    return render_template("download.html",links2= x)

@app.route('/download_dashboard',methods = ['POST'])
def downloadFile_dashboard ():
    #For windows you need to use drive name [ex: F:/Example.pdf]
    x  = download_blob_to_file()
    if len(x) ==1:
        x = x
    else:
        x = x
    return render_template("download.html",links2= x)
    #return x

#-------------------------------------------------------------------
if __name__ == "__main__":
    app.run()

           


           

