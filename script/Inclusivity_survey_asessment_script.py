"""
Cleaning script
"""

#------------------Importing the libraries----------------
import os

import pandas as pd
import numpy as np
import re
pd.options.display.max_colwidth = 120
pd.set_option('display.max_colwidth', None)

#------------------Importing survey key-------------------
def survey_key(Survey_key_file,Analysis_file):
    # os.chdir(SURVEY_KEY_PATH)
    # filename2 = os.listdir()
    
    #data_sur = pd.ExcelFile(filename2[0])
    #data_sur = pd.ExcelFile(Survey_key_file)
    data_sur = Survey_key_file
    sheet_names2 = data_sur.sheet_names
    lst = []
    for i in sheet_names2:
        file = pd.read_excel(Survey_key_file,sheet_name=str(i))
        #file = pd.read_excel(filename2[0],sheet_name=str(i))
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
        
    #os.chdir("..")
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
    data_raw = Analysis_file
    sheet_names1 = data_raw.sheet_names
    
    file1 = pd.read_excel(Analysis_file,sheet_name=str(sheet_names1[0]))
    
    df1 = file1
    
    #os.chdir("..")
    
    
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
        #os.chdir("..")
    #os.chdir("Upload")

    
    # os.chdir(DATA_FILE_PATH)
    # os.chdir("..")
    # folder = "./analysed_file"
    # if os.path.exists(folder):
    #     shutil.rmtree(folder)
    # if not os.path.exists(folder):
    #     os.makedirs(folder)
    # os.chdir(folder)
    # df9.to_csv(x+"_analysed_"+".csv")
    # h = os.getcwd()
    g = "Analysis is completed"
    return df9
