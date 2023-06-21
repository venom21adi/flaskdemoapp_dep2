import os
#import shutil
import pandas as pd
import numpy as np
def dashboard_creation(dataframe_file):

    def main_table(dataframe_file):
        #os.chdir(SOURCE_PATH)
        
        
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
    
    
    # ORG_NAME = "test"
    # SOURCE_PATH =r"C:\EDI\Data\Survey_Monkey_Data\Dashboard_strategy\Input"
    # DESTINATION_PATH =r"C:\EDI\Data\Survey_Monkey_Data\Dashboard_strategy\Input"
    # OUTPUT_PATH = r"C:\EDI\Data\Survey_Monkey_Data\Dashboard_strategy\Output"
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
    
    lst_master =[op4,op5,op6,op7,op8,op9]
    return lst_master
