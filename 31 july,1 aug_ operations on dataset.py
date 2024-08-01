# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 08:36:59 2024

@author: HP
"""
##31 july 2024
import pandas as pd
#let us import dataset
df=pd.read_csv("D:/DS/5-Data_prep/ethnic diversity.csv")
#check data types of columns
df.dtypes
#salaries data type is float, convert into int
#df1=df.Salaries.astype(int)
df.Salaries=df.Salaries.astype(int)
df.dtypes
#now the data type of salaries is not 
#similarly age data type must be float
#presently it is int
df.age=df.age.astype(float)
df.dtypes

###############################################
#identify the duplicates
df_new=pd.read_csv("D:/DS/5-Data_prep/education.csv")
duplicate=df_new.duplicated()
#output of this function is single column
#if there is duplicate records  output-true
#if there is no duplicate records output-false
#series will be created
duplicate
sum(duplicate)
#output will be 0
#now let us import another dataset
df_new1=pd.read_csv("D:/DS/5-Data_prep/mtcars_dup.csv")
duplicate1=df_new1.duplicated()
duplicate1
sum(duplicate1)
#there are 3 duplicate records 
#row 17 is duplicate of row 2 like wise you can 3 duplicates
#records
#there is a function called drop_duplicates()
#which will drop all the duplicate rcords
df_new2=df_new1.drop_duplicates()
duplicate2=df_new2.duplicated()
duplicate2
sum(duplicate2)

#################################################
#outliers treatment
import pandas as pd
import seaborn as sns
df=pd.read_csv("D:/DS/5-Data_prep/ethnic diversity.csv")
#now let us find outliers in Salaries
sns.boxplot(df.Salaries)
#there are outliers
#let us check outliers in age column
sns.boxplot(df.age)
#there are no outliers
#let us calculate IQR
IQR=df.Salaries.quantile(0.75)-df.Salaries.quantile(0.25)
#have observed IQR in variable explorer
#no, because IQR is in capital letters
#treated as constants
IQR

#------------------------------------------------------
#1 August 2024#


#but if we will try as I, Iqr or iqr it is showing
#I=df.Salaries.quantile(0.75)-df.Salaries.quantile(0.25)
lower_limit=df.Salaries.quantile(0.25)-1.5*IQR
upper_limit=df.Salaries.quantile(0.75)+1.5*IQR
lower_limit
upper_limit
#now it you will check the lower limit of 
#salary it is : -19446, 93992

#there is negative salary 
#so make it as 0
# how to make it 
# go to the variable explorer and make it as 0

###################################################
#Trimming
import numpy as np
outliers_df=np.where(df.Salaries>upper_limit,True, np.where(df.Salaries<lower_limit,True,False))
#you can check outliers_df column in variable explorer

df_trimmed=df.loc[~outliers_df]
df.shape
#(310,13)
df_trimmed.shape
#(306,13)
sns.boxplot(df_trimmed.Salaries)
#########################################################3
#Replcament Techniques
#drawback of trimming technique is we are losing the data
df=pd.read_csv("D:/DS/5-Data_prep/ethnic diversity.csv")
df.describe()
#record no.23 has got outliers
#map all the outlier values to upper limit
df_replaced=pd.DataFrame(np.where(df.Salaries>upper_limit,
            upper_limit,np.where(df.Salaries<lower_limit,lower_limit,df.Salaries)))
#if the values are greater than upper_limit
#map it to upper limit , and less than lower_limit
#map it to lower limit , if it within the range
#then keep as it is
sns.boxplot(df_replaced[0])

###################################################
#Winsorizer
from feature_engine.outliers import Winsorizer

# Example usage
winsor = Winsorizer(capping_method='iqr', tail='both', fold=1.5)

#copy winsorizer and paste in help tab of 
#top right window , study the method
#
df_t=winsor.fit_transform(df[['Salaries']])
sns.boxplot(df['Salaries'])
sns.boxplot(df_t['Salaries'])
