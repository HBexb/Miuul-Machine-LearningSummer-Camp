# Pandas

# Pandas Series

import pandas as pd

s = pd.Series([18, 77, 12, 4, 5])

type(s)

s.index
s.dtype
s.size
s.ndim
s.values
s.head(3)
s.tail(3)

# Reading Data

df = pd.read_csv("pythonProgramlama/Data Science with Python Pandas/datasets/advertising.csv")
df.head()

# Quick View of Data

import seaborn as sns

df = sns.load_dataset("titanic")
df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe().T
df.isnull().values.any()
df.isnull()
df.isnull().values

df.isnull().sum()

df["sex"].head()
df["sex"].value_counts()

# Selection Transactions

import seaborn as sns

df = sns.load_dataset("titanic")
df.head()

df.index

df[0:13]
df.drop(0, axis=0).head()

delete_indexes = [1, 3, 5, 7]

df.drop(delete_indexes, axis=0).head(10)

# df = df.drop(delete_indexes, axis=0)
# df.drop(delete_indexes, axis= 0, inplace = True)

# Convert variable to Index

df["age"]

df.age.head()

df.index = df["age"]
df.index
df.drop("age", axis=1, inplace=True)
df.head()

df["age"] = df.index

df.head()
df.drop("age", axis=1, inplace=True)

df = df.reset_index().head()
df.head()

# Operations on variables

import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")

"age" in df

a = df["age"].head()
b = df.age.head()
c = df[["age"]].head()
d = df[["age", "alive"]]
type(a)
type(b)
type(c)

col_names = ["age", "adult_male", "alive"]
df[col_names]

df["age**2"] = df["age"] ** 2
df.head()

df["age3"] = df["age"] / df["age**2"]

df.drop("age3", axis=1).head()
df.drop(col_names, axis=1).head()

df.loc[:, df.columns.str.contains("age")].head()
# if you want to delete those chosen variables
df.loc[:, ~df.columns.str.contains("age")].head()

# Loc & Iloc

# iloc: integer based selection

df.iloc[0:3]  # Chose up to 3
df.iloc[0:0]

# loc: label based selection

df.loc[0:3]  # Choose from 0 to 3, included 3. Loc chooses as index


df.iloc[0:3, 0:3]
df.loc[0:3, "age"]

col_names = ["age", "embarked", "alive"]

df.loc[0:3, col_names]

# Conditional Selection

import pandas as pd
import seaborn as sns

pd.set_option("display.max_columns", None)
df =  sns.load_dataset("titanic")
df.head()

df[df["age"] > 50].head()
df[df["age"] > 50].count()
df[df["age"] > 50]["age"].count()

#Age > 50 and Males
df.loc[(df["age"] > 50) & (df["sex"] == "male"), ["age","class"]].head()

df.loc[(df["age"] > 50)
        & (df["sex"] == "male")
        & (df["embark_town"] == "Cherbourg")
        , ["age","class","embark_town"]].head()

df["embark_town"].value_counts()

df_new = df.loc[(df["age"] > 50) & (df["sex"] == "male")
                & ((df["embark_town"] == "Cherbourg") | (df["embark_town"] == "Southampton")),
                ["age","class","embark_town"]]

df_new["embark_town"].value_counts()


##########
#Agregation & Grouping

# count()
# first()
# last()
# mean()
# median()
# min()
# max()
# std()
# var()
# sum()


import pandas as pd
import seaborn as sns
pd.set_option("display.max_columns", None)

df = sns.load_dataset("titanic")
df.head()

df["age"].mean()
df.groupby("sex")["age"].mean()
df.groupby("sex").agg({"age": "mean"})
df.groupby("sex").agg({"age": ["mean","sum"]})

df.groupby("sex").agg({"age" : ["mean","sum"],
                       "survived":"mean"})

df.groupby(["sex","embark_town"]).agg({"age" : ["mean"],
                       "survived":"mean"})

df.groupby(["sex","embark_town","class"]).agg({"age" : ["mean"],
                       "survived":"mean"})

df.groupby(["sex","embark_town","class"]).agg({
    "age" : ["mean"],
    "survived":"mean",
    "sex":"count"})


#Pivot Table:

import pandas as pd
import seaborn as sns
pd.set_option("display.max_columns", None)

df = sns.load_dataset("titanic")
df.head()

df.pivot_table("survived","sex","embarked")

df.pivot_table("survived","sex","embarked", aggfunc= "std")

df.pivot_table("survived","sex",["embarked","class"])

df.head()

df["new_age"] = pd.cut(df["age"], [0,10,18,25,40,90])

df.pivot_table("survived","sex",["new_age","class"])

pd.set_option('display.width',500)

df.pivot_table("survived","sex",["new_age","class"])


## Apply & Lambda

import pandas as pd
import seaborn as sns
pd.set_option("display.max_columns", None)
pd.set_option('display.width',500)
df = sns.load_dataset("titanic")
df.head()

df["age2"] = df["age"]*2
df["age3"] = df["age"]*5

(df["age"]/10).head()
(df["age2"]/10).head()
(df["age3"]/10).head()

for col in df.columns:
    if "age" in col:
        print(col)

for col in df.columns:
    if "age" in col:
        print((df[col]/10).head())

for col in df.columns:
    if "age" in col:
        df[col] = df[col] /10

df.head()

df[["age","age2","age3"]].apply(lambda x: x/10).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x: x/10).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x: (x-x.mean())/x.std()).head()

def standart_scaler(col_name):
    return (col_name - col_name.mean()) / col_name.std()

df.loc[:, df.columns.str.contains("age")].apply(standart_scaler).head()


df.loc[:,["age","age2","age3"]] = df.loc[:, df.columns.str.contains("age")].apply(standart_scaler).head()

df.loc[:,df.columns.str.contains("age")] = df.loc[:, df.columns.str.contains("age")].apply(standart_scaler)

df.head()

# Join

import numpy as np

m = np.random.randint(1,30, size = (5,3))
df1 = pd.DataFrame(m, columns = ["var1",
                                 "var2",
                                 "var3"])
df2 = df1+99

# CONCAT #axis = 0 alt alta #### axis = 1 yan yana
pd.concat([df1,df2])
pd.concat([df1,df2], ignore_index = True,axis = 1)
pd.concat([df1,df2], ignore_index = True)

df1= pd. DataFrame({'employees': ["john",'dennis', 'mark', 'maria'],
                    'group' : ['accounting','engineering',' engineering','hr']})

df2 = pd.DataFrame({'employees': ['mark', "john", "dennis", 'maria'],
                    'start_date': [2010,2009,2014,2019]})

df3 = pd.merge(df1, df2)

df4 = pd.DataFrame({'group': ['accounting', 'engineering', 'hr'],
                    'manager': ['Caner', 'Mustafa', 'Berkcan']})

# We want to reach information of each employees' manager

pd.merge(df3, df4)
