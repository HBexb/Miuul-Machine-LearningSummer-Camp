# Categorical Summary

def cat_summary(dataframe, col_name, tail=5, quantile=False, plot=False):
    """
    This function takes the parameters dataframe, col_name(column name), tail, quantile and plot,
    and uses them to return that a summary of DataFrame with taken column name parameters.
    When you run this code you are going to get the result that variable count and ratio to all that variable.

    Ratio is:  100 * dataframe[col_name].value_counts()/len(dataframe)
    Plot is :  sns.countplot(x=dataframe[col_name], data=dataframe)
    Parameters
    ----------
    dataframe
    col_name
        Column Name
    tail
    quantile
    plot

    Returns
    -------
    Col_name and Ratio

    #Example by using Titanic data
    cat_summary(df,"sex")

                sex       Ratio
    Female      577      64.758698
    Male        314      35.241302

    """
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("############################################")
    print(dataframe.tail(tail))
    if quantile:
        print(dataframe.quantile([0, 0.05, .5, .95, .99, 1]).T)
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

### Pandas Trainings


# 1- Import Titanic data set from seaborn
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = sns.load_dataset("titanic")

# 2- Find number of male and female passengers
df["sex"].value_counts()

# 3- Find unique values belongs to each column

unique_values = {col : len(df[col].unique()) for col in df.columns}

# 4- Find number of uniq values of pclass variable
df[["pclass","parch"]].nunique()
pclass_uniq = {"pclass": df["pclass"].unique().size }

    #So what are they?

pclass_uniq_vals = df["pclass"].unique()

# 5- Find number of unique values of pclass and parch variables

pclass_uniq = {"pclass": df["pclass"].unique().size }
parch_uniq =  {"parch": df["parch"].unique().size }

parch_uniq_vals = df["parch"].unique()

# 6- Control the type of the embarked variable, and change its type as category

df["embarked"].dtypes

df["embarked"] = df["embarked"].astype("category")

# 7- Show all information of passengers whose embarked variable is value of C

df[df["embarked"]=="C"]

# 8- Show all information of passengers whose embarked variable is not value of C

df[df["embarked"]!="S"]

# 9- Show all information of passengers whose age <30 and sex is woman

df[(df["age"]<30) & (df["sex"] != "male")]

# 10- Show all information of passengers whose "fare" variable > 500 or age>70

df[(df["age"]>70) | (df["fare"]>500)]

# 11- Find the sum of the empty values in each variable

df.isnull().sum()

# 12- Drop "who" variable from dataframe

df.drop("who", axis=1, inplace=True)

# 13- Fill in the empty values in the deck variable with the most repeated value (mode) of the deck variable

df["deck"].fillna(df["deck"].mode()[0], inplace = True)

df["deck"].isnull().any()

# 14- Fill in the empty values in the age variable with the median of age variable

df["age"].fillna(df["age"].median(), inplace = True)
df["age"].isnull().any()

# 15 - Find the sum, count, mean values of the pclass and gender variables of the survived variable.

sum = df.groupby(["pclass","sex"])["survived"].agg({"survived":["sum","count","mean"]})


# 16- Write a function that will return 1 for those under 30, 0 for those equal to or above 30. titanic data using the function you wrote
  #Create a variable named age_flag in the set. (use apply and lambda constructs)

df["age_flag"] = df["age"].apply(lambda x: 1 if x<30 else 0)

# 17- Import the "Tips" dataset from seaborn library

dfTips = sns.load_dataset("Tips")

# 18- Find the sum, min, max and mean values of the total_bill value according to the categories (Dinner, Lunch) of the time variable.

dfTips.groupby(["time"]).agg({"total_bill" : ["sum","min","max","mean"]})


# 19- Find the sum, min, max and mean of values of total_bill, according to Day and Time

dfTips.groupby(["day","time"]).agg({"total_bill":["sum","max","min","mean"]})


# 20 - Find the sum, min, max and mean values of the "total_bill" and "tips" values of the lunch time and female customers according to the day.
#time = lunch sex=female total_bill and tip

dfTips[["total_bill","tip","day"]].loc[(dfTips["time"]=="Lunch")&(dfTips["sex"]=="Female")].groupby("day").agg({"total_bill":["sum","min","max","mean"],"tip":["sum","min","max","mean"]})

#fancy index
dfTips[]
# 21- What is the average of orders with size less than 3 and total_bill greater than 10? (use loc)

dfp = pd.DataFrame(dfTips)
dfp2 = dfp.loc[(dfp["size"] < 3) & (dfp["total_bill"] > 10)].mean()

# 22- Create a new variable called total_bill_tip_sum. Let him give the sum of the total bill and tip paid by each customer.

dfTips["total_bill_tip_sum"] = dfTips["total_bill"] + dfTips["tip"]

# 23- Find the mean of the total_bill variable separately for men and women. 0, above and equal to those below the averages you found
    # Create a new total_bill_flag variable where 1 is given to those.
    # The averages of Female for women and Male for men will be taken into account. Gender and total_bill as parameters
    # Start by writing a function that takes (will include if-else conditions)


#24- Using the total_bill_flag variable, observe the number of below and above average by gender.



#25- Sort the data from largest to smallest according to the total_bill_tip_sum variable and assign the first 30 people to a new dataframe.

