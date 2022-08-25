import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings

df = pd.read_csv("feature_engineering/Telco Customer Churn Feature Engineering/Telco-Customer-Churn.csv")

warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)




####################################################################
# EDA (Explanatory Data Analysis)
#####################################################################


#1- General View

df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe().T
df.isnull().values.any()
df.isnull().sum()

    def check_df(dataframe, head=5):
        print("########### SHAPE ###########")
        print(dataframe.shape)
        print("########### TYPES ###########")
        print(dataframe.dtypes)
        print("########### HEAD ###########")
        print(dataframe.head(head))
        print("########### TAIL ###########")
        print(dataframe.tail(head))
        print("########### NA ###########")
        print(dataframe.isnull().sum())
        print("########### DESCRIBE ###########")
        print(dataframe.describe([0, 0.25, 0.5, 0.75, 0.95, 1]).T)

    check_df(df)

# 2- Grab numeric and categorical variables
    def grab_col_names(dataframe, cat_th=10, car_th=20):
        """

        Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
        Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

        Parameters
        ------
            dataframe: dataframe
                    Değişken isimleri alınmak istenilen dataframe
            cat_th: int, optional
                    numerik fakat kategorik olan değişkenler için sınıf eşik değeri
            car_th: int, optional
                    kategorik fakat kardinal değişkenler için sınıf eşik değeri

        Returns
        ------
            cat_cols: list
                    Kategorik değişken listesi
            num_cols: list
                    Numerik değişken listesi
            cat_but_car: list
                    Kategorik görünümlü kardinal değişken listesi

        Examples
        ------
            import seaborn as sns
            df = sns.load_dataset("iris")
            print(grab_col_names(df))


        Notes
        ------
            cat_cols + num_cols + cat_but_car = toplam değişken sayısı
            num_but_cat cat_cols'un içerisinde.

        """
        # cat_cols, cat_but_car
        cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
        num_but_cat = [col for col in dataframe.columns if
                       dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
        cat_but_car = [col for col in dataframe.columns if
                       dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
        cat_cols = cat_cols + num_but_cat
        cat_cols = [col for col in cat_cols if col not in cat_but_car]

        # num_cols
        num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
        num_cols = [col for col in num_cols if col not in num_but_cat]

        print(f"Observations: {dataframe.shape[0]}")
        print(f"Variables: {dataframe.shape[1]}")
        print(f'cat_cols: {len(cat_cols)}')
        print(f'num_cols: {len(num_cols)}')
        print(f'cat_but_car: {len(cat_but_car)}')
        print(f'num_but_cat: {len(num_but_cat)}')

        return cat_cols, num_cols, cat_but_car


    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    cat_cols
    num_cols
    cat_but_car

# 3 & 4- Analyze the numeric and categorical variables

    ##################################
    # Functions for analyzes
    ##################################

    def cat_summary(dataframe, col_name, plot=False):
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("##########################################")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)

    def num_summary(dataframe, numerical_col, plot=False):
        quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
        print(dataframe[numerical_col].describe(quantiles).T)

        if plot:
            dataframe[numerical_col].hist(bins=20)
            plt.xlabel(numerical_col)
            plt.title(numerical_col)
            plt.show()

    def target_summary_with_cat(dataframe, target, categorical_col):
        print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

    def target_summary_with_num(dataframe, target, numerical_col):
            print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

    ##################################
    # Analysis of categorical variables
    ##################################

    cat_summary(df, "Outcome")