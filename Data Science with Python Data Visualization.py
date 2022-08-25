# categorical variable: column chart, countlot bar
# numeric variable: histogram, boxplot


#Categorical Variable Visualization

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.width',500)
pd.set_option('display.max_columns',None)

df = sns.load_dataset("titanic")
df.head()

df["sex"].value_counts().plot(kind='bar')
plt.show()

# Numeric Variable Visualization

plt.hist(df["age"])
plt.show()

plt.boxplot(df["fare"])
plt.show()

# Matplotlib Features
############
#plot
#############

import matplotlib.pyplot as plt
import numpy as np

x = np.array([1,8])
y = np.array([0,150])

plt.plot(x,y)
plt.show()

plt.plot(x,y, 'o')
plt.show

#########
x = np.array([2,4,6,8,10])
y = np.arange(1,10,2)

plt.plot(x,y)
plt.show()

plt.plot(x,y, 'o')
plt.show


#############
# marker
#############
y = np.array([13,28,11,100])

plt.plot(y, marker ='o')
plt.show()

#############
#line
#############

y = np.array([13,28,11,100])
plt.plot(y, linestlye = 'dashed', color = "r") #dotted, dashdot, etc.
plt.show()

#############
# Multiple Lines
#############

y = np.array([13,28,11,100])
x = np.array([23,18,31,10])

plt.plot(x)
plt.plot(y)
plt.show()

#############
# Labels
#############

x = np.arange(20,150,20)
y = np.arange(40,300,40)

plt.plot(x,y)
#############
# Title
#############

plt.title("Main Title")

#############
# Give name to X axis
#############

plt.xlabel("X axis")

#############
# Give name to Y axis
#############

plt.ylabel("Y axis")

plt.grid()
plt.show()

#############
# Subplots

# Plot 1
x = np.arange(20,150,20)
y = np.arange(40,300,40)

plt.subplot(1,3,1)
plt.title("1")
plt.plot(x,y)

# Plot 2
x = np.arange(50,100,20)
y = np.arange(100,200,40)

plt.subplot(1,3,2)
plt.title("2")
plt.plot(x,y)

# Plot 3

x = np.arange(10,100,20)
y = np.arange(20,200,40)
plt.subplot(1,3,3)
plt.title("3")
plt.plot(x,y)


#############
# SEABORN
#############

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

df = sns.load_dataset("tips")
df.head()

df["sex"].value_counts()

# categorical variable visualization
sns.countplot(x = df["sex"], data = df)
plt.show()

#matplotlib
#df["sex"].value_counts().plot(kind = "bar")

# numeric variable visualization

sns.boxplot(x=df["total_bill"])
plt.show()

df["total_bill"].hist()
plt.show()