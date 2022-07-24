#: 1 Replace comma and dot with space, and capitalize the given string

text = "The goal is to turn data into information, and information into insights."

a = text.replace(",", "")
b = a.replace(".", "")

print(b.upper().split())

#: 2

lst = list("DATASCIENCE")
print(lst)

print(len(lst))

print(lst[0], lst[10])

print(lst[:4])

lst.pop(8)

lst.append("AI")

lst.insert(8, "N")
print(lst)

#: 3

dict = {'Christian': ["America", 18],
        'Daisy': ["England", 12],
        'Antonio': ["Spain", 22],
        'Dante': ["Italy", 25]}

dict.keys()
dict.values()

dict['Daisy'][1] = 13
dict

dict['Ahmet'] = ["Turkey", 24]
dict

del dict['Antonio']

# or

dict.pop('Antonio')

#: 4

l = [2, 13, 18, 93, 22]

def even_odd(liste):
    even_list = []
    odd_list = []
    for i in liste:
        if i % 2 == 0:
            even_list.append(i)
        else:
            odd_list.append((i))
    return even_list, odd_list

even_odd(l)


#: 5 List Comprehension

import seaborn as sns

df = sns.load_dataset("car_crashes")
df.columns

abc = ["NUM_" + col.upper() if df[col].dtype != "O" else col.upper() for col in df.columns ]

#: 6 List Comprehension 2

abc2 = [col.upper() + "_FLAG" if "no" not in col else col.upper() for col in df.columns]


#: 7 List Comprehension 3
og_list = ["abbrev", "no_previous"]

df = sns.load_dataset("car_crashes")
df.columns

clmns = [col for col in df.columns if col != "abbrev" and col != "no_previous"]

new_df = df[clmns]

new_df.head()