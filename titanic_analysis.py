import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = sns.load_dataset('titanic')
print(df)

df.isnull().sum()
df["deck"] = df["deck"].astype(object)
df["age"].fillna(df["age"].median(),inplace=True)
df["embarked"].fillna(df["embarked"].mode()[0],inplace=True)
df["deck"].fillna("Unknown",inplace=True)
df.dropna(subset = ['embark_town'],inplace=True)

print(df.head())

print(sns.countplot(x="survived",hue="sex",palette="coolwarm",data=df))
plt.show()
print(sns.countplot(x='pclass',hue="sex",palette="rainbow",data=df))
plt.show()
print(sns.histplot(x='age',bins = 20,kde = True,data=df))
plt.show()
print(sns.heatmap(df.corr(numeric_only=True),annot=True))
plt.show()

survival_gender = df.groupby("sex")["survived"].value_counts(normalize=True).unstack()*100
print(survival_gender)
survival_pclass = df.groupby("pclass")["survived"].value_counts(normalize=True).unstack()*100
print(survival_pclass)
survival_gender_class = df.groupby(["sex","pclass"])["survived"].value_counts(normalize=True).unstack()*100
print(survival_gender_class)

survival_gender_class.plot(kind="bar",stacked=True,colormap="coolwarm")
plt.show()

df["family_size"] = df["sibsp"]+df["parch"]+1
family_survival =df.groupby("family_size")["survived"].mean()*100
print(family_survival)

family_survival.plot(kind="bar",color="skyblue")
plt.show()

sns.kdeplot(df.loc[df["survived"]==1,"age"],label="survived",fill=True)
plt.show()

sns.kdeplot(df.loc[df["survived"]==0,"age"],label="Unsurvived",fill=True)
plt.show()
                            

