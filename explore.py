import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px

def customize_edu_level(x):
    if 'Bachelors' in x:
        return 'Bachelor’s degree'
    if 'Masters' in x:
        return 'Master’s degree'
    if 'Prof-school' in x or 'Doctorate' in x:
        return 'Ph.d or Doctorate degree'
    return 'Less than a Bachelors'

def customize_age(x):
    if x < 20:
        return 'Below 20'
    if x <= 30 and x >= 20:
        return '20 - 30'
    if x <= 40 and x > 30:
        return '31 - 40'
    if x <= 50 and x > 40:
        return '41 - 50'
    if x <= 60 and x > 50:
        return '51- 60'
    if x > 60:
        return 'Above 60'

@st.cache
def load_model():
    df = pd.read_csv("income.csv")
    df['income']=df['income'].map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1})
    df['native.country'] = df['native.country'].replace('?',np.nan)
    df['workclass'] = df['workclass'].replace('?',np.nan)
    df['occupation'] = df['occupation'].replace('?',np.nan)

    df['native.country'] = ['Others' if x != ('Mexico') and x != ('United-States') and x != ('Canada') and x != ('India') and x != ('Philippines')
                                 and x != ('Germany') and x != ('Puerto-Rico') and x != ('El-Salvador') else x for x in df['native.country']]
    
    df = df.dropna()

    df['age'] = df['age'].apply(customize_age)
    df['education'] = df['education'].apply(customize_edu_level)
    df_new = df.copy()
    df_new.dropna(how='any',inplace=True)
    df_new.drop(['fnlwgt', 'capital.gain','capital.loss', 'education.num', 'sex', 'race', 'marital.status', 'relationship', 'hours.per.week'], axis=1, inplace=True)
    return df_new

df = load_model()

def explore_income_variations():
    st.title("Interactive visualization for various features")

    data = df["occupation"].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(data, labels=data.index, autopct="%1.1f%%", shadow=True, startangle=90)
    ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.write("""#### Frequency distribution of different occupation""")
    st.markdown("Data predict that majority of workforce are employed as Professional specialist, Craft Occupation, executive Manager, Admin clerical and Sales")
    st.pyplot(fig1)

    fig2,ax=plt.subplots(1,figsize=(18,8))
    ax = df['income'].value_counts().plot.pie(explode=[0,0],autopct='%1.1f%%',ax=ax,shadow=True)
    # ax.set_title('Visualize Income based on salary')
    st.write("""#### Visualize Income based on salary""")
    st.markdown("According to the adult census dataset 75% of the people fall under less than 50k income slab which includes the whole world.")
    st.pyplot(fig2)



    fig3, ax = plt.subplots(figsize=(12, 8))
    ax = sns.countplot(x="workclass", hue="income", data=df, palette="Set1")
    # ax.set_title("Frequency distribution of workclass wrt income")
    ax.legend(loc='upper right')
    # plt.show()
    st.write("""#### Frequency distribution of workclass wrt income""")
    st.markdown("Most of the workforce are employed in a private sector ")
    st.pyplot(fig3);
