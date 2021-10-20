import os, random

import numpy as np
from scipy.stats import chi2_contingency
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import streamlit as st

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

@st.cache
def load_data():
	df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
	new_df = df.drop_duplicates()

	to_cast = list(filter(lambda c: c if len(new_df[c].unique()) <= 5 else None,
                       new_df.columns))
	for col in to_cast:
		new_df[col] = new_df[col].astype('category')
	return df, new_df


def correlation_quant(df):
	fig, ax = plt.subplots(figsize=(10, 8))
	sns.heatmap(data=df.astype({'DEATH_EVENT': 'int64'}).corr(),
	            annot=True, cmap='Spectral', cbar_kws={'aspect': 50},
	            square=True, ax=ax)
	plt.xticks(rotation=30, ha='right');
	plt.tight_layout()
	st.write(fig)

def cramers_corrected_stat(contingency_table):
    
    try:
        chi2 = chi2_contingency(contingency_table)[0]
    except ValueError:
        return np.NaN
    
    n = contingency_table.sum().sum()
    phi2 = chi2/n
    
    r, k = contingency_table.shape
    r_corrected = r - (((r-1)**2)/(n-1))
    k_corrected = k - (((k-1)**2)/(n-1))
    phi2_corrected = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    
    return (phi2_corrected / min( (k_corrected-1), (r_corrected-1)))**0.5

def categorical_corr_matrix(df):

    df = df.select_dtypes(include='category')
    cols = df.columns
    n = len(cols)
    corr_matrix = pd.DataFrame(np.zeros(shape=(n, n)), index=cols, columns=cols)
    
    excluded_cols = list()
    
    for col1 in cols:
        for col2 in cols:
            if col1 == col2:
                corr_matrix.loc[col1, col2] = 1
                break
            df_crosstab = pd.crosstab(df[col1], df[col2], dropna=False)
            corr_matrix.loc[col1, col2] = cramers_corrected_stat(df_crosstab)
                
    corr_matrix += np.tril(corr_matrix, k=-1).T
    return corr_matrix


def correlation_categorical(df):
	fig, ax = plt.subplots(figsize=(11, 5))
	sns.heatmap(categorical_corr_matrix(df), annot=True, cmap='Spectral', 
	            cbar_kws={'aspect': 50}, square=True, ax=ax)
	plt.xticks(rotation=30, ha='right');
	plt.tight_layout()
	st.write(fig)


def visualization_categorical(df):

    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(13, 11))
    titles = list(df.select_dtypes(include='category'))

    ax_title_pairs = zip(axs.flat, titles)

    for ax, title in ax_title_pairs:
        sns.countplot(x=title, data=df, palette='Pastel1', ax=ax)

    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.6, 
                    hspace=0.8)

    st.write(fig)



def visualization_continuous(df):

    df_grouped = df.groupby(by='DEATH_EVENT')
    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(12, 11))
    titles = list(df.select_dtypes(exclude='category'))

    ax_title_pairs = zip(axs.flat, titles)

    for ax, title in ax_title_pairs:
        sns.distplot(df_grouped.get_group(0)[title], bins=10, ax=ax, label='No')
        sns.distplot(df_grouped.get_group(1)[title], bins=10, ax=ax, label='Yes')
        ax.legend(title='DEATH_EVENT')
        
    axs.flat[-1].remove()

    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.6, 
                    hspace=0.8)

    st.write(fig)


def return_categoricals(df, threshold=5):

    return list(filter(lambda c: c if len(df[c].unique()) <= threshold else None,
                       df.columns))


def to_categorical(columns, df):
    for col in columns:
        df[col] = df[col].astype('category')
    return df


def modified_countplot(**kargs):
    """
        Assumes that columns to be plotted are in of pandas dtype='CategoricalDtype'
    """
    facet_gen = kargs['facet_generator']    ## Facet generator over facet data
    curr_facet, facet_data = None, None
    
    while True:
        ## Keep yielding until non-empty dataframe is found
        curr_facet = next(facet_gen)            ## Yielding facet genenrator
        df_rows = curr_facet[1].shape[0]
        
        ## Skip the current facet if its corresponding dataframe empty
        if df_rows:
            facet_data = curr_facet[1]
            break
    
    x_hue = (kargs.get('x'), kargs.get('hue'))
    cols = [col for col in x_hue if col]
    col_categories = [facet_data[col].dtype.categories if col else None for col in x_hue]
    
    palette = kargs['palette'] if 'palette1' in kargs.keys() else 'Pastel1'
    sns.countplot(x=cols[0], hue=x_hue[1], 
                  order=col_categories[0], hue_order=col_categories[1],
                  data=facet_data.loc[:, cols], palette=palette)


def smoking_blood(df, modified_countplot):

    facet = sns.FacetGrid(df, row='smoking', col='sex', sharex=False,
                          sharey=False, margin_titles=True)
    facet.map(modified_countplot, x='high_blood_pressure', hue='DEATH_EVENT',
              palette='Pastel2', facet_generator=facet.facet_data())
    facet.set_xlabels('high_blood_pressure')
    facet.set_ylabels('Count')
    facet.add_legend(title='DEATH_EVENT')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()


def diabetes_anemia(df, modified_countplot):

    facet = sns.FacetGrid(df, row='diabetes', col='sex', sharex=False,
                          sharey=False, margin_titles=True)
    facet.map(modified_countplot, x='anaemia', hue='DEATH_EVENT',
              palette='Pastel2', facet_generator=facet.facet_data())
    facet.set_xlabels('anaemia')
    facet.set_ylabels('Count')
    facet.add_legend(title='DEATH_EVENT')
    st.pyplot()



def knn(x, y):

    knn = KNeighborsClassifier(weights='distance')
    cv_scores = cross_val_score(knn, x, y, cv=5)
    st.write('mean validation accuracy for K Nearest Neighbors: ', np.mean(cv_scores))

def ridge(x, y):
    ridge = RidgeClassifier()
    cv_scores = cross_val_score(ridge, x, y, cv=5)
    st.write('mean validation accuracy for Ridge Regression: ', np.mean(cv_scores))


def random_forest(x, y):
    rf = RandomForestClassifier(max_depth=4, criterion='entropy',class_weight = 'balanced')
    cv_scores = cross_val_score(rf, x, y, cv=5)
    st.write('mean validation accuracy for Random Forest: ', np.mean(cv_scores))

def mlp(x, y):
    mlp = MLPClassifier(random_state=0, max_iter=1000, early_stopping=True)
    cv_scores = cross_val_score(mlp, x,y, cv=5)
    st.write('mean validation accuracy for MLP: ', np.mean(cv_scores))

def gassianNB(x, y):
    gau_nb = GaussianNB()
    cv_scores = cross_val_score(gau_nb, x, y, cv=5)
    st.write('mean validation accuracy for GaussianNB: ', np.mean(cv_scores))




df, new_df = load_data()

st.title("Heart Failure Prediction")

st.write("Cardiovascular diseases (CVDs) are ranked to have the highest death rate globally, which takes about 17.9 millions of lives annually and accounts for about 31% of all deaths worldwide. Heart failure is a common symptom of CVDs, which brings serious consequences, such as death.") 

st.write("This clinical dataset is from Kaggle (https://www.kaggle.com/andrewmvd/heart-failure-clinical-data) and contains 12 features, 300 rows of data, which can be used to predict mortality by heart failure. Most CVDs can be prevented by addressing behaviral risk factors such as smoking, obesity, lack of physical, alcohol, etc. People with or with high CVDs risk need early detection, and machine learning models might be a good choice.")


# Part 1: Overview on yelp covid features
st.markdown("## 1. Heart Failure Overview")
st.write("Let's first look at the raw dataframe from the original dataset.")

st.dataframe(df.head())

st.write("To avoid any confusions, note that **time** suggests follow-up period (days) and **ejection_fraction** suggests percentage of blood leaving the heart at each contraction. ")

st.write("After clear inspection, this dataset does not have any missing values. The 12 features can be splitted into two categoreis: quantitative and categorical. I further changed the categorical variables as type: categorical.")
st.markdown("- Quantitative: **age**, **creatinine_phosphokinase**, **ejaction_fraction**, **platelets**, **serum_creatinine**, **serum_sodium**, **time**")
st.markdown("- Categorical: **anaemia**, **diabetes**, **high_blood_pressure**, **sex**, **smoking**, **DEATH_EVENT**")


description = st.checkbox('Wanna check the statistics for quantitative variables?')
if description: 
    st.dataframe(df.describe().T)

# Part 2: Data Correlation
st.markdown("## 2. Corellation Visualization")

st.write("You can choose quantitative or categorical values to inspect the correclation matrix. Since we want to predict mortality, I included DEATH_EVENT in both cases.")
choices = st.multiselect(
    'Quantitative or Categorical?',
    ('Quantitative', 'Categorical')
)

if 'Quantitative' in choices:
    correlation_quant(new_df)
    st.write("For quantitative variables, I used Pearson correlation. We can see that time (folllow-up period), serum_creatinine, ejaction_fraction, and are are the four most correlated factors. ")

    

if 'Categorical' in choices:
    correlation_categorical(new_df)
    st.write("For categorical variables, I used Cramer's V correlation. We can see that smoking and and high_blood_pressure are two most correlated factors. ")


#Part 3: Data Visualization
st.markdown("## 3. Data Visualization")


st.markdown("### 3.1 How long do they plan to close")

st.write("You can choose quantitative or categorical values to see the its distribution.")
choices = st.multiselect(
    'Quantitative or Categorical?',
    ('Quantitative Variables', 'Categorical Variables')
)

if 'Quantitative Variables' in choices:
    visualization_continuous(new_df)
    st.write("For continuous variables, I seperated the data by DEATH_EVENT. We can see that for serum_sodium, ejaction_fraction, and density, the distribution are quite different: the mean are apprantly different. ")


if 'Categorical Variables' in choices:
    visualization_categorical(new_df)
    st.write("For categorical variables, we can see that there is some imbalance of data: for DEATH_EVENT, there are about 200 people dead but only 100 alive. Anaemia and diabetes relatively equat number of data. For other factors, the count between two categories are off by about half. A very interesting thing to notice is that diabetes, sex, and smoking seem to be uncorrelated to DEATH_EVENT at all.")

to_cast = return_categoricals(df, threshold=5)
df = to_categorical(to_cast, df)


st.markdown("### 3.2 How does gender influence heart failure based on unhealthy habits?")

smoking = st.checkbox('Wanna see how does heart failure among gender based on smoking and blood pressure?')
if smoking:
    smoking_blood(df, modified_countplot)
    st.write("We can see that combining smoking and high blood pressure make many male die from heart failure, while not creating a significant case for female. Gender does make some interesing contrasts between some variables. ")

anemia = st.checkbox('Wanna see how does heart failure among gender based on anemia and diabetes?')
if anemia: 
    diabetes_anemia(df, modified_countplot)
    st.write("We don't notice anyting much significant here, but it's interesting that having both of diabetes and anaemia for male have a higher chance of dying from heart failure than female. Gender does make some interesing contrasts between some variables. ")

#Part 4: ML models
st.markdown("## 4. Machine Learning Models")

x = df.iloc[:, :-1]
y = df['DEATH_EVENT']

st.write("Now, we would like to explore which machine learning model can predict mortality by heart failure the best. You can choose the following ML models to inspect the performances. You can also choose different combinations of variables to compare their performances. We used cross validations for all five models and computed the average accuracy: 0.8 for traning, and 0.2 for validation. This is a simple binary classification problem. To deal with the data imbalance problem, I used upsampling method. ")

total_targets = x.columns[0:]
label = st.multiselect('Select the variables that you would like to explore.', total_targets)

x = x[label]


choices = st.multiselect(
    'ML Model to choose: ',
    ('KNN', 
    'Ridge Regression',
    'Random Forest',
    'MLP',
    'Gaussian NB'))

if 'KNN' in choices:
    knn(x, y)

if 'Ridge Regression' in choices:
    ridge(x, y)

if 'Random Forest' in choices:
    random_forest(x, y)

if 'MLP' in choices:
    mlp(x, y)

if 'Gaussian NB' in choices:
    gassianNB(x, y)


