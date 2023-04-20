import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
import re

import nltk as nl
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from time import perf_counter
import warnings
warnings.filterwarnings(action='ignore')
from nltk.tokenize import word_tokenize
nltk.download('punkt')
stopwords = stopwords.words('english')
stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()

dataset = pd.read_csv("C://Users/Maca/Desktop/spam_ham_dataset_v5.csv")
dataset.head()
dataset.info()
dataset_mails = dataset.drop(dataset.columns[0:2], axis=1)
pd.set_option('display.max_colwidth', 80)
print(dataset_mails)


def clean_text(text):
    new_text = text.lower()
    clean_text = re.sub("[^a-z]+", " ", new_text)
    clean_text_stopwords = ""
    for i in clean_text.split(" ")[1:]:
        if not i in stopwords and len(i) > 3:
            clean_text_stopwords += i
            clean_text_stopwords += " "
            clean_text_stopwords = lemmatizer.lemmatize(clean_text_stopwords)
            clean_text_stopwords = stemmer.stem(clean_text_stopwords)
    return clean_text_stopwords


dataset_mails["text_clean"] = dataset_mails.text.apply(clean_text)
print(dataset_mails)

dataset_mails_clean = dataset_mails.drop(dataset_mails.columns[0:1],axis=1)
dataset_mails_clean['len'] = dataset_mails_clean['text_clean'].str.len()
print(dataset_mails_clean["text_clean"])

plt.rcParams['figure.figsize'] = (14, 9)
sns.boxenplot(x = dataset_mails_clean['label_num'], y = dataset_mails_clean['len'])
plt.title('odnos duzine spam i ham mejla')
plt.show()



plt.figure(figsize = (14, 9))
sns.countplot(data = dataset, x = 'label');


count1 = Counter(" ".join(dataset_mails_clean[dataset_mails_clean['label_num']==0]["text_clean"]).split()).most_common(20)
df1 = pd.DataFrame.from_dict(count1)
df1 = df1.rename(columns={0: "words in non-spam", 1 : "count"})
count2 = Counter(" ".join(dataset_mails_clean[dataset_mails_clean['label_num']==1]["text_clean"]).split()).most_common(20)
df2 = pd.DataFrame.from_dict(count2)
df2 = df2.rename(columns={0: "words in spam", 1 : "count_"})

df1.plot.bar(legend = False,color='blue')
y_pos = np.arange(len(df1["words in non-spam"]))
plt.xticks(y_pos, df1["words in non-spam"])
plt.title('Najcesce reci u ham mejlovima')
plt.xlabel('Reci')
plt.ylabel('Broj')
plt.show()

df2.plot.bar(legend = False, color = 'orange')
y_pos = np.arange(len(df2["words in spam"]))
plt.xticks(y_pos, df2["words in spam"])
plt.title('Najcesce reci u spam mejlovima')
plt.xlabel('Reci')
plt.ylabel('Broj')
plt.show()


x = dataset_mails_clean['text_clean']
y = dataset_mails_clean['label_num']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)
print(x_train.shape, x_test.shape)

bow_vec = CountVectorizer()
train_bow = bow_vec.fit_transform(x_train)
test_bow = bow_vec.transform(x_test)
cv_df = pd.DataFrame(train_bow.toarray(),columns = bow_vec.get_feature_names_out())
cv_df.head()

feature_names = bow_vec.get_feature_names_out()
print("Number of features: {}".format(len(feature_names)))


def ML_modeling(models,params,  X_train, X_test, y_train, y_test, performance_metrics):
    if not set(models.keys()).issubset(set(params.keys())):
        raise ValueError('Some estimators are missing parameters')

    for key in models.keys():
        model = models[key]
        param = params[key]
        gs = GridSearchCV(model,param, cv=10, error_score=0, refit=True)
        gs.fit(X_train, y_train)
        y_pred = gs.predict(X_test)


        accuracy_sc = accuracy_score(y_test, y_pred)
        precision_sc = precision_score(y_test, y_pred, average='macro')
        recall_sc = recall_score(y_test, y_pred, average='macro')
        f1_sc = f1_score(y_test, y_pred, average='macro')

        performance_metrics.append([key, accuracy_sc, precision_sc, recall_sc, f1_sc])
        print(key, ':', gs.best_params_)
        print("Accuracy: %1.3f \tPrecision: %1.3f \tRecall: %1.3f \t\tF1: %1.3f\n" % (
        accuracy_sc, precision_sc, recall_sc, f1_sc))

    return

models = {
    'SVM': SVC(),
    'Naive Bayes': MultinomialNB(),
}

params = {
    'SVM': { 'kernel': ['linear', 'rbf'] },
    'Naive Bayes': { 'alpha': [0.5, 1], 'fit_prior': [True, False] },
}


performance_metrics_bow = []
print("==============Bag of Words==============\n")
ML_modeling(models,params, train_bow, test_bow, y_train, y_test, performance_metrics_bow)

metrics_bow_df = pd.DataFrame(performance_metrics_bow,columns=['Model' , 'Accuracy', 'Precision' , 'Recall', "F1 Score"])

tfidf = TfidfVectorizer()
train_tfidf = tfidf.fit_transform(x_train)
test_tfidf = tfidf.transform(x_test)
tfidf_df = pd.DataFrame(train_tfidf.toarray(), columns = tfidf.get_feature_names())

print("==============TF-IDF==============\n")
performance_metrics_tfidf = []
ML_modeling(models,params, train_tfidf, test_tfidf, y_train, y_test, performance_metrics_tfidf)

metrics_tfidf_df = pd.DataFrame(performance_metrics_tfidf,columns=['Model' , 'Accuracy', 'Precision' , 'Recall', "F1 Score"])
print("==============Bag of Words==============\n")
print(metrics_bow_df)
print("==============TF-IDF==============\n")
print(metrics_tfidf_df)

labels = ['Accuracy', "F1 Score", 'Precision' , 'Recall']
sm = [round(metrics_bow_df['Accuracy'][0],2), round(metrics_bow_df['F1 Score'][0],2), round(metrics_bow_df['Precision'][0],2), round(metrics_bow_df['Recall'][0],2)]
nb = [round(metrics_bow_df['Accuracy'][1],2), round(metrics_bow_df['F1 Score'][1],2), round(metrics_bow_df['Precision'][1],2), round(metrics_bow_df['Recall'][1],2)]
sm2 = [metrics_tfidf_df['Accuracy'][0], metrics_tfidf_df['F1 Score'][0], metrics_tfidf_df['Precision'][0], metrics_tfidf_df['Recall'][0]]
nb2 = [metrics_tfidf_df['Accuracy'][1], metrics_tfidf_df['F1 Score'][1], metrics_tfidf_df['Precision'][1], metrics_tfidf_df['Recall'][1]]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x+width/2 , sm, width, label='Bag-of-Words')
rects2 = ax.bar(x-width/2 , sm2, width, label='TF-IDF')

ax.set_title('SVM')
ax.set_ylabel('Parameters')
ax.set_xticks(x, labels)
ax.legend()

fig.tight_layout()
plt.show()


x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x+width/2 , nb, width, label='Bag-of-Words')
rects2 = ax.bar(x-width/2 , nb2, width, label='TF-IDF')

ax.set_title('Naive Bayes')
ax.set_ylabel('Parameters')
ax.set_xticks(x, labels)
ax.legend()


fig.tight_layout()

plt.show()


X = bow_vec.fit_transform(dataset_mails_clean['text_clean'])

# Get the categories
y1 = dataset['label_num']
X_train, X_test, y1_train, y1_test = train_test_split(X, y1, test_size=0.2, random_state=0)




models2 = {
    "Random Forest": {"model":RandomForestClassifier(), "perf":0},
    "Gradient Boosting": {"model":GradientBoostingClassifier(), "perf":0},
    "XGBoost": {"model":XGBClassifier(eval_metric='mlogloss'), "perf":0},
    "MultinomialNB": {"model":MultinomialNB(), "perf":0},
    "Logistic Regr.": {"model":LogisticRegression(), "perf":0},
    "KNN": {"model":KNeighborsClassifier(), "perf":0},
    "Decision Tree": {"model":DecisionTreeClassifier(), "perf":0},
    "SVM (Linear)": {"model":LinearSVC(), "perf":0},
    "SVM (RBF)": {"model":SVC(), "perf":0}
}

for name, model in models2.items():
    start = perf_counter()
    model['model'].fit(X_train, y1_train)
    duration = perf_counter() - start
    duration = round(duration,2)
    model["perf"] = duration
    print(f"{name:20} trained in {duration} sec")


models_acc = []
for name, model in models2.items():
    models_acc.append([name, model["model"].score(X_test, y1_test),model["perf"]])

df_acc = pd.DataFrame(models_acc)
df_acc.columns = ['Model', 'Accuracy w/o scaling', 'Training time (sec)']
df_acc.sort_values(by = 'Accuracy w/o scaling', ascending = False, inplace=True)
df_acc.reset_index(drop = True, inplace=True)
print(df_acc)

plt.figure(figsize = (15,5))
sns.barplot(x = 'Model', y = 'Accuracy w/o scaling', data = df_acc)
plt.title('Accuracy on the test set\n(the Y-Axis is between 0.8 and 1.0)', fontsize = 15)
plt.ylim(0.8,1)
plt.show()

plt.figure(figsize = (15,5))
sns.barplot(x = 'Model', y = 'Training time (sec)', data = df_acc)
plt.title('Training time for each model in sec', fontsize = 15)
plt.ylim(0,40)
plt.show()

dtv = bow_vec.transform(x_train)

#lr = LogisticRegression(verbose=1)
#lr = LogisticRegression(solver='liblinear', penalty ='l2' , C = 1.0)
#lr=SVC(probability=True,kernel='rbf')
lr=MultinomialNB(alpha=0.5,fit_prior=False)
#lr= DecisionTreeClassifier()
#lr= XGBClassifier(eval_metric='mlogloss')
#lr=KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski')
#lr=GradientBoostingClassifier()
#lr=RandomForestClassifier()
lr.fit(dtv, y_train)


def predict_class(lr):
    text = input('Enter Text(Subject of the mail): ')
    text = [' '.join([ word for word in word_tokenize(text)  if not word in stopwords])]
    t_dtv = bow_vec.transform(text).toarray()
    print('Predicted Class:', end = ' ')
    print('Spam' if lr.predict(t_dtv)[0] else 'Ham')
    prob = lr.predict_proba(t_dtv)*100
    print(f"Ham: {prob[0][0]}%\nSpam: {prob[0][1]}%")
    plt.figure(figsize=(12, 6))
    sns.barplot(x =['Ham', 'Spam'] , y = [prob[0][0], prob[0][1]])
    plt.xlabel('Class')
    plt.ylabel('Probalility')
    plt.show()


predict_class(lr)