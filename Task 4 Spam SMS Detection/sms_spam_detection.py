#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd


# In[2]:


df = pd.read_csv('spam.csv', encoding='ISO-8859-1')


# In[3]:


df.sample(5)


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace =True)


# In[7]:


df.sample(5)


# In[8]:


df.rename(columns={'v1':'target','v2':'text'},inplace=True)


# In[9]:


df.sample(5)


# In[10]:


from sklearn.preprocessing import LabelEncoder
encoder =LabelEncoder()


# In[11]:


df['target']=encoder.fit_transform(df['target'])


# In[12]:


df.head()


# In[13]:


df.isnull().sum()


# In[14]:


df.duplicated().sum()


# In[15]:


df=df.drop_duplicates(keep='first')


# In[16]:


df.duplicated().sum()


# In[17]:


df.shape


# In[18]:


df.head()


# In[19]:


df['target'].value_counts()


# In[20]:


import matplotlib.pyplot as plt


# In[21]:


plt.pie(df['target'].value_counts(),labels=['ham','spam'],autopct="%0.2f")
plt.show()


# In[22]:


df['num_characters']=df['text'].apply(len)


# In[23]:


import nltk
nltk.data.path.append('C:\\Users\\pc\\AppData\\Roaming\\nltk_data')


from nltk.tokenize import sent_tokenize

text = "This is a sample sentence. It contains multiple sentences. NLTK will tokenize it."
sentences = sent_tokenize(text)

print(sentences)



# In[24]:


df.head()


# In[25]:


df['num_words']=df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[26]:


df.head()


# In[27]:


df['num_sentences']=df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[28]:


df.head()


# In[29]:


df[['num_characters','num_words','num_sentences']].describe()


# In[30]:


df[df['target']==0][['num_characters','num_words','num_sentences']].describe()


# In[31]:


df[df['target']==1][['num_characters','num_words','num_sentences']].describe()


# In[32]:


import seaborn as sns


# In[33]:


sns.histplot(df[df['target']==0]['num_characters'])
sns.histplot(df[df['target']==1]['num_characters'],color='red')


# In[34]:


sns.pairplot(df,hue='target')


# In[35]:


sns.heatmap(df.corr(),annot=True)


# In[36]:


from nltk.corpus import stopwords

english_stopwords = stopwords.words('english')
print(english_stopwords)


# In[37]:


import string
string.punctuation


# In[38]:


from nltk.stem.porter import PorterStemmer
ps =PorterStemmer()
ps.stem('dancing')


# In[39]:


def transform_text(text):
    text =text.lower()
    text =nltk.word_tokenize(text)
    
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english')and i not in string.punctuation:
            y.append(i)
            
        text=y[:]
        y.clear()
        
        for i in text:
            y.append(ps.stem(i))
    return " ".join(y)


# In[40]:


transform_text("I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today.")


# In[41]:


df['transformed_text']=df['text'].apply(transform_text)


# In[42]:


df.head()


# In[43]:


transform_text('I yt loved .How about you?')


# In[44]:


df['text'][10]


# In[45]:


import wordcloud


# In[46]:


from wordcloud import WordCloud
wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')


# In[47]:


spam_wc=wc.generate(df[df['target']==1]['transformed_text'].str.cat(sep=" "))


# In[48]:


plt.imshow(spam_wc)


# In[49]:


ham_wc=wc.generate(df[df['target']==0]['transformed_text'].str.cat(sep=" "))


# In[50]:


plt.imshow(spam_wc)


# In[51]:


df.head()


# In[52]:


spam_corpus = []

for msg in df[df['target'] == 1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)


# In[53]:


len(spam_corpus)


# In[56]:


from collections import Counter


# In[57]:


word_counts = Counter(spam_corpus)
word_counts_df = pd.DataFrame(word_counts.most_common(10), columns=["Word", "Count"])
sns.barplot(x="Word", y="Count", data=word_counts_df)
plt.show()


# In[58]:


ham_corpus = []

for msg in df[df['target'] == 0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)


# In[59]:


len(ham_corpus)


# In[60]:


word_counts = Counter(ham_corpus)
word_counts_df = pd.DataFrame(word_counts.most_common(10), columns=["Word", "Count"])
sns.barplot(x="Word", y="Count", data=word_counts_df)
plt.show()


# In[61]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)


# In[62]:


X = tfidf.fit_transform(df['transformed_text']).toarray()


# In[63]:


X.shape


# In[64]:


y= df['target'].values


# In[65]:


y


# In[66]:


from sklearn.model_selection import train_test_split


# In[67]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)


# In[68]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


# In[69]:


gnb=GaussianNB()
mnb=MultinomialNB()
bnb=BernoulliNB()


# In[70]:


gnb.fit(X_train,y_train)
y_pred1 =gnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[71]:


mnb.fit(X_train,y_train)
y_pred2 =mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))


# In[72]:


bnb.fit(X_train,y_train)
y_pred3 =bnb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))


# In[73]:


get_ipython().system('pip install xgboost')


# In[74]:


import xgboost


# In[75]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


# In[76]:


svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)
xgb = XGBClassifier(n_estimators=50,random_state=2)


# In[77]:


clfs = {
    'SVC' : svc,
    'KN' : knc, 
    'NB': mnb, 
    'DT': dtc, 
    'LR': lrc, 
    'RF': rfc, 
    'AdaBoost': abc, 
    'BgC': bc, 
    'ETC': etc,
    'GBDT':gbdt,
    'xgb':xgb
}


# In[78]:


def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    
    return accuracy,precision


# In[79]:


train_classifier(svc,X_train,y_train,X_test,y_test)


# In[80]:


accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
    
    current_accuracy,current_precision = train_classifier(clf, X_train,y_train,X_test,y_test)
    
    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)


# In[81]:


performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)


# In[82]:


performance_df


# In[83]:


performance_df1 = pd.melt(performance_df, id_vars = "Algorithm")


# In[84]:


performance_df1


# In[85]:


sns.catplot(x = 'Algorithm', y='value', 
               hue = 'variable',data=performance_df1, kind='bar',height=5)
plt.ylim(0.5,1.0)
plt.xticks(rotation='vertical')
plt.show()


# In[86]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_max_ft_3000':accuracy_scores,'Precision_max_ft_3000':precision_scores}).sort_values('Precision_max_ft_3000',ascending=False)


# In[87]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_scaling':accuracy_scores,'Precision_scaling':precision_scores}).sort_values('Precision_scaling',ascending=False)


# In[88]:


new_df = performance_df.merge(temp_df,on='Algorithm')


# In[89]:


new_df_scaled = new_df.merge(temp_df,on='Algorithm')


# In[90]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_num_chars':accuracy_scores,'Precision_num_chars':precision_scores}).sort_values('Precision_num_chars',ascending=False)


# In[91]:


new_df_scaled.merge(temp_df,on='Algorithm')


# In[92]:


svc = SVC(kernel='sigmoid', gamma=1.0,probability=True)
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)

from sklearn.ensemble import VotingClassifier


# In[93]:


voting = VotingClassifier(estimators=[('svm', svc), ('nb', mnb), ('et', etc)],voting='soft')


# In[94]:


voting.fit(X_train,y_train)


# In[95]:


new_df_scaled.merge(temp_df,on='Algorithm')


# In[96]:


import pickle 
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))


# In[ ]:




