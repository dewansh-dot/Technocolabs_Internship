#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


df = pd.read_csv(r"F:\4th sem\internship\Technocolab 4th sem internship\cleaned data.csv")


# In[10]:


df = pd.DataFrame(df)


# In[11]:


df.info()


# In[12]:


df.drop('SYMBOL', axis = 1 , inplace = True)


# In[13]:


df.drop('DATE', axis = 1 , inplace = True)


# In[14]:


df


# In[15]:


df.isnull().sum()


# In[16]:


df.info()


# In[17]:


#for standardizing features, we ll use the standardScaler module
from sklearn.preprocessing import StandardScaler
#sk learn is one of the most widly used lib for ml we,ll use it for kmeans and pca module
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# In[18]:


from sklearn.preprocessing import MinMaxScaler

ms = MinMaxScaler()
segmentation_std = ms.fit_transform(df)


# In[19]:


segmentation_std 


# In[20]:


#PCA
pca = PCA()
pca.fit(segmentation_std)


# In[21]:


#the attribute shows how much variance is explained by each of the 5 individual components
pca.explained_variance_ratio_


# In[22]:


plt.figure(figsize = (10,8))
plt.plot(range(0,5),pca.explained_variance_ratio_.cumsum(),marker ='o',linestyle ='--')
plt.title ('Explained Varaince by components')
plt.xlabel('Number of Components')
plt.ylabel('cumulative Explained Variance')


# In[23]:


pca = PCA(n_components = 3)


# In[24]:


pca.fit(segmentation_std )


# In[25]:


pca.transform(segmentation_std )


# In[26]:


scores_pca = pca.transform(segmentation_std )


# In[ ]:





# In[27]:


wcss = []
for i in range(1, 10):
    kmeans_pca = KMeans(n_clusters=i ,init = 'k-means++', random_state=42)
    kmeans_pca.fit(scores_pca)
    wcss.append(kmeans_pca.inertia_)


# In[28]:


#Elbow Method for K means
# Import ElbowVisualizer
from yellowbrick.cluster import KElbowVisualizer


# In[29]:


model = KMeans()
# k is range of number of clusters.
visualizer = KElbowVisualizer(model, k=(1,10),random_state = 42, timings= True)
visualizer.fit(df)        # Fit data to visualizer
visualizer.show()


# In[30]:


plt.figure(figsize = (10,8))
plt.plot(range(1,10), wcss, marker ='o' , linestyle ='--')
plt.xlabel('no of cluster')
plt.ylabel('wcss')
plt.title ('K-MEANS WITH PCA')



# In[31]:


kmeans_pca = KMeans(n_clusters = 3, init = 'k-means++',random_state=42) 


# In[32]:


kmeans_pca.fit(scores_pca)


# In[33]:


df_seg_pca_kmeans = pd.concat([df.reset_index(drop = True),pd.DataFrame(scores_pca)], axis =1)
df_seg_pca_kmeans.columns.values[-3: ] =['component 1','component 2','component 3']

df_seg_pca_kmeans['Segment k-means PCA'] = kmeans_pca.labels_


# In[34]:


df_seg_pca_kmeans.head(50)


# In[35]:


df_seg_pca_kmeans['Segment'] =df_seg_pca_kmeans['Segment k-means PCA'].map({0:'first',1:'second',2:'third'}) 


# In[36]:


import seaborn as sns
sns.set()


# In[37]:


x_axis = df_seg_pca_kmeans['component 2']
y_axis = df_seg_pca_kmeans['component 1']
plt.figure(figsize = (10,8))
sns.scatterplot(x_axis ,y_axis ,hue =df_seg_pca_kmeans['Segment'], palette =['g','r','c'] )
plt.title('cluster by PCA components')
plt.show()


# ### LOGISTIC REGRESSION

# #### 1.Import required libraries

# In[38]:


import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import preprocessing

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from matplotlib import pyplot as plt

import seaborn as sns


# #### 2. exploring the data

# In[39]:


df_seg_pca_kmeans 


# In[42]:


dataset =df_seg_pca_kmeans 


# In[43]:


dataset.drop('component 1', axis=1, inplace=True)
dataset.drop('component 2', axis=1, inplace=True)
dataset.drop('component 3', axis=1, inplace=True)
dataset.drop('Segment', axis=1, inplace=True)


# In[54]:


dataset.rename(columns = {"OPEN":"OPEN","HIGH":"HIGH","LOW":"LOW","CLOSE":"CLOSE","VOLUME":"VOLUME","Segment k-means PCA":"cluster"},inplace = True)


# In[55]:


dataset


# #### 3. Preprocessing the Dataset

# In[56]:


X = dataset.iloc[:,dataset.columns != 'cluster']
y = dataset.cluster


# In[60]:


X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.20, random_state=5, stratify=y)


# In[61]:


scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)


# #### 4. Exploratory Data Visualization

# In[62]:


import matplotlib.colors as mcolors
colors = list(mcolors.CSS4_COLORS.keys())[10:]
def draw_histograms(dataframe, features, rows, cols):
    fig=plt.figure(figsize=(20,20))
    for i, feature in enumerate(features):
        ax=fig.add_subplot(rows,cols,i+1)
        dataframe[feature].hist(bins=20,ax=ax,facecolor=colors[i])
        ax.set_title(feature+" Histogram",color=colors[35])
        ax.set_yscale('log')
    fig.tight_layout() 
    plt.savefig('Histograms.png')
    plt.show()
draw_histograms(dataset,dataset.columns,8,4)


# In[63]:


plt.figure(figsize = (38,16))
sns.heatmap(dataset.corr(), annot = True)
plt.savefig('heatmap.png')
plt.show()


# ### Model Building and Training

# #### 1 – Building the Logistic Regression Model

# In[64]:


model = LogisticRegression()


# In[65]:


model.fit(X_train_scaled, y_train)


# ### Evaluating the model

# #### 1 – Evaluating on Training Set

# In[66]:


train_acc = model.score(X_train_scaled, y_train)
print("The Accuracy for Training Set is {}".format(train_acc*100))


# In[71]:


X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)


# #### 2 – Evaluating on Test Set

# In[72]:


test_acc = accuracy_score(y_test, y_pred)
print("The Accuracy for Test Set is {}".format(test_acc*100))


# #### 3 – Generating Classification Report

# In[73]:


print(classification_report(y_test, y_pred))


# #### 4 – Visualizing using Confusion Matrix

# In[74]:


cm=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(12,6))
plt.title("Confusion Matrix")
sns.heatmap(cm, annot=True,fmt='d', cmap='Blues')
plt.ylabel("Actual Values")
plt.xlabel("Predicted Values")
plt.savefig('confusion_matrix.png')


# ### NAIVE BAYES

# ####  Importing the libraries

# In[75]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn


# #### Splitting the dataset into the Training set and Test set

# In[76]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# #### Feature Scaling

# In[78]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# #### Training the Naive Bayes model on the Training set

# In[79]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# #### Predicting the Test set results

# In[80]:


y_pred  =  classifier.predict(X_test)


# In[81]:


y_pred 


# In[82]:


y_test


# #### Making the Confusion Matrix

# In[84]:


from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test,y_pred)


# In[86]:


cm


# In[87]:


ac


# In[89]:


ac


# In[90]:


cm=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(12,6))
plt.title("Confusion Matrix")
sns.heatmap(cm, annot=True,fmt='d', cmap='Blues')
plt.ylabel("Actual Values")
plt.xlabel("Predicted Values")
plt.savefig('confusion_matrix.png')


# In[ ]:




