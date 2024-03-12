# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 03:26:33 2019

@author: nikolaos damianos
"""


def unique(list11): 
  
    # intilize a null list 
    unique_list = [] 
      
    # traverse for all elements 
    #for list1 in list11:    
    for x in list11: 
            # check if exists in unique_list or not 
        if x not in unique_list: 
            unique_list.append(x) 
            
    
    return unique_list




import pandas as pd
import numpy as np 
from sklearn import preprocessing
from collections import defaultdict

df = pd.read_csv("./movie_metadata.csv")
df.head()

for jj in list(df):
    print( jj,'\n')
print(unique(df['content_rating']))

######Feature Selection#######
what_we_want=['movie_title','director_name','director_facebook_likes','num_critic_for_reviews','num_voted_users','imdb_score','title_year',
              'movie_facebook_likes','duration','cast_total_facebook_likes','language','budget','country','num_user_for_reviews',
              'content_rating','actor_1_name','actor_1_facebook_likes','actor_2_name','actor_2_facebook_likes',
              'actor_3_name','actor_3_facebook_likes','genres','plot_keywords','gross']

clean_Dataset = []
features_that_we_want_noNaN = ['director_name','director_facebook_likes','num_critic_for_reviews','num_voted_users','imdb_score','title_year',
              'movie_facebook_likes','duration','cast_total_facebook_likes','language','budget','country','num_user_for_reviews',
              'actor_1_name','actor_1_facebook_likes','actor_2_name','actor_2_facebook_likes',
              'actor_3_name','actor_3_facebook_likes','genres','plot_keywords','gross']

for index, row in df.iterrows():
    row_feature = []
    add = True
    for feature in what_we_want:
        row_feature.append(row[feature])
        if feature in features_that_we_want_noNaN:
            if type(row[feature]) is str:
                if row[feature] == "":
                    add = False
                
            else:
                if np.isnan(row[feature]):
                    add = False
                
            
    
    if row_feature[14] in ['Not Rated', 'Unrated', 'Approved' ,'M', 'Passed']:
        row_feature[14] = 'PG-13'
    elif(row_feature[14] in ['X','TV-MA',]):
        row_feature[14] = 'NC-17'
    elif(row_feature[14] =='GP'):
        row_feature[14] = 'PG'
    elif not(type(row_feature[14])is str) and np.isnan(row_feature[14]):
        row_feature[14] = 'PG-13'
    if add == True:
        clean_Dataset.append(row_feature)


dataset = pd.DataFrame(clean_Dataset, columns=what_we_want)
print(unique(dataset['content_rating']))
dataset.to_csv('MovieRevenue5000.csv', index=False)   



#######################Normalization###########################################
copy_C_dataset = dataset.copy
Dataset_C_copied = copy_C_dataset()

classes = np.round(Dataset_C_copied['gross']/(10**8))
classes[classes == 21] = 8 


norm=preprocessing.RobustScaler()
dataset[['budget', 'gross','num_voted_users','movie_facebook_likes']] =  norm.fit_transform(dataset[['budget', 'gross','num_voted_users','movie_facebook_likes']])
Dataset_C_copied[['budget','num_voted_users','movie_facebook_likes']] = norm.fit_transform(dataset[['budget', 'num_voted_users','movie_facebook_likes']])
#####dioxnoume tous outliars
dataset=dataset[dataset['budget'] < 7]
Dataset_C_copied=Dataset_C_copied[Dataset_C_copied['budget'] < 7]
Dataset_C_copied['gross'] = classes
####kanoume ta strings ari8mous###########################
Dictionary_genres = defaultdict(preprocessing.LabelEncoder)
genres = dataset['genres'].str.split('|').apply(pd.Series, 1)#kane tin sunartish pd.series stis grammes
keywords= dataset['plot_keywords'].str.split('|').apply(pd.Series, 1)

genres_indexes = []
for gen in range(len(genres.columns)):
    
    genres_indexes.append("genre"+ str((gen+1)))

keywords_indexes = []
for gen in range(len(keywords.columns)):
    
    keywords_indexes.append("keyword"+ str((gen+1)))
    
genres.columns = genres_indexes
keywords.columns = keywords_indexes

copy_dataset = dataset.copy
Dataset_copied = copy_dataset()


genres = genres.fillna('')
keywords = keywords.fillna('')

genres_num = genres.apply(lambda x: Dictionary_genres[x.name].fit_transform(x))#stis stilles epidi den exei noumero
keywords_num = keywords.apply(lambda x: Dictionary_genres[x.name].fit_transform(x))

for row in range(len(genres_num)):
    if row in [967 ,1267, 2146, 2157,2696,2711,2760,2906,2932,3005,3292,3297,3633]:
        continue
    else:
        genres_num['genre1'][row] = genres_num['genre1'][row]+1
        

label_strings = ['director_name','language','country','content_rating','actor_1_name','actor_2_name','actor_3_name']
otherString  = Dataset_copied[label_strings]

label_strings_to_num = otherString.apply(lambda x: Dictionary_genres[x.name].fit_transform(x))
gross = Dataset_copied['gross']
Dataset_copied[label_strings] = label_strings_to_num
Dataset_copied=Dataset_copied.drop(columns=['genres','plot_keywords','gross'])

Dataset_C_copied[label_strings]= label_strings_to_num
Dataset_C_copied = Dataset_C_copied.drop(columns=['genres','plot_keywords'])


Dataset_copied = Dataset_copied.join(genres_num)
Dataset_copied = Dataset_copied.join(keywords_num)

Dataset_C_copied = Dataset_C_copied.join(genres_num)
Dataset_C_copied = Dataset_C_copied.join(keywords_num)

Dataset_copied  =Dataset_copied.join(gross)

classification_gross = pd.DataFrame(Dataset_C_copied['gross'])
Dataset_C_copied = Dataset_C_copied.drop(columns=['gross'])
Dataset_C_copied  = Dataset_C_copied.join(classification_gross)

Dataset_copied.to_csv('MovieRevenue5000_withoutStringsandNormalized_.csv', index=False)  
Dataset_C_copied.to_csv('MovieRevenue5000Classifiacation.csv', index=False)  

Dataset_copied=Dataset_copied.drop(columns=['movie_title','gross','genre4','genre5','genre6','genre7','genre8','language','cast_total_facebook_likes','keyword1','keyword2','keyword3','keyword4','keyword5','director_facebook_likes','country','imdb_score','duration'])
Dataset_C_copied = Dataset_C_copied.drop(columns=['movie_title','gross','genre4','genre5','genre6','genre7','genre8','language','cast_total_facebook_likes','keyword1','keyword2','keyword3','keyword4','keyword5','director_facebook_likes','country','imdb_score','duration'])
###############Train Test ######################################################
Dataset_copied = np.array(Dataset_copied)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(Dataset_copied, np.array(gross), 
                                                    test_size=0.25, random_state=47)



############# Models ################################################################

#############Libraries###############################################################
import tensorflow as tf
from sklearn.svm import SVR,SVC
from sklearn.metrics import mean_squared_error
import keras
from keras.models import Sequential
from keras.layers import Dense , Dropout
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier

def nn(in_dim):
    regressor = Sequential()
    regressor.add(Dense(units = 50, kernel_initializer = keras.initializers.RandomUniform(seed=47), activation = 'tanh', input_dim = in_dim))#'uniform'
    

    regressor.add(Dense(units = 40, kernel_initializer = keras.initializers.RandomUniform(seed=47)))
    regressor.add(keras.layers.LeakyReLU(alpha=0.3))

    regressor.add(Dense(units = 1, kernel_initializer = keras.initializers.RandomUniform(seed=47) ))
    regressor.add(keras.layers.LeakyReLU(alpha=0.5))

    regressor.compile(optimizer = keras.optimizers.RMSprop(lr=0.0001), loss = 'mse')
    
    return regressor

NN = nn(Dataset_copied.shape[1])
NN.fit(x_train, y_train, epochs = 1000)

model_json = NN.to_json()
with open("NN.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
NN.save_weights("NN.h5")
print("Saved model to disk")

def deepnn(in_dim):
    regressor = Sequential()
    regressor.add(Dense(units = 50, kernel_initializer = keras.initializers.RandomUniform(seed=47), activation = 'tanh', input_dim = in_dim))#'uniform'


    regressor.add(Dense(units = 50, kernel_initializer = keras.initializers.RandomUniform(seed=47), activation = 'tanh'))
    
    regressor.add(Dense(units = 40, kernel_initializer = keras.initializers.RandomUniform(seed=47), activation = 'tanh'))
    regressor.add(Dense(units = 40, kernel_initializer = keras.initializers.RandomUniform(seed=47), activation = 'tanh'))
    #regressor.add(Dense(units = 30, kernel_initializer = keras.initializers.RandomUniform(seed=47), activation = 'tanh'))
    
    
    regressor.add(Dense(units = 30, kernel_initializer = keras.initializers.RandomUniform(seed=47)))
    regressor.add(keras.layers.LeakyReLU(alpha=0.3))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units = 30, kernel_initializer = keras.initializers.RandomUniform(seed=47)))
    regressor.add(keras.layers.LeakyReLU(alpha=0.3))
    regressor.add(Dense(units = 20, kernel_initializer = keras.initializers.RandomUniform(seed=47)))
    regressor.add(keras.layers.LeakyReLU(alpha=0.4))
    regressor.add(Dropout(0.1))
    regressor.add(Dense(units = 20, kernel_initializer = keras.initializers.RandomUniform(seed=47)))
    regressor.add(keras.layers.LeakyReLU(alpha=0.5))
    
    regressor.add(Dense(units = 1, kernel_initializer = keras.initializers.RandomUniform(seed=47)))
    regressor.add(keras.layers.LeakyReLU(alpha=0.6))
    
    regressor.compile(optimizer = keras.optimizers.RMSprop(lr=0.0001), loss = 'mse')
    
    return regressor

NND = deepnn(Dataset_copied.shape[1])
NND.fit(x_train, y_train, epochs = 1000)

model_json = NND.to_json()
with open("deepnn.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
NN.save_weights("deepnn.h5")
print("Saved model to disk")


SVM_models = {}
mse_SVM = {}
for i in ['rbf','sigmoid']:
    SVM_models[i] = SVR(kernel=i,C = 5)


for i in ['rbf','sigmoid']:
    print(i)
    SVM_models[i].fit(x_train,y_train)
    mse_SVM[i] = mean_squared_error(SVM_models[i].predict(x_test),y_test)
    


mseNN = mean_squared_error(NN.predict(x_test),y_test)
mseDeepNN = mean_squared_error(NND.predict(x_test),y_test)
mse = pd.DataFrame({'NN':mseNN,'DeepNN':mseDeepNN}, index=[0])
Mse = pd.DataFrame(mse_SVM , index=[0])
Mse = Mse.join(mse)

Mse.to_csv('MSE.csv', index=False)



###############################################################################################
###############################Classification##################################################
###############################################################################################

def classification_labels(classi):
    classes = np.array(unique(classi))
    print(classes)
    __classes =  np.zeros((len(classi),len(classes)))
    
    classi = np.array(classi)
    
    print(classi.shape)
    for i in classes:
        ind = int(abs(i-8))
        
        for row in range(len(classi)):
            if (classi[row] == i):
                __classes[row][ind] = 1



    return __classes


dummy_target = classification_labels(classification_gross['gross'])
x_train, x_test, y_train, y_test = train_test_split(Dataset_C_copied,dummy_target , 
                                                    test_size=0.25, random_state=47)




def nnClass(in_dim):
    classifier = Sequential()
    classifier.add(Dense(units = 50, kernel_initializer = keras.initializers.RandomUniform(seed=47), activation = 'tanh', input_dim = in_dim))#'uniform'


    classifier.add(Dense(units = 60, kernel_initializer = keras.initializers.RandomUniform(seed=47)))
    classifier.add(keras.layers.LeakyReLU(alpha=0.5))

    classifier.add(Dense(units = 9, kernel_initializer = keras.initializers.RandomUniform(seed=47), activation = 'softmax'))


    classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    
    return classifier

NNC = nnClass(Dataset_C_copied.shape[1])
NNC.fit(x_train, np.array(y_train), epochs = 1000)





model_json = NNC.to_json()
with open("NNclass.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
NNC.save_weights("NNclass.h5")
print("Saved model to disk")

def deepnnClass(in_dim):
    classifier = Sequential()
    classifier.add(Dense(units = 50, kernel_initializer = keras.initializers.RandomUniform(seed=47), activation = 'tanh', input_dim = in_dim))#'uniform'
    classifier.add(Dropout(0.2))

    classifier.add(Dense(units = 40, kernel_initializer = keras.initializers.RandomUniform(seed=47), activation = 'tanh'))
    
    
    classifier.add(Dense(units = 35, kernel_initializer = keras.initializers.RandomUniform(seed=47), activation = 'tanh'))
    classifier.add(Dense(units = 30, kernel_initializer = keras.initializers.RandomUniform(seed=47), activation = 'tanh'))
    classifier.add(Dropout(0.2))
    
    classifier.add(Dense(units = 20, kernel_initializer = keras.initializers.RandomUniform(seed=47), activation = 'tanh'))
    
    classifier.add(Dense(units = 15, kernel_initializer = keras.initializers.RandomUniform(seed=47)))
    classifier.add(keras.layers.LeakyReLU(alpha=0.5))
    classifier.add(Dropout(0.1))
    classifier.add(Dense(units = 15, kernel_initializer = keras.initializers.RandomUniform(seed=47)))
    classifier.add(keras.layers.LeakyReLU(alpha=0.5))
    classifier.add(Dense(units = 9, kernel_initializer = keras.initializers.RandomUniform(seed=47), activation = 'softmax'))


    classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    
    return classifier

NNDC = deepnnClass(Dataset_C_copied.shape[1])
NNDC.fit(x_train, y_train, epochs = 1000)

model_json = NNDC.to_json()
with open("deepnnClass.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
NNDC.save_weights("deepnnClass.h5")
print("Saved model to disk")
acc = {}
predicted = NNC.predict(x_test)
predicted = np.argmax(predicted, axis=1)
acc['NN']=accuracy_score(np.argmax(y_test, axis=1), predicted)


predicted = NNDC.predict(x_test)
predicted = np.argmax(predicted, axis=1)
acc['deepNN']=accuracy_score(np.argmax(y_test, axis=1), predicted)

x_train, x_test, y_train, y_test = train_test_split(Dataset_C_copied,np.array(classification_gross['gross']) , 
                                                    test_size=0.25, random_state=47)
SVM_Class_models = {}
svms = ['rbf', 'sigmoid']
for i in svms:
    SVM_Class_models[i] = SVC(kernel=i)

def accuracy(y_pred, y_true):
    cnt = 0
    for i in range(len(y_true)):
        if y_pred[i] == y_true[i]:
            cnt+=1
    
    return cnt/float(len(y_true))


for i in svms:
    print(i)
    SVM_Class_models[i].fit(x_train, y_train)
    acc[i] = accuracy_score(np.array(y_test),np.array(SVM_Class_models[i].predict(x_test)))













ada_class = AdaBoostClassifier()

ada_class=ada_class.fit(x_train,y_train)

acc['AdaBoost']=accuracy_score(y_test, np.array(ada_class.predict(x_test)))
Accuracy = pd.DataFrame(acc , index=[0])


Accuracy.to_csv('ACC.csv', index=False)