from sklearn import datasets, linear_model
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

columns = "FinalGrade;PSS_Stress;TotalQuestions;avg_durationperquestion;avg_tbd;decision_time_efficiency;good_decision_time_efficiency;maxduration;median_tbd;minduration;num_decisions_made;question_enter_count;ratio_decisions;ratio_good_decisions;totalduration;variance_tbd".split(";")
data = pd.read_csv("Dataset_DecisionPSS.csv")
df = pd.DataFrame(data,columns=columns)
print(df)
dfimp = df.fillna(df.mean())
print(dfimp)
y=dfimp.iloc[:,1]
print(y)
X_train,X_test,Y_train,Y_test = train_test_split(dfimp,y,test_size=0.2)
svmgo = svm.SVC(gamma='scale')

svmgo.fit(X_train,Y_train)
svmgo.score(X_test,Y_test)
'''
#load the diabetes housing dataset
columns = "age sex bmi map tc ldl hdl tch ltg glu".split()#declare the columns names
diabetes = datasets.load_diabetes()#Call the diabetes dataset from sklearn
df = pd.DataFrame(diabetes.data,columns=columns)#load the dataset as pandas dataframe

y = diabetes.target#define the target variable (dependent variable) as y
#create train and test vars
X_train,X_test,Y_train,Y_test = train_test_split(df,y,test_size=0.2)

#fit a model
lm = linear_model.LinearRegression()

model = lm.fit(X_train,Y_train)
predictions = lm.predict(X_test)

print("Score:",model.score(X_test,Y_test))


kf = KFold(n_splits=10)#define the split - into 10 folds
kf.get_n_splits(X_train) # returns number of splitting iterations in the cross validator

print(kf)

from sklearn import metrics

#10-fold cross validation
scores = cross_val_score(model,df,y,cv=kf)
print("cross-validated scores:",scores)
print("Baseline Accuracy: %.2f%% (%.2f%%)" % (scores.mean()*100,scores.std()*100))
#make cross validated predictions

predictions = cross_val_predict(model,df,y,cv=10)
plt.scatter(y,predictions)

#/////////////////////////////////////////////////

#standardization & normalization
input_data = np.array([[3,-1.5,3,-6.4],[0,3,-1.3,4.1],[1,2.3,-2.9,-4.3]])
data_scaler = MinMaxScaler(feature_range=(0,1))
data_scaled = data_scaler.fit_transform(input_data)
print("min max scaled data",data_scaled)

data_normalized = normalize(input_data,norm = '11')
print('Normalized data:', data_normalized)

data_standardized = scale(input_data)
print("Standardized data : ", data_standardized)

#Lable enconder
lable_enconder = LableEnconder()
input_classes = ['toyota','ford','suzuki','bmw']
lable_enconder.fit(input_classes)

print("Class mapping:")

for i , item in enumerate(lable_enconder.classes_):
    print(item,'-->',i)
lables = ['toyota','ford','suzuki']
enconded_lables = lable_enconder.transform(lables)
print("Lables",lables)
print("Encoded Lables",list(enconded_lables))

#binarization
input_data2 = np.array([[3,-1.5,3,-6.4],[0,3,-1.3,4.1],[1,2.3,-2.9,-4.3]])

data_binarized = Binarizer(threshold=1.4).transform(input_data2)

print("binarized data = ",data_binarized)

#Missing values
X = np.array([[23.56],[53.45],["NaN"],[44.44],[77.78],["NaN"]])

imp = Imputer(missing_values='NaN', strate='mean', axis=0)
print("imputer = ",imp.fit_transform(X))

#Discretization

factors = np.random.randn(30)

pd.qcut(factors,5).value_counts()
pd.cut(factors,5).value_counts()
'''


