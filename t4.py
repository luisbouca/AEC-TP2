from sklearn import datasets, linear_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm,model_selection
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.preprocessing import Binarizer
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import VarianceThreshold,SelectFromModel

def getTreatedData(missingval,normalize,binarize):
    columns = "ExamID;FinalGrade;PSS_Stress;TotalQuestions;avg_durationperquestion;avg_tbd;decision_time_efficiency;good_decision_time_efficiency;maxduration;median_tbd;minduration;num_decisions_made;question_enter_count;ratio_decisions;ratio_good_decisions;totalduration;variance_tbd".split(";")
    data = pd.read_csv("Dataset_DecisionPSS.csv")
    df = pd.DataFrame(data,columns=columns)
    #df.drop(['StudyID', 'median_tbd'], axis=1) 
    print("---------------Before data Processing---------------")
    print(df)
    
    #if(normalize):

    if(binarize):
        teste = np.array(df['FinalGrade']).reshape(1, -1)
        data_binar=Binarizer(threshold=0.5).transform(teste)
        data_binar=np.transpose(data_binar) 
        df2 = pd.DataFrame({"FinalGrade":data_binar[:,0]})
        df['FinalGrade'] = df2['FinalGrade']

    #if(featureSelect):

    if missingval:
        df = df.fillna(df.mean())
    else:
        df.dropna(inplace=True)
    print("---------------After data Processing---------------")
    print(df)
    return df

def featureSelection():
    print("wow")

def Scorer(model,X_test,Y_test):
    score = model.score(X_test, Y_test)
    print("Score: "+ str(score))

def CheckModels(dataFrame,PredictFeature):
    X_train,X_test,Y_train,Y_test = train_test_split(dataFrame,PredictFeature,test_size=0.2) 
    scoring = 'accuracy'
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', svm.SVC(gamma="scale")))
    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg) 

def DecisionTree(dataFrame,PredictFeature):
    X_train,X_test,Y_train,Y_test = train_test_split(dataFrame,PredictFeature,test_size=0.2) 
    cart = DecisionTreeClassifier()
    cart.fit(X_train, Y_train)
    Scorer(cart,X_test,Y_test)

def SVC(dataFrame,PredictFeature,c,kernel,gamma,maxiter):
    X_train,X_test,Y_train,Y_test = train_test_split(dataFrame,PredictFeature,test_size=0.2) 
    model = svm.SVC(C=c,kernel=kernel,gamma=gamma,max_iter=maxiter)
    model.fit(X_train, Y_train)
    Scorer(model,X_test,Y_test)

df = getTreatedData(True,False,True)
y=df.iloc[:,2]
CheckModels(df,y)
DecisionTree(df,y)
#svc_param_selection(df,y,10)
#SVC(df,y,1,'sigmoid','scale',-1)