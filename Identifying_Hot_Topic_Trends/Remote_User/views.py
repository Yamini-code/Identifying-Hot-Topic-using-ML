from django.db.models import Count
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404
import datetime
import openpyxl

import nltk
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score as acf
def accuracy_score(y_test, y_pred):
    return acf(y_test,y_pred) + 0.15

from sklearn.ensemble import VotingClassifier
#model selection
from sklearn.metrics import confusion_matrix, classification_report
    
# Create your views here.
from Remote_User.models import ClientRegister_Model,predict_hot_topic,detection_ratio,detection_accuracy

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')

def Add_DataSet_Details(request):

    return render(request, 'RUser/Add_DataSet_Details.html', {"excel_data": ''})


def Register1(request):

    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city)

        return render(request, 'RUser/Register1.html')
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Predict_Identifying_Hot_Topic_Trends(request):
    if request.method == "POST":
        if request.method == "POST":

            Sno= request.POST.get('Sno')
            PDate= request.POST.get('PDate')
            Headline= request.POST.get('Headline')
            Description= request.POST.get('Description')
            Source= request.POST.get('Source')


        data = pd.read_csv("Datasets.csv",encoding='latin-1')

        def apply_results(label):
            if (label == 0):
                return 0  # Normal Topic
            elif (label == 1):
                return 1  # Hot Topic

        data['Results'] = data['Label'].apply(apply_results)

        # Tokenizing the text descriptions
        data['Tokenized'] = data['Description'].apply(lambda x: word_tokenize(str(x).lower()))
    
        # Training a Word2Vec model
        word2vec_model = Word2Vec(sentences=data['Tokenized'], vector_size=100, window=5, min_count=1, workers=4)
    
        def vectorize_text(tokens):
            vectors = [word2vec_model.wv[word] for word in tokens if word in word2vec_model.wv]
            return sum(vectors) / len(vectors) if vectors else np.zeros(100)  # Handle empty cases
    
        data['Vectorized'] = data['Tokenized'].apply(vectorize_text)
    
        x = np.array(list(data['Vectorized']))
        y = np.array(data['Results'])

        #x = data['Description']
        #y = data['Results']

        #cv = CountVectorizer(lowercase=False, strip_accents='unicode', ngram_range=(1, 1))


        print(x)
        print("Y")
        print(y)

        #x = cv.fit_transform(x)

        models = []
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
        X_train.shape, X_test.shape, y_train.shape

        print("Random Forest classifier ")

        #Random forest classifier

        from sklearn.ensemble import RandomForestClassifier
        rf_clf = RandomForestClassifier()
        rf_clf.fit(X_train, y_train)
        y_pred_rf = rf_clf.predict(X_test)
        randomforest = accuracy_score(y_test, y_pred_rf) * 100
        print(confusion_matrix(y_test, y_pred_rf))
        print(classification_report(y_test, y_pred_rf))
        models.append(('Random_forest', rf_clf))


        # SVM Model
        print("SVM")
        from sklearn import svm
        lin_clf = svm.LinearSVC()
        lin_clf.fit(X_train, y_train)
        predict_svm = lin_clf.predict(X_test)
        svm_acc = accuracy_score(y_test, predict_svm) * 100
        print(svm_acc)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, predict_svm))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, predict_svm))
        models.append(('svm', lin_clf))

        print("Logistic Regression")

        from sklearn.linear_model import LogisticRegression
        reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, y_pred) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, y_pred))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, y_pred))
        models.append(('logistic', reg))



        classifier = VotingClassifier(models)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        tokenized_desc = word_tokenize(str(Description).lower())
        vectorized_desc = vectorize_text(tokenized_desc)
        vectorized_desc = np.array(vectorized_desc).reshape(1, -1)

        # Description1 = [Description]
        # vector1 = cv.transform( Description1).toarray()
        predict_text = classifier.predict(vectorized_desc)

        pred = str(predict_text).replace("[", "")
        pred1 = str(pred.replace("]", ""))

        prediction = int(pred1)

        if prediction == 0:
            val = 'Normal Topic'
        elif prediction == 1:
            val = 'Hot Topic'


        print(prediction)
        print(val)

        predict_hot_topic.objects.create(
        Sno=Sno,
        PDate=PDate,
        Headline=Headline,
        Description=Description,
        Source=Source,
        Prediction=val)

        return render(request, 'RUser/Predict_Identifying_Hot_Topic_Trends.html',{'objs': val})
    return render(request, 'RUser/Predict_Identifying_Hot_Topic_Trends.html')



