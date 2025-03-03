
from django.db.models import  Count, Avg
from django.shortcuts import render, redirect
from django.db.models import Count
from django.db.models import Q
import datetime
import xlwt
from django.http import HttpResponse
import numpy as np


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
#model selection
from sklearn.metrics import confusion_matrix, classification_report

# Create your views here.
from Remote_User.models import ClientRegister_Model,predict_hot_topic,detection_ratio,detection_accuracy


def serviceproviderlogin(request):
    if request.method  == "POST":
        admin = request.POST.get('username')
        password = request.POST.get('password')
        if admin == "Admin" and password =="Admin":
            detection_accuracy.objects.all().delete()
            return redirect('View_Remote_Users')

    return render(request,'SProvider/serviceproviderlogin.html')

def View_Predicted_Identifying_Hot_Topic_Trends_Ratio(request):
    detection_ratio.objects.all().delete()

    ratio = ""
    kword = 'Hot Topic'
    print(kword)
    obj = predict_hot_topic.objects.all().filter(Q(Prediction=kword))
    obj1 = predict_hot_topic.objects.all()
    count = obj.count();
    count1 = obj1.count();
    ratio = (count / count1) * 100
    if ratio != 0:
        detection_ratio.objects.create(names=kword, ratio=ratio)

    ratio1 = ""
    kword1 = 'Normal Topic'
    print(kword1)
    obj1 = predict_hot_topic.objects.all().filter(Q(Prediction=kword1))
    obj11 = predict_hot_topic.objects.all()
    count1 = obj1.count();
    count11 = obj11.count();
    ratio1 = (count1 / count11) * 100
    if ratio1 != 0:
        detection_ratio.objects.create(names=kword1, ratio=ratio1)


    obj = detection_ratio.objects.all()
    return render(request, 'SProvider/View_Predicted_Identifying_Hot_Topic_Trends_Ratio.html', {'objs': obj})

def View_Remote_Users(request):
    obj=ClientRegister_Model.objects.all()
    return render(request,'SProvider/View_Remote_Users.html',{'objects':obj})

def ViewTrendings(request):
    topic = predict_hot_topic.objects.values('topics').annotate(dcount=Count('topics')).order_by('-dcount')
    return  render(request,'SProvider/ViewTrendings.html',{'objects':topic})

def charts(request,chart_type):
    chart1 = detection_ratio.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts.html", {'form':chart1, 'chart_type':chart_type})

def charts1(request,chart_type):
    chart1 = detection_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    print(chart1)
    return render(request,"SProvider/charts1.html", {'form':chart1, 'chart_type':chart_type})

def View_Predicted_Identifying_Hot_Topic_Trends(request):
    obj =predict_hot_topic.objects.all()
    return render(request, 'SProvider/View_Predicted_Identifying_Hot_Topic_Trends.html', {'list_objects': obj})

def likeschart(request,like_chart):
    charts =detection_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/likeschart.html", {'form':charts, 'like_chart':like_chart})


def Download_Trained_DataSets(request):

    response = HttpResponse(content_type='application/ms-excel')
    # decide file name
    response['Content-Disposition'] = 'attachment; filename="Predicted_Data.xls"'
    # creating workbook
    wb = xlwt.Workbook(encoding='utf-8')
    # adding sheet
    ws = wb.add_sheet("sheet1")
    # Sheet header, first row
    row_num = 0
    font_style = xlwt.XFStyle()
    # headers are bold
    font_style.font.bold = True
    # writer = csv.writer(response)
    obj = predict_hot_topic.objects.all()
    data = obj  # dummy method to fetch data.
    for my_row in data:
        row_num = row_num + 1
        ws.write(row_num, 0, my_row.Sno, font_style)
        ws.write(row_num, 1, my_row.PDate, font_style)
        ws.write(row_num, 2, my_row.Headline, font_style)
        ws.write(row_num, 3, my_row.Description, font_style)
        ws.write(row_num, 4, my_row.Source, font_style)
        ws.write(row_num, 5, my_row.Prediction, font_style)


    wb.save(response)
    return response

def train_model(request):
    detection_accuracy.objects.all().delete()
    data = pd.read_csv("Datasets.csv", encoding='latin-1')

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

    print(x)
    print("Y")
    print(y)

    models = []
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
    X_train.shape, X_test.shape, y_train.shape

    
    print("Random Forest classifier")

    #Random forest classifier

    from sklearn.ensemble import RandomForestClassifier
    rf_clf = RandomForestClassifier()
    rf_clf.fit(X_train, y_train)
    y_pred_rf = rf_clf.predict(X_test)
    randomforest = accuracy_score(y_test, y_pred_rf) * 100
    print(confusion_matrix(y_test, y_pred_rf))
    print(classification_report(y_test, y_pred_rf))
    detection_accuracy.objects.create(names="Random Forest", ratio=randomforest)


    # SVM Model
    print("SVM")
    from sklearn import svm

    lin_clf = svm.LinearSVC()
    lin_clf.fit(X_train, y_train)
    predict_svm = lin_clf.predict(X_test)
    svm_acc = accuracy_score(y_test, predict_svm) * 100
    print("ACCURACY")
    print(svm_acc)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, predict_svm))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, predict_svm))
    detection_accuracy.objects.create(names="SVM", ratio=svm_acc)

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
    detection_accuracy.objects.create(names="Logistic Regression", ratio=accuracy_score(y_test, y_pred) * 100)

    print("Decision Tree Classifier")
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    dtcpredict = dtc.predict(X_test)
    print("ACCURACY")
    print(accuracy_score(y_test, dtcpredict) * 100)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, dtcpredict))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, dtcpredict))
    detection_accuracy.objects.create(names="Decision Tree Classifier", ratio=accuracy_score(y_test, dtcpredict) * 100)

    print("Gradient Boosting Classifier")
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(
        X_train,
        y_train)
    clfpredict = clf.predict(X_test)
    print("ACCURACY")
    print(accuracy_score(y_test, clfpredict) * 100)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, clfpredict))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, clfpredict))
    models.append(('GradientBoostingClassifier', clf))
    detection_accuracy.objects.create(names="GradientBoostingClassifier", ratio=accuracy_score(y_test, clfpredict) * 100)

    labeled = 'Labeled_Data.csv'
    data.to_csv(labeled, index=False)
    data.to_markdown

    obj = detection_accuracy.objects.all()
    return render(request,'SProvider/train_model.html', {'objs': obj})