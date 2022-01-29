from django.shortcuts import render
import pickle as pkl
import numpy as np

# Create your views here.

def index(request):
    return render(request, 'index.html')

def load(filename):
    file = open(filename, 'rb')
    data = pkl.load(file)
    file.close()
    return data

def predict(request):
    model = load('model.pkl')
    gender_label = load('gender_label.pkl')
    gender_onehot = load('gender_onehot.pkl')
    severity_label = load('severity_label.pkl')
    severity_onehot = load('severity_onehot.pkl')
    contact_label = load('contact_label.pkl')
    contact_onehot = load('contact_onehot.pkl')
    minmax = load('minmax.pkl')

    age = int(request.GET['age'])
    gender = request.GET['gender']
    fever = int(request.GET['temp'])
    bodypain = int(request.GET['body'])
    runny_nose = int(request.GET['runny'])
    diff = int(request.GET['diff'])
    nasal = int(request.GET['nasal'])
    sore_throat = int(request.GET['sore'])
    severity = request.GET['sev']
    contact = request.GET['contact']

    gender = gender_onehot.transform([gender_label.transform([gender])]).toarray()
    severity = severity_onehot.transform([severity_label.transform([severity])]).toarray()
    contact = contact_onehot.transform([contact_label.transform([contact])]).toarray()
    test_data = np.array([[age, fever, bodypain, runny_nose, diff, nasal, sore_throat]])
    test_x = np.c_[test_data, gender, severity, contact]
    test_x = minmax.transform(test_x)
    pred = model.predict(test_x)

    return render(request,'predict.html')



