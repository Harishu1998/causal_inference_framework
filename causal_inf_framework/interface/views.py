from django.http import HttpResponse
from django.shortcuts import render
import pandas as pd

def first_page(request):
    return render(request, 'first_page.html')

def bsts(request):

    return render(request, 'bsts_page.html')

def effect(request):
    return render(request, 'effect_page.html')
 
def upload_file(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['file']
        df = pd.read_csv(uploaded_file)

        # Save the DataFrame or perform any necessary processing

        return render('File uploaded successfully')

    return HttpResponse('Invalid request method')
