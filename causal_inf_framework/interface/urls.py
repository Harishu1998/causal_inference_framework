from django.urls import path

from . import views

urlpatterns = [
    path("", views.first_page, name="firstpage"),
    path("bsts", views.bsts, name= 'bsts'),
    path('effect', views.effect, name= 'effect'),
    path('analyse/', views.upload_file, name='analyse'),
]