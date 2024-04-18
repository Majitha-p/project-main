from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict, name='predict'),  # Route the 'predict/' URL to the predict view
   
    path('scrape/', views.scrape, name='scrape'),  # Route the 'scrape/' URL to the scrape view
]
