from django.urls import path
from . import views

urlpatterns = [
    path('', views.BASE, name='BASE'),
    path('scrape/', views.scrape, name='scrape'),
    path('translate/', views.translate, name='translate'),
    path('highlight_similar_sentences/', views.highlight_similar_sentences, name='highlight_similar_sentences'),
    path('compare_sentences/', views.compare_sentences, name='compare_sentences'),
]