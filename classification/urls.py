from django.urls import path
from django.shortcuts import redirect
from . import views

urlpatterns = [
    path('',lambda request: redirect('home/', permanent=False)),
    path('AddTrainImage/', views.addTrainingImage, name='addTrainingImage'),
    path('testImage/', views.testImage, name='testImage'),
    path('graphs/', views.graphs, name='graphs'),
    path('augment/', views.augment, name='augmentInput'),
    path('retrain/', views.re_train_model, name='retrainingdata'),
    path('home/', views.direct , name="home"),
    path('Merge/', views.Merge , name="hub2"),
    path('image/', views.display_images , name="hub3"),
    path('analysis/', views.analysis_model,name="analysis"),
]
#
