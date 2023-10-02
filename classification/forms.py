from django import forms
from .models import *
import os
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text) ]

class UploadTrainImage(forms.Form):
    dirs_available = next(os.walk('User_Custom_Train'))[1]
    dirs_available.sort(key = natural_keys)
    choices = [(i,i) for i in dirs_available]
    class_name = forms.ChoiceField(choices=choices)
    img_file = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': True}))


class UploadTestImage(forms.Form):
    testing_file = forms.FileField(widget=forms.ClearableFileInput())

class Augmentations(forms.Form):
    augs = forms.CharField(max_length=300,widget=forms.TextInput(attrs={'type':'hidden'}))
        