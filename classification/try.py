import h5py

f = h5py.File('/home/abhishek/django_project4/classification/model/new_model.h5', 'r')
print(f.attrs.get('keras_version')) 