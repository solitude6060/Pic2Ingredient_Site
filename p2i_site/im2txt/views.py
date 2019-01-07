from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseForbidden
from django import forms
from django.conf import settings
import os
from django.contrib.staticfiles.templatetags.staticfiles import static
#for keras
from pickle import load
from .utils.model import *
from keras.models import load_model
import numpy as np
from PIL import Image
from keras.applications.inception_v3 import preprocess_input
import imghdr
from keras.models import model_from_json
from keras import backend as K


def handle_uploaded_file(f, fname):
    path = './test_data/'
    with open(path+fname, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)

def extract_features(filename):
    print("!!!!!!!!!!!!!!!!!!!")
    model = defineCNNmodel()
    # load the photo
    image = load_img(filename, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # get features
    feature = model.predict(image, verbose=0)
    return feature


def index(request):
    img_path = "static/images/ncku.png"
    caption = "NCKU CSIE ICML COURSE"
    if 'upload' in request.POST:
        print("call upload !")
        dirpath = settings.PROJECT_ROOT+"/test_data/"
        abs_path = "E:\Git Project\Course\ML_final_project\pic2ingredient\p2i_site\im2txt\static\images"
        
        for f in os.listdir(dirpath):
            os.remove(dirpath+f)
        
        fname = request.FILES['myfile'].name
        filepath = dirpath+fname
        with open(filepath, 'wb+') as destination:
            for chunk in request.FILES['myfile'].chunks():
                destination.write(chunk)
        destination.close()

        with open(abs_path+"/"+fname, 'wb+') as destination:
            for chunk in request.FILES['myfile'].chunks():
                destination.write(chunk)
        destination.close()
        
        img_path = "static/images/"+fname
        print(img_path)
        print("-----------------------")
        print("running test!")

        # load the tokenizer
        tokenizer_path = settings.PROJECT_ROOT+'/model_data/tokenizer.pkl'
        tokenizer = load(open(tokenizer_path, 'rb'))
        # pre-define the max sequence length (from training)
        max_length = 38
        # load the model
        model_path = settings.PROJECT_ROOT+'/model_data/model_recipe_7.h5'
        with open(settings.PROJECT_ROOT+"/model_data/model_structure.json", "r") as text_file:
            json_string = text_file.read()
        
        model = Sequential()
        model = model_from_json(json_string)
        model.load_weights(model_path, by_name=False)

        # load and prepare the photograph
        test_path = settings.PROJECT_ROOT+'/test_data'
        for image_file in os.listdir(test_path):
                try:
                    image_type = imghdr.what(os.path.join(test_path, image_file))
                    if not image_type:
                        continue
                except IsADirectoryError:
                    continue
        image = extract_features(settings.PROJECT_ROOT+"/test_data/"+image_file)
        # generate description
        description = generate_desc(model, tokenizer, image, max_length)
        caption = 'Caption: ' + description.split()[1].capitalize()
        for x in description.split()[2:len(description.split())-1]:
            caption = caption + ' ' + x
        caption += '.'
        print(caption)
        print("-----------------------")
        K.clear_session()
    return render(request, 'im2txt/index.html', {"caption":caption, "img_path":img_path})
