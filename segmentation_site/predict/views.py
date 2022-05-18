from unittest.mock import DEFAULT
from django.shortcuts import render
from django.core.files.images import ImageFile
from django.core.files.uploadedfile import InMemoryUploadedFile

import os
import numpy as np
from PIL import Image
from io import StringIO

from .tfpipeline import prepare_batches, denormalize
from .apps import model as tf_model
from .utilities import convert_numpy_to_django_files

import tensorflow as tf
from .forms import UploadImageModelForm

def upload_image_view(request):
    # create form object with a post request if post request is valid value otherwise none. 
    # Load the form with files if valid value otherwise do the same for files.
    image_upload_form = UploadImageModelForm(request.POST or None, request.FILES or None)
    
    # validate the form
    if image_upload_form.is_valid():
        obj = image_upload_form.save(commit=False)
        obj.save()

        # image file names for reading them from server
        if obj.image_file_2:

            image_files = [obj.image_file.name, obj.image_file_2.name]
            image_file_paths = [os.path.join(os.getcwd() + "/media/", img) for img in image_files]
            
            # batch the images 
            image_batches = prepare_batches(image_file_paths)
            output_masks = tf_model.predict(image_batches)
            
            # scale the values above 40 to 255, and below 40 to 0. To increase the visibility of the preditions.
            threshold = 20
            
            image_names_and_files = []
            
            for i, output_mask in enumerate(output_masks):
                
                output_mask = denormalize(output_mask, threshold)
                output_mask = Image.fromarray(output_mask.astype(np.uint8))
                output_mask = output_mask.convert("L")

                image_name = image_files[i][:-4] + "_mask.png"
                output_mask_django_files = convert_numpy_to_django_files(output_mask, image_name)
                
                image_names_and_files.append((image_name, output_mask_django_files))
            
            obj.mask_file.save( 
                os.path.basename(image_names_and_files[0][0]), 
                image_names_and_files[0][1]
            )

            obj.mask_file_2.save( 
                os.path.basename(image_names_and_files[1][0]), 
                image_names_and_files[1][1]
            )    
            
            context = {"image": obj.image_file, 
                        "mask":obj.mask_file, 
                        "image_2":obj.image_file_2, 
                        "mask_2":obj.mask_file_2, 
                        "first_image":"First Image",
                        "second_image":"Second Image"}
            return render(request, "predict/imagemodel_form.html", context)
        
    return render(request, "predict/imagemodel_form.html", {"form":image_upload_form})