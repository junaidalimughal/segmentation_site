from django import forms
from .models import ImageModel, ImageModelResize

class UploadImageModelForm(forms.ModelForm):
    class Meta:
        model = ImageModelResize
        fields = ["image_file", "image_file_2"]
        labels = {
            "image_file": "Select Image",
            "image_file_2": "Select Second Image"
        }