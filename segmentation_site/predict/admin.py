from django.contrib import admin
from .models import ImageModel, ImageModelResize
# Register your models here.
admin.site.register(ImageModel)
admin.site.register(ImageModelResize)
