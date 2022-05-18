from statistics import mode
from django.db import models
from django_resized import ResizedImageField
import django_resized



# Create your models here.
class ImageModelResize(models.Model):
    image_file = ResizedImageField(size=[512, 512], upload_to="media")
    mask_file = models.FileField(null=True)
    image_file_2 = ResizedImageField(size=[512, 512], upload_to="media", null=True)
    mask_file_2 = models.FileField(null=True)
    upload_date = models.DateTimeField(auto_now_add=True)


class ImageModel(models.Model):
    image_file = models.FileField()
    mask_file = models.FileField(null=True)
    upload_date = models.DateTimeField(auto_now_add=True)


    """def save(self, **kwargs):
        return super(ImageModel, self).save(**kwargs)
        
    def clean(self):
        print(self.image_file, self.image_file.shape)

        return self.image_file"""