from io import BytesIO
from django.core.files.base import ContentFile
from django.core.files.uploadedfile import InMemoryUploadedFile

def convert_numpy_to_django_files(pil_image, image_name):
    thumb_io = BytesIO()
    pil_image.save(thumb_io, format="png")
    return ContentFile(thumb_io.getvalue())
    
    #thumb_file = InMemoryUploadedFile(thumb_io, None, image_name, "image/png", thumb_io.len, None)


    #return thumb_file