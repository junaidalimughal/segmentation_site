from django.conf import settings

def main_page_media(request):
    return {"MEDIA_URL": settings.MEDIA_URL}