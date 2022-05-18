from django.conf import settings
from django.conf.urls.static import static

from django.urls import path
from .views import UploadView

urlpatterns = [
    path('predict/', UploadView.as_view(), name="fileupload")
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)