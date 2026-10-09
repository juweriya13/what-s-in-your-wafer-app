from django.urls import path
from .views import OCRView, WebhookView, FetchParsedView

urlpatterns = [
    path('ocr/', OCRView.as_view(), name='ocr'),
    path('webhook/', WebhookView.as_view(), name='webhook'),
    path('fetch/<str:request_id>/', FetchParsedView.as_view(), name='fetch_parsed'),
]
