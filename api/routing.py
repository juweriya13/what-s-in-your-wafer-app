from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/ocr/(?P<task_id>[\w-]+)/$', consumers.OCRConsumer.as_asgi()),
]
