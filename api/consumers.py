import json
from channels.generic.websocket import AsyncWebsocketConsumer

class OCRConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.request_id = self.scope['url_route']['kwargs']['task_id']
        self.room_group_name = f'ocr_{self.request_id}'

        # Join room group
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        await self.accept()

    async def disconnect(self, close_code):
        # Leave room group
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    # Receive message from room group
    async def ocr_result(self, event):
        data = event['data']
        # Send message to WebSocket
        await self.send(text_data=json.dumps(data))
        await self.close()
