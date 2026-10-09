import os
import time
import requests
import json
import google.generativeai as genai
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from django.core.cache import cache
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List
from pydantic import BaseModel, Field

# Define Pydantic models for structured output (tool calling)
class DataPoint(BaseModel):
    label: str = Field(description="Item Name")
    value: float = Field(description="Numeric value")
    unit: str = Field(description="Unit like g, %, or mg")

class Visualization(BaseModel):
    chart_type: str = Field(description="Use pie_chart for percentages or proportions, bar_chart for absolute values/comparisons")
    title: str = Field(description="Chart Title")
    data: List[DataPoint] = Field(description="List of data points to visualize")

class NutritionalInfo100g(BaseModel):
    energy_kcal: float = Field(description="Energy in kcal per 100g", default=0.0)
    protein_g: float = Field(description="Protein in grams per 100g", default=0.0)
    sugar_g: float = Field(description="Total Sugar (including added sugars) in grams per 100g", default=0.0)
    fat_g: float = Field(description="Total Fat in grams per 100g", default=0.0)
    sodium_mg: float = Field(description="Sodium in mg per 100g", default=0.0)
    fiber_g: float = Field(description="Dietary Fiber in grams per 100g", default=0.0)

class HealthScore(BaseModel):
    score: int = Field(description="Score from 1 to 10 (1 is terrible/unhealthy, 10 is excellent/healthy)")
    rating: str = Field(description="One of: 'Excellent', 'Good', 'Moderate', 'Poor', 'Not Recommended'")
    label: str = Field(description="Target group label, e.g., 'Babies & Toddlers', 'Adults (18-60)', 'Seniors (60+)'")
    reason: str = Field(description="Short, strict reason for the score. Be extremely harsh for high sugar or additives.")

class AIHealthAssessment(BaseModel):
    baby: HealthScore = Field(description="Assessment for Babies & Toddlers. Be absolutely brutal: >15g sugar/100g or ANY artificial additives/colors means score MUST be <= 3 ('Not Recommended').")
    adult: HealthScore = Field(description="Assessment for Adults. >30g sugar/100g means score MUST be <= 4 ('Poor'). >60g sugar/100g MUST be <= 2 ('Poor').")
    senior: HealthScore = Field(description="Assessment for Seniors. Penalize high sodium and high sugar severely. Reward fiber and protein.")

class DashboardConfiguration(BaseModel):
    """Configuration for dashboard visualizations based on extracted packet text."""
    visualizations: List[Visualization] = Field(description="List of visualizations to render")
    allergens: List[str] = Field(description="List of identified allergens, such as Milk, Soy, Wheat, Nuts, etc.", default=[])
    is_veg: str = Field(description="Dietary indicator, either 'Veg', 'Non-Veg', or 'Unknown'", default="Unknown")
    health_warnings: List[str] = Field(description="List of health warnings based on high sugar, high fat, or artificial additives", default=[])
    ingredients_list: List[str] = Field(description="List of extracted individual ingredients", default=[])
    fssai_license: str = Field(description="FSSAI License number if found, else empty string", default="")
    nutritional_info_per_100g: NutritionalInfo100g = Field(description="Standardized nutritional information per 100g or 100ml. Calculate this from serving size if 'Per 100g' is not directly provided in the text.", default=None)
    health_indices: AIHealthAssessment = Field(description="AI-generated strict health scores based on the nutritional profile and ingredients.")

class ChartProposal(BaseModel):
    proposed_charts: List[str] = Field(description="List of proposed charts (e.g., 'Pie chart of Macronutrients: Protein, Carbs, Fats')")

# Define Graph State
class AgentState(TypedDict):
    input_text: str
    dashboard_json: str
    use_double_layer: bool

# Define Graph Node
def analyze_node(state: AgentState):
    input_text = state["input_text"]
    use_double_layer = state.get("use_double_layer", False)
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    model_name = os.environ.get("AI_MODEL_NAME", "gemini-1.5-flash")
    
    llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=gemini_api_key)
    structured_llm = llm.with_structured_output(DashboardConfiguration)
    
    if use_double_layer:
        # Step 1: Chart Proposal
        structured_proposer = llm.with_structured_output(ChartProposal)
        proposal_prompt = f"""
        Analyze the following OCR text from a product label and propose a list of charts that can be generated.
        You MUST include at least one 'pie_chart' (e.g., for macronutrient breakdown: Protein, Carbs, Fats) if such data exists.
        Describe the charts and the exact data fields that should go into them. Do NOT generate the JSON data yet.
        
        OCR Text:
        {input_text}
        """
        proposal = structured_proposer.invoke(proposal_prompt)
        chart_instructions = "\n".join([f"- {chart}" for chart in proposal.proposed_charts])
        
        # Step 2: Full Extraction
        prompt = f"""
        You are an expert data visualization and health analysis assistant. Read the text below extracted from a product packet.
        Follow these instructions carefully:
        1. Create visualizations STRICTLY based on the following proposed charts:
        {chart_instructions}
        2. VERY IMPORTANT: Extract nutritional info standardized to 'Per 100g' into `nutritional_info_per_100g`. If the packet only lists 'Per Serving', mathematically scale it to 100g.
        3. Determine strict health scores in `health_indices` for Baby, Adult, and Senior based on the standardized 100g data and ingredients. Be ruthless with high sugar! A product with 69g sugar per 100g is pure junk and MUST get scores of 1-3.
        4. Identify allergens, dietary indicator (Veg/Non-Veg), ingredients, and FSSAI License number.
        5. Add short warnings to `health_warnings` for high sugar, high fat, or controversial additives.
        
        Extracted Text:
        {input_text}
        """
    else:
        prompt = f"""
        You are an expert data visualization and health analysis assistant. Read the text below extracted from a product packet.
        1. Identify all quantitative data and determine the best way to visualize this data using charts.
        2. VERY IMPORTANT: Extract nutritional info standardized to 'Per 100g' into `nutritional_info_per_100g`. If the packet only lists 'Per Serving', mathematically scale it to 100g.
        3. Determine strict health scores in `health_indices` for Baby, Adult, and Senior based on the standardized 100g data and ingredients. Be ruthless with high sugar! A product with 69g sugar per 100g is pure junk and MUST get scores of 1-3.
        4. Identify allergens, dietary indicator (Veg/Non-Veg), ingredients, and FSSAI License number.
        5. Add short warnings to `health_warnings` for high sugar, high fat, or controversial additives.
        
        Extracted Text:
        {input_text}
        """
        
    result = structured_llm.invoke(prompt)
    return {"dashboard_json": result.model_dump_json()}

# Compile Graph
workflow = StateGraph(AgentState)
workflow.add_node("analyze", analyze_node)
workflow.add_edge(START, "analyze")
workflow.add_edge("analyze", END)
graph = workflow.compile()
def parse_with_gemini(data, request_id):
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    
    if not gemini_api_key:
        return {'status': 'success', 'data': data, 'is_structured': False}
        
    use_double_layer = cache.get(f"dl_{request_id}", False)
        
    try:
        print("=== SENDING TO GEMINI VIA LANGGRAPH ===")
        if use_double_layer:
            print("Mode: Double Layer AI Enabled")
        
        result = graph.invoke({"input_text": data, "use_double_layer": use_double_layer})
        parsed_json_str = result["dashboard_json"]
        
        print("=== RECEIVED FROM GEMINI ===")
        print(parsed_json_str)
        print("============================")
        
        structured_data = json.loads(parsed_json_str)
        # We no longer need to call calculate_health_indices(structured_data)
        # structured_data['health_indices'] is directly provided by Gemini
        
        return {'status': 'success', 'data': structured_data, 'is_structured': True, 'raw_text': data}
    except Exception as e:
        print(f"Gemini parsing failed: {e}")
        return {'status': 'success', 'data': data, 'is_structured': False, 'raw_text': data}

class OCRView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request, format=None):
        print("FILES:", request.FILES)
        print("POST:", request.POST)
        print("DATA:", request.data)
        file_obj = request.FILES.get('image')
        if not file_obj:
            return Response({'error': 'No image provided', 'received_files': list(request.FILES.keys())}, status=400)


        api_key = os.environ.get("DATALAB_API_KEY")
        if not api_key:
            return Response({'error': 'DATALAB_API_KEY not configured on server'}, status=500)
            
        headers = {
            "X-API-Key": api_key
        }

        # Ensure the filename has a valid extension for Datalab
        file_name = file_obj.name
        if not any(file_name.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp', '.gif', '.pdf']):
            file_name = 'image.jpg'

        # Submit the file to Datalab convert API
        submit_url = "https://www.datalab.to/api/v1/convert"
        
        # Datalab expects the correct content type and extension
        content_type = file_obj.content_type
        if content_type == 'application/octet-stream':
            content_type = 'image/jpeg'
            
        files = {
            'file': (file_name, file_obj.read(), content_type)
        }
        
        # We can also pass model configuration if needed
        data = {
            # 'model': 'chandra' # if required
        }

        submit_resp = requests.post(submit_url, headers=headers, files=files, data=data)
        if submit_resp.status_code != 200:
            return Response({'error': f'Datalab submission failed: {submit_resp.text}'}, status=submit_resp.status_code)

        resp_json = submit_resp.json()
        request_id = resp_json.get('request_id')

        if not request_id:
            return Response({'error': 'No request_id returned from Datalab'}, status=500)
            
        use_double_layer = request.POST.get('use_double_layer') == 'true'
        if use_double_layer:
            cache.set(f"dl_{request_id}", True, timeout=86400)

        # Return the request_id immediately. The client will open a WebSocket using this ID to listen for the result.
        return Response({
            'status': 'processing',
            'request_id': request_id,
            'message': 'Image submitted successfully. Connect to WebSocket to receive results.'
        })

class WebhookView(APIView):
    def post(self, request, *args, **kwargs):
        payload = request.data
        request_id = payload.get('request_id')
        webhook_secret = payload.get('webhook_secret')
        
        # 1. Verify the Webhook Secret
        expected_secret = os.environ.get('DATALAB_WEBHOOK_SECRET')
        if not expected_secret or webhook_secret != expected_secret:
            print(f"Webhook rejected: Invalid secret. Expected {expected_secret}, got {webhook_secret}")
            return Response({'error': 'Invalid webhook_secret'}, status=403)
        
        if not request_id:
            print("Webhook received with no request_id in payload:", payload)
            return Response({'status': 'ignored, no request_id'})
            
        # 2. The webhook doesn't contain the OCR data, just a notification that it's done.
        # We must fetch the actual result from Datalab.
        api_key = os.environ.get("DATALAB_API_KEY")
        headers = {"X-API-Key": api_key}
        # We can use the provided request_check_url or fallback to the convert endpoint
        check_url = payload.get('request_check_url', f"https://www.datalab.to/api/v1/convert/{request_id}")
        
        try:
            resp = requests.get(check_url, headers=headers)
            if resp.status_code == 200:
                poll_json = resp.json()
                status = poll_json.get('status')
                
                if status in ['complete', 'completed', 'finished', 'success', 'done']:
                    data = poll_json.get('markdown') or poll_json.get('result') or poll_json.get('output') or poll_json
                    # Pass the data to Gemini for parsing
                    msg = parse_with_gemini(data, request_id)
                else:
                    msg = {'error': f"OCR status is {status}, but webhook fired."}
            else:
                msg = {'error': f"Failed to fetch final result from Datalab: {resp.text}"}
        except Exception as e:
            msg = {'error': f"Error fetching result: {str(e)}"}
            
        # 3. Send to the WebSocket consumer waiting for this request_id
        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.group_send)(
            f'ocr_{request_id}',
            {
                'type': 'ocr_result',
                'data': msg
            }
        )
        
        return Response({'status': 'ok'})

class FetchParsedView(APIView):
    def get(self, request, request_id):
        api_key = os.environ.get("DATALAB_API_KEY")
        headers = {"X-API-Key": api_key}
        
        check_url = f"https://www.datalab.to/api/v1/marker/{request_id}"
        resp = requests.get(check_url, headers=headers)
        
        if resp.status_code != 200:
            check_url = f"https://www.datalab.to/api/v1/convert/{request_id}"
            resp = requests.get(check_url, headers=headers)
            
        if resp.status_code == 200:
            poll_json = resp.json()
            status = poll_json.get('status')
            
            if status in ['complete', 'completed', 'finished', 'success', 'done']:
                data = poll_json.get('markdown') or poll_json.get('result') or poll_json.get('output') or poll_json
                msg = parse_with_gemini(data, request_id)
                return Response(msg)
            else:
                return Response({'error': f"OCR status is {status}."}, status=400)
        else:
            return Response({'error': f"Failed to fetch from Datalab: {resp.text}"}, status=400)
