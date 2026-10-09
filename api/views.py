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
from asgiref.sync import async_to_sync
from django.core.cache import cache

def calculate_health_indices(data):
    data_points = {}
    for viz in data.get('visualizations', []):
        for item in viz.get('data', []):
            label = str(item.get('label', '')).lower()
            try:
                val = float(item.get('value', 0.0))
                data_points[label] = val
            except (ValueError, TypeError):
                continue

    def get_val(keys):
        for k in keys:
            for dp_label, dp_val in data_points.items():
                if k in dp_label:
                    return dp_val
        return 0.0

    protein = get_val(['protein'])
    sugar = get_val(['sugar'])
    sodium = get_val(['sodium', 'salt'])
    fat = get_val(['fat'])
    fiber = get_val(['fibre', 'fiber'])
    energy = get_val(['energy', 'calories'])
    
    ingredients = [str(i).lower() for i in data.get('ingredients_list', [])]
    warnings = [str(w).lower() for w in data.get('health_warnings', [])]
    all_text = " ".join(ingredients + warnings)
    
    has_artificial_sweeteners = any(x in all_text for x in ['steviol', 'sucralose', 'aspartame', 'sweetener', 'saccharin'])
    has_caffeine = any(x in all_text for x in ['caffeine', 'coffee', 'tea'])
    has_synthetic_colors = any(x in all_text for x in ['synthetic color', 'synthetic colour', 'e102', 'e110', 'e133', 'e129'])
    has_preservatives = any(x in all_text for x in ['bha', 'bht', 'preservative', 'benzoate', 'sorbate', 'nitrite'])
    has_artificial_flavors = any(x in all_text for x in ['artificial flavor', 'artificial flavour', 'artificial cheese'])
    has_palm_oil = 'palm' in all_text

    # Adult
    adult_score = 8
    if sugar > 15: adult_score -= 2
    if fat > 20: adult_score -= 1
    if sodium > 400: adult_score -= 2
    if protein > 10: adult_score += 1
    if fiber > 3: adult_score += 1
    if has_preservatives or has_synthetic_colors: adult_score -= 1
    adult_score = max(1, min(10, adult_score))
    adult_rating = "Excellent" if adult_score >= 8 else "Good" if adult_score >= 6 else "Moderate" if adult_score >= 4 else "Poor"
    
    # Senior
    senior_score = 8
    if protein < 10: senior_score -= 2 
    if fiber < 2: senior_score -= 1
    if sodium > 300: senior_score -= 3
    if sugar > 10: senior_score -= 1
    if protein > 15: senior_score += 2
    if has_preservatives or has_synthetic_colors: senior_score -= 1
    senior_score = max(1, min(10, senior_score))
    senior_rating = "Excellent" if senior_score >= 8 else "Good" if senior_score >= 6 else "Moderate" if senior_score >= 4 else "Poor"
    
    # Baby
    baby_score = 10
    baby_reason = "Suitable based on data."
    
    if sodium > 100 or sodium > 1: # >100mg or >1g salt
        baby_score -= 3
        baby_reason = "Sodium/Salt too high."
    if sugar > 5: 
        baby_score -= 3
        baby_reason = "Added sugar not recommended."
    if fat > 10:
        baby_score -= 2
        if baby_score >= 8:
            baby_reason = "High fat content."
            
    dealbreakers = []
    if has_artificial_sweeteners: dealbreakers.append("sweeteners")
    if has_caffeine: dealbreakers.append("caffeine")
    if has_synthetic_colors: dealbreakers.append("synthetic colors")
    if has_preservatives: dealbreakers.append("preservatives (e.g., BHA/BHT)")
    if has_artificial_flavors: dealbreakers.append("artificial flavors")
    
    if dealbreakers:
        baby_score -= 6
        baby_reason = f"Avoid: Contains {', '.join(dealbreakers)}."
    elif has_palm_oil and baby_score >= 8:
        baby_score -= 1
        baby_reason = "Contains palm oil."
        
    if warnings and baby_score >= 6:
        baby_score -= 2
        if baby_reason == "Suitable based on data.":
            baby_reason = "Has health warnings."

    baby_score = max(1, min(10, baby_score))
    baby_rating = "Excellent" if baby_score >= 8 else "Good" if baby_score >= 6 else "Moderate" if baby_score >= 4 else "Not Recommended"

    return {
        "adult": {"score": adult_score, "rating": adult_rating, "label": "Adults (18-60)"},
        "senior": {"score": senior_score, "rating": senior_rating, "label": "Seniors (60+)"},
        "baby": {"score": baby_score, "rating": baby_rating, "label": "Babies & Toddlers", "reason": baby_reason if baby_score < 6 else ""}
    }

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

class DashboardConfiguration(BaseModel):
    """Configuration for dashboard visualizations based on extracted packet text."""
    visualizations: List[Visualization] = Field(description="List of visualizations to render")
    allergens: List[str] = Field(description="List of identified allergens, such as Milk, Soy, Wheat, Nuts, etc.", default=[])
    is_veg: str = Field(description="Dietary indicator, either 'Veg', 'Non-Veg', or 'Unknown'", default="Unknown")
    health_warnings: List[str] = Field(description="List of health warnings based on high sugar, high fat, or artificial additives", default=[])
    ingredients_list: List[str] = Field(description="List of extracted individual ingredients", default=[])
    fssai_license: str = Field(description="FSSAI License number if found, else empty string", default="")

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
        2. Identify any listed allergens (like Milk, Soy, Wheat) and determine if the product is 'Veg', 'Non-Veg', or 'Unknown'.
        3. Extract the list of ingredients individually.
        4. If Sugar or Fats are very high, or if there are controversial artificial additives, add short warnings to `health_warnings`.
        5. Extract the FSSAI License number if present.
        
        Extracted Text:
        {input_text}
        """
    else:
        prompt = f"""
        You are an expert data visualization and health analysis assistant. Read the text below extracted from a product packet.
        1. Identify all quantitative data (such as nutritional info: Energy, Protein, Carbs, Fats, etc.) and determine the best way to visualize this data using charts.
        2. Identify any listed allergens (like Milk, Soy, Wheat) and determine if the product is 'Veg', 'Non-Veg', or 'Unknown'.
        3. Extract the list of ingredients individually.
        4. If Sugar or Fats are very high, or if there are controversial artificial additives, add short warnings to `health_warnings`.
        5. Extract the FSSAI License number if present.
        
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
        structured_data['health_indices'] = calculate_health_indices(structured_data)
        
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
