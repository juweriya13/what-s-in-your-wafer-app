import requests

def get_current_weather(city, state=None):
    """
    Fetches the current temperature for a given city using Open-Meteo API (Free, no key required).
    """
    if not city:
        return None
        
    try:
        # 1. Geocoding: Get lat/lon for the city
        search_query = f"{city}"
        if state:
            search_query += f" {state}"
            
        geocode_url = f"https://geocoding-api.open-meteo.com/v1/search?name={search_query}&count=1&language=en&format=json"
        geo_resp = requests.get(geocode_url, timeout=5)
        
        if geo_resp.status_code == 200:
            geo_data = geo_resp.json()
            if geo_data.get('results') and len(geo_data['results']) > 0:
                location = geo_data['results'][0]
                lat = location['latitude']
                lon = location['longitude']
                
                # 2. Weather: Get current temperature
                weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
                weather_resp = requests.get(weather_url, timeout=5)
                
                if weather_resp.status_code == 200:
                    weather_data = weather_resp.json()
                    current_temp = weather_data.get('current_weather', {}).get('temperature')
                    return current_temp
                    
    except Exception as e:
        print(f"Weather API error: {e}")
        
    return None
