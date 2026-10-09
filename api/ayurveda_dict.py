HEATING_INGREDIENTS = [
    'ginger', 'garlic', 'pepper', 'black pepper', 'sesame', 'jaggery', 'gur',
    'mustard', 'bajra', 'peanut', 'peanuts', 'chili', 'chilies', 'chilli', 'chillies',
    'cinnamon', 'clove', 'cloves', 'coffee', 'caffeine', 'turmeric', 'ajwain', 'asafoetida', 'hing'
]

COOLING_INGREDIENTS = [
    'coconut', 'fennel', 'saunf', 'cardamom', 'mint', 'pudina', 'coriander', 
    'ghee', 'cucumber', 'rose', 'watermelon', 'milk', 'sabja', 'basil seeds', 'sugar'
]

def get_ayurvedic_insight(ingredients_list, temperature=None):
    if temperature is None:
        return "No climate data provided. Add your location for Ayurvedic seasonal insights."
        
    found_heating = []
    found_cooling = []
    
    # Clean ingredients list
    cleaned_ingredients = [str(i).lower() for i in ingredients_list]
    
    for ing in cleaned_ingredients:
        # Check heating
        for heat_ing in HEATING_INGREDIENTS:
            if heat_ing in ing:
                found_heating.append(heat_ing)
                break # Move to next ingredient
                
        # Check cooling
        for cool_ing in COOLING_INGREDIENTS:
            if cool_ing in ing:
                found_cooling.append(cool_ing)
                break
                
    found_heating = list(set(found_heating))
    found_cooling = list(set(found_cooling))
    
    insight = ""
    
    if temperature > 30: # Summer / Hot Weather
        if len(found_heating) >= 2:
            insight = f"☀️ Summer Warning: It's {temperature}°C. This product contains heating ingredients ({', '.join(found_heating)}). According to Ritucharya, consuming these may increase body heat (Pitta) and is not ideal for the current weather."
        elif len(found_cooling) >= 1:
            insight = f"☀️ Summer Benefit: Contains cooling ingredients ({', '.join(found_cooling)}) which are excellent for the current {temperature}°C weather."
        else:
            insight = f"☀️ It's {temperature}°C. This product is generally neutral for the current weather."
            
    elif temperature < 20: # Winter / Cool Weather
        if len(found_heating) >= 1:
            insight = f"❄️ Winter Benefit: Contains warming ingredients ({', '.join(found_heating)}) which are beneficial for the current {temperature}°C weather according to Ritucharya."
        elif len(found_cooling) >= 2:
            insight = f"❄️ Winter Warning: It's {temperature}°C. This product contains cooling ingredients ({', '.join(found_cooling)}) which may increase coldness (Kapha) in the body."
        else:
            insight = f"❄️ It's {temperature}°C. This product is generally neutral for the current weather."
            
    else: # Moderate Weather
        insight = f"☁️ Mild Weather ({temperature}°C). The ingredients are fine for this climate."
        
    return insight
