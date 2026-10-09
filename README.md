# What's in Your Wafer App - Backend

This is the Django backend for the What's in Your Wafer application. It handles OCR processing, interacts with the Gemini AI model to extract nutritional information and ingredients from product labels, and calculates detailed health indices for different age groups (Babies, Adults, and Seniors).

## Setup
1. Create a virtual environment: `python -m venv venv`
2. Activate the environment: `venv\Scripts\activate` (Windows)
3. Install dependencies: `pip install -r requirements.txt`
4. Run migrations: `python manage.py migrate`
5. Start the server: `python manage.py runserver`
