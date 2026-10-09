# What's in Your Wafer App - Backend

This is the Django backend for the What's in Your Wafer application. It handles OCR processing, interacts with the Gemini AI model to extract nutritional information and ingredients from product labels, and calculates detailed health indices for different age groups (Babies, Adults, and Seniors).

## Setup

1. **Navigate to the backend directory:**

   ```powershell
   cd ocr_backend
   ```
2. **Create a virtual environment:**

   ```powershell
   python -m venv venv
   ```
3. **Activate the environment:**

   - **Windows (PowerShell):** `.\venv\Scripts\Activate.ps1`
   - **Windows (CMD):** `venv\Scripts\activate.bat`
   - **Mac/Linux:** `source venv/bin/activate`
4. **Install dependencies:**

   ```powershell
   pip install -r requirements.txt
   ```
5. **Run database migrations:**

   ```powershell
   python manage.py migrate
   ```
6. **Start the Django development server:**

   ```powershell
   python manage.py runserver
   ```

## Project Structure & File Roles

```text
Panda's_Project/
├── ocr_scanner/                  # Flutter Frontend App
│   ├── android/                  # Android native code
│   ├── ios/                      # iOS native code
│   ├── lib/
│   │   └── main.dart             # Main UI, camera, theme logic & rendering
│   ├── test/                     # Flutter unit and widget tests
│   ├── web/                      # Web platform code
│   ├── windows/                  # Windows platform code
│   ├── macos/                    # macOS platform code
│   ├── linux/                    # Linux platform code
│   ├── pubspec.yaml              # Flutter dependencies and project metadata
│   ├── pubspec.lock              # Locked dependency versions
│   ├── analysis_options.yaml     # Dart linting rules
│   └── README.md                 # Frontend specific readme
│
└── ocr_backend/                  # Django Backend API
    ├── backend/                  # Django Project Configuration
    │   ├── __init__.py
    │   ├── asgi.py               # Configured for Channels/WebSockets
    │   ├── settings.py           # Core Django settings, databases, INSTALLED_APPS
    │   ├── urls.py               # Root URL configuration
    │   └── wsgi.py               # WSGI config for production web servers
    │
    ├── api/                      # Main App Logic
    │   ├── migrations/           # Database migration files
    │   ├── __init__.py
    │   ├── admin.py              # Django admin panel configurations
    │   ├── apps.py               # App configuration
    │   ├── consumers.py          # WebSocket handling for live AI results
    │   ├── models.py             # Database models
    │   ├── routing.py            # WebSocket URL routing (Channels)
    │   ├── tests.py              # Backend unit tests
    │   ├── urls.py               # REST API endpoints
    │   └── views.py              # OCR, Gemini LangGraph & Health algorithm
    │
    ├── venv/                     # Python Virtual Environment (ignored in git)
    ├── .env                      # API Keys for Gemini and Datalab (ignored in git)
    ├── .gitignore                # Rules for what NOT to push to GitHub
    ├── db.sqlite3                # Local SQLite database
    ├── manage.py                 # Django command-line utility
    ├── requirements.txt          # List of Python dependencies to install
    └── README.md                 # This file
```

### 📱 Frontend (Flutter)

- `ocr_scanner/lib/main.dart`: Contains the entire Flutter UI logic. It handles taking pictures, uploading them to the backend, displaying the loading screens, rendering the AI charts, checking the "Double Layer" settings, and formatting the strict Health Ratings badges.

### ⚙️ Backend (Django)

- `ocr_backend/api/views.py`: The brain of the API.
  - `OCRView`: Accepts the image upload and sends it to the Datalab API to extract raw text.
  - `WebhookView` & `FetchParsedView`: Retrieves the finished raw text from Datalab and passes it to the Gemini AI models.
  - `parse_with_gemini`: Sends the raw text to LangGraph/Gemini to extract ingredients, allergens, and visualizations. Contains the "Double Layer" logic.
  - `calculate_health_indices`: A pure, strict algorithm (no AI involved) that analyzes extracted sugars, sodium, synthetic colors (E102/E110), artificial flavors, and BHA/preservatives to generate deterministic health scores out of 10 for Babies, Adults, and Seniors.
- `ocr_backend/api/consumers.py`: Handles Django Channels WebSockets. When the heavy OCR text extraction and AI parsing finish in the background, this consumer instantly pushes the JSON results down to the Flutter app via WebSockets so the user doesn't have to refresh.
- `ocr_backend/api/routing.py` & `ocr_backend/api/urls.py`: Maps URL paths to their respective views and WebSocket consumers.
- `ocr_backend/.env`: Stores critical API keys for Datalab (`DATALAB_API_KEY`) and Google Gemini (`GEMINI_API_KEY`). This is securely ignored by git via `.gitignore`.
