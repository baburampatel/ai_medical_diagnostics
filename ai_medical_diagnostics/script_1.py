# Generate requirements.txt
requirements = """streamlit>=1.28.0
pydantic>=2.0.0
httpx>=0.24.0
requests>=2.31.0
pytesseract>=0.3.10
pillow>=9.5.0
pypdf>=3.15.0
unstructured[pdf]>=0.10.0
camelot-py[cv]>=0.11.0
tabula-py>=2.8.0
pandas>=2.0.0
numpy>=1.24.0
python-dotenv>=1.0.0
aiofiles>=23.0.0
asyncio-throttle>=1.0.2
tenacity>=8.2.0
paddleocr>=2.7.0
python-multipart>=0.0.6
openpyxl>=3.1.0
reportlab>=4.0.0
"""

with open("ai_medical_diagnostics/requirements.txt", "w") as f:
    f.write(requirements.strip())

print("Requirements.txt created successfully!")

# Generate config files
models_yaml = """# LLM Provider Configuration
default_provider: "chatgpt_api_free"
providers:
  chatgpt_api_free:
    endpoint: "https://chatgpt-api.shn.hk/v1/chat/completions"
    model: "gpt-3.5-turbo"
    temperature: 0.2
    max_tokens: 2000
    timeout: 30
    max_retries: 3
    backoff_factor: 2
    
  openai:
    endpoint: "https://api.openai.com/v1/chat/completions"
    model: "gpt-3.5-turbo"
    temperature: 0.2
    max_tokens: 2000
    timeout: 30
    max_retries: 3
    backoff_factor: 2

# Concurrency settings
max_concurrency: 2
request_timeout: 30

# Agent timeout settings
agent_timeout: 45
"""

features_yaml = """# Feature flags
enable_vision: false
store_files: false
enable_database: false
enable_debugging: true

# Supported languages
languages:
  - "en"

# File processing settings
max_file_size_mb: 50
supported_formats:
  - "pdf"
  - "png"
  - "jpg" 
  - "jpeg"

# OCR settings
ocr_engines:
  - "pytesseract"
  - "paddleocr"
  
# PDF processing
pdf_table_extraction: true
pdf_text_extraction: true

# Logging
log_level: "INFO"
log_file: "medical_diagnostics.log"
"""

with open("ai_medical_diagnostics/config/models.yaml", "w") as f:
    f.write(models_yaml)

with open("ai_medical_diagnostics/config/features.yaml", "w") as f:
    f.write(features_yaml)

print("Configuration files created successfully!")