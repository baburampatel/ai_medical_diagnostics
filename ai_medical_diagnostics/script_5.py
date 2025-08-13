# Create README.md with basic instructions
readme_content = '''# AI Medical Diagnostics - Streamlit Application

**Educational disclaimer**: This project is intended **for educational and informational purposes only**. It does **not** replace professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical decisions. In emergencies, call local emergency services.

---

## Overview

This repository contains a minimal yet fully runnable MVP that replicates the idea behind *AI-Agents-for-Medical-Diagnostics* with:

* ✅ **Multi-document intake** (PDFs/images)
* ✅ **Specialist LLM agents** (cardiology, pulmonology, infectious disease, endocrinology)
* ✅ **Parallel async orchestration** with concurrency limit (default 2)
* ✅ **Structured JSON outputs** validated by Pydantic schemas and JSON Schema
* ✅ **Aggregation layer** that reconciles agents’ findings into an **actionable report**
* ✅ **Streamlit UI** for upload, progress, and interactive report + download (JSON / PDF)
* ✅ **Free ChatGPTAPIFree** provider adapter by default (no key required) with retries / schema repair
* ✅ Safe defaults: transient processing, no vision by default, clear red-flag warnings

![Architecture](./docs/architecture.png)

---

## Quick start

```bash
# 1) Clone & install
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r ai_medical_diagnostics/requirements.txt

# 2) Install system deps (Ubuntu example)
#    • Tesseract OCR
#    • Java (for Tabula)
#    • Ghostscript / Tk for Camelot
sudo apt update && sudo apt install -y tesseract-ocr default-jre ghostscript python3-tk

# 3) Run Streamlit UI
streamlit run ai_medical_diagnostics/ui/app.py

# 4) CLI example
python ai_medical_diagnostics/run.py \
  --inputs ./ai_medical_diagnostics/samples/case1 \
  --provider chatgpt_api_free \
  --enable-vision false
```

---

## Folder structure

```
ai_medical_diagnostics/
├── agents/              # Specialist agents (JSON-only outputs)
├── orchestrator/        # Router + aggregator
├── llm/                 # Provider abstraction (ChatGPTAPIFree, OpenAI)
├── io/                  # Ingestion, OCR, PDF utils
├── schemas/             # Pydantic + JSON Schema models
├── ui/                  # Streamlit frontend
├── utils/               # Retry, text helpers, etc.
├── config/              # YAML config & prompts
├── tests/               # Unit tests (schemas/parsers)
└── run.py               # Simple CLI wrapper
```

---

## Safety & limitations

1. **No clinical use** – results may be inaccurate / incomplete.
2. **Vision analysis OFF** by default (toggle in sidebar) – currently relies on text radiology reports.
3. **Transient processing** – files are kept in-memory only unless `store_files=true` in `features.yaml`.
4. **Concurrency limit** – ChatGPTAPIFree proxy is rate-limited; default is 2.
5. **Data privacy** – no documents leave your machine unless you point the provider to a remote endpoint.

---

## ChatGPTAPIFree details

* Endpoint: https://chatgpt-api.shn.hk/v1/chat/completions
* Request shape compatible with OpenAI API (no Auth header)
* Retries: exponential backoff with jitter (utils/retries.py)
* JSON mode: automatic repair & validation (utils/json_schemas.py)

---

## Feature flags

Edit `config/features.yaml`:

```yaml
enable_vision: false   # X-ray image path (stub)
store_files: false     # Persist uploads (off by default)
log_level: INFO        # DEBUG for verbose logs
```

---

## Roadmap

* 🔲 Fine-tuned image model for X-ray analysis behind explicit toggle
* 🔲 More specialties (nephrology, neurology)
* 🔲 SQLite opt-in storage for longitudinal cases

Pull requests welcome! 🎉
'''

with open("ai_medical_diagnostics/README.md", "w") as f:
    f.write(readme_content)

print("README created!")