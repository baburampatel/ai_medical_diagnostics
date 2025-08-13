# AI Agents for Medical Diagnostics (Educational MVP)

⚠ **DISCLAIMER**  
This project is for **educational and informational purposes only**.  
It is **NOT** a substitute for professional medical advice, diagnosis, or treatment.  
Always consult a qualified healthcare provider.

---

## **Overview**
This app allows you to upload multiple medical documents — such as lab reports, prescriptions, radiology text reports, and medical bills — and have multiple specialist AI agents analyze them in parallel.  

It uses the **ChatGPTAPIFree** provider by default (**no API key required**), processes files locally without storing them (by default), and produces a consolidated medical insight report.

---

## **Features**
- Upload multiple PDFs/images (`.pdf`, `.png`, `.jpg`, `.jpeg`)
- Automatic **classification**, **OCR text extraction**, and **table parsing**
- Multiple specialist medical agents:
  - Cardiology
  - Pulmonology
  - Infectious Disease
  - Endocrinology
- Aggregated output with:
  - Suspected conditions
  - Red-flag warnings
  - Recommended next steps
  - Evidence citations
- **Streamlit UI** and **CLI mode**
- **No paid API** required

---

## **Installation & Setup**

### 1. Prerequisites
- **Python 3.10+**
- **Tesseract OCR** installed:
  - **Ubuntu/Debian**: `sudo apt install tesseract-ocr`
  - **Mac** (Homebrew): `brew install tesseract`
  - **Windows**: [Download here](https://github.com/UB-Mannheim/tesseract/wiki) and add to PATH.

If Tesseract is in a custom location, set:



export TESSERACT_CMD=/path/to/tesseract
### 2. Clone & Install

git clone <your-repo-url>.git
cd ai_medical_diagnostics
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt


---

## **Quick Start — Streamlit Web UI**


streamlit run ui/app.py
1. Open browser at the link shown in terminal.
2. Upload your medical files (PDFs & images).
3. Adjust provider/model in the sidebar (default is ChatGPTAPIFree).
4. Click **Process** to run AI agents.
5. View interactive results & download as JSON/PDF.

---

## **Quick Start — CLI Mode**

python run.py --inputs ./samples/case1 --provider chatgpt_api_free --enable-vision false
- Processes all files in `./samples/case1`.
- Saves structured JSON output in `outputs/`.

---

## **Configuration**
Configs in `config/`:
- `models.yaml` — LLM provider, model, temperature, concurrency.
- `features.yaml` — Feature toggles (`enable_vision`, `store_files`, etc.).

Override via CLI flags, for example:

python run.py --max-concurrency 1 --temperature 0.5

---

## **File Structure**

---

## **Example Output (JSON snippet)**
{
"executive_summary": [
"Mild anemia and elevated WBC detected.",
"Possible chest infection; immediate review advised."
],
"conditions_ranked": [
{ "name": "Chest Infection", "confidence_pct": 85 },
{ "name": "Mild Anemia", "confidence_pct": 78 }
],
"red_flag_warnings": [
"Respiratory distress symptoms present in clinical notes."
],
"action_plan": {
"immediate": ["Consult pulmonologist"],
"soon": ["Repeat CBC in 1 week"],
"discuss": ["Dietary review for low iron"]
}
}

---

## **Screenshots**
*(Replace with your own when running locally)*  
**Streamlit Dashboard Example:**  
![UI Screenshot Placeholder](docs/images/ui_screenshot.png)  

**Aggregated Report Example:**  
![Report Screenshot Placeholder](docs/images/report_screenshot.png)  

---

## **Default API Provider: ChatGPTAPIFree**
- Endpoint: `https://chatgpt-api.shn.hk/v1/chat/completions`
- No API key required
- Compatible with OpenAI payload format

---

## **Limitations**
- Educational use only
- X-ray vision mode disabled by default
- Limited concurrency (`max_concurrency = 2`)

---

## **License**
MIT License — see `LICENSE` file.


