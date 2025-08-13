# Create utility files

# Retry utilities
retries_util = '''"""Retry and backoff utilities."""
import asyncio
import random
import time
import logging
from functools import wraps
from typing import Callable, Any, Type, Union, Tuple

logger = logging.getLogger(__name__)

def retry_with_backoff(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    max_backoff: float = 60.0,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception
):
    """Decorator for retry with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries} retries: {str(e)}")
                        raise
                    
                    # Calculate backoff time with jitter
                    backoff_time = min(
                        backoff_factor ** attempt + random.uniform(0, 1),
                        max_backoff
                    )
                    
                    logger.warning(
                        f"Function {func.__name__} failed on attempt {attempt + 1}, "
                        f"retrying in {backoff_time:.2f} seconds: {str(e)}"
                    )
                    time.sleep(backoff_time)
            
        return wrapper
    return decorator

def async_retry_with_backoff(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    max_backoff: float = 60.0,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception
):
    """Decorator for async retry with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        logger.error(f"Async function {func.__name__} failed after {max_retries} retries: {str(e)}")
                        raise
                    
                    # Calculate backoff time with jitter
                    backoff_time = min(
                        backoff_factor ** attempt + random.uniform(0, 1),
                        max_backoff
                    )
                    
                    logger.warning(
                        f"Async function {func.__name__} failed on attempt {attempt + 1}, "
                        f"retrying in {backoff_time:.2f} seconds: {str(e)}"
                    )
                    await asyncio.sleep(backoff_time)
            
        return wrapper
    return decorator

class AsyncThrottle:
    """Async throttle to limit concurrent operations."""
    
    def __init__(self, max_concurrent: int = 2):
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def __aenter__(self):
        await self.semaphore.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.semaphore.release()

async def gather_with_concurrency(max_concurrent: int, *coroutines):
    """Execute coroutines with limited concurrency."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def sem_coro(coro):
        async with semaphore:
            return await coro
    
    return await asyncio.gather(*(sem_coro(c) for c in coroutines))
'''

# JSON schema utilities
json_utils = '''"""JSON schema and validation utilities."""
import json
import re
import logging
from typing import Dict, Any, Optional
from jsonschema import validate, ValidationError
import pydantic

logger = logging.getLogger(__name__)

def repair_json(json_str: str) -> str:
    """Attempt to repair malformed JSON."""
    if not json_str:
        return "{}"
    
    # Remove any text before the first {
    json_str = json_str.strip()
    start_idx = json_str.find('{')
    if start_idx > 0:
        json_str = json_str[start_idx:]
    
    # Remove any text after the last }
    end_idx = json_str.rfind('}')
    if end_idx > 0:
        json_str = json_str[:end_idx + 1]
    
    # Common JSON repairs
    repairs = [
        (r',\\s*}', '}'),  # Remove trailing commas
        (r',\\s*]', ']'),  # Remove trailing commas in arrays
        (r'\\n', ' '),      # Replace newlines with spaces
        (r'\\t', ' '),      # Replace tabs with spaces
        (r'\\s+', ' '),     # Normalize whitespace
    ]
    
    for pattern, replacement in repairs:
        json_str = re.sub(pattern, replacement, json_str)
    
    # Try to fix unclosed quotes
    quote_count = json_str.count('"')
    if quote_count % 2 != 0:
        json_str += '"'
    
    # Try to fix unclosed braces
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    if open_braces > close_braces:
        json_str += '}' * (open_braces - close_braces)
    
    return json_str

def validate_json_output(data: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> bool:
    """Validate JSON output against schema."""
    if schema is None:
        return True
    
    try:
        validate(instance=data, schema=schema)
        return True
    except ValidationError as e:
        logger.error(f"JSON validation error: {str(e)}")
        return False

def create_medical_output_schema() -> Dict[str, Any]:
    """Create JSON schema for medical agent output."""
    return {
        "type": "object",
        "properties": {
            "suspected_conditions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "rationale": {"type": "string"},
                        "evidence_citations": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "confidence_pct": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100
                        }
                    },
                    "required": ["name", "rationale", "confidence_pct"]
                }
            },
            "risks_and_red_flags": {
                "type": "array",
                "items": {"type": "string"}
            },
            "recommended_next_steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "reason": {"type": "string"},
                        "priority": {
                            "type": "string",
                            "enum": ["immediate", "soon", "discuss", "routine"]
                        },
                        "evidence_citations": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["action", "reason", "priority"]
                }
            },
            "data_gaps": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["suspected_conditions", "risks_and_red_flags", "recommended_next_steps", "data_gaps"]
    }

def safe_json_parse(json_str: str) -> Optional[Dict[str, Any]]:
    """Safely parse JSON with repair attempts."""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        logger.info("Initial JSON parse failed, attempting repair")
        try:
            repaired = repair_json(json_str)
            return json.loads(repaired)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON after repair: {json_str[:200]}...")
            return None
'''

# Text utilities
text_utils = '''"""Text processing and citation utilities."""
import re
import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

def extract_citations(text: str) -> List[str]:
    """Extract citation references from text."""
    # Pattern to match [doc_id], [doc_id:section], etc.
    citation_pattern = r'\\[([^\\]]+)\\]'
    matches = re.findall(citation_pattern, text)
    return list(set(matches))  # Remove duplicates

def highlight_text_snippets(text: str, keywords: List[str], context_chars: int = 100) -> List[Dict[str, Any]]:
    """Highlight and extract text snippets containing keywords."""
    snippets = []
    
    for keyword in keywords:
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        for match in pattern.finditer(text):
            start = max(0, match.start() - context_chars)
            end = min(len(text), match.end() + context_chars)
            
            snippet = {
                "keyword": keyword,
                "text": text[start:end],
                "start_pos": start,
                "end_pos": end,
                "match_start": match.start() - start,
                "match_end": match.end() - start
            }
            snippets.append(snippet)
    
    return snippets

def create_evidence_citations(
    findings: List[str], 
    doc_mapping: Dict[str, str]
) -> List[str]:
    """Create properly formatted evidence citations."""
    citations = []
    
    for finding in findings:
        # Extract any existing citations
        found_citations = extract_citations(finding)
        
        if found_citations:
            citations.extend(found_citations)
        else:
            # If no citations, add a generic document reference
            if doc_mapping:
                first_doc = list(doc_mapping.keys())[0]
                citations.append(first_doc)
    
    return list(set(citations))

def normalize_medical_terms(text: str) -> str:
    """Normalize common medical terms and abbreviations."""
    normalizations = {
        # Common lab abbreviations
        r'\\bhb\\b': 'hemoglobin',
        r'\\bhgb\\b': 'hemoglobin',
        r'\\bwbc\\b': 'white blood cells',
        r'\\brbc\\b': 'red blood cells',
        r'\\bplt\\b': 'platelets',
        r'\\bbun\\b': 'blood urea nitrogen',
        r'\\bcr\\b': 'creatinine',
        
        # Units
        r'\\bmg/dl\\b': 'mg/dL',
        r'\\bg/dl\\b': 'g/dL',
        r'\\bmmol/l\\b': 'mmol/L',
        
        # Common medications
        r'\\bace inhibitor\\b': 'ACE inhibitor',
        r'\\barb\\b': 'angiotensin receptor blocker',
        r'\\bnsaid\\b': 'NSAID',
        r'\\bppi\\b': 'proton pump inhibitor',
    }
    
    normalized_text = text
    for pattern, replacement in normalizations.items():
        normalized_text = re.sub(pattern, replacement, normalized_text, flags=re.IGNORECASE)
    
    return normalized_text

def extract_key_phrases(text: str, min_length: int = 3) -> List[str]:
    """Extract key medical phrases from text."""
    # Simple phrase extraction - could be enhanced with NLP
    sentences = re.split(r'[.!?]+', text)
    key_phrases = []
    
    medical_keywords = [
        'diagnosis', 'treatment', 'symptom', 'condition', 'disease',
        'abnormal', 'elevated', 'decreased', 'positive', 'negative',
        'chronic', 'acute', 'severe', 'mild', 'moderate'
    ]
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) >= min_length:
            # Check if sentence contains medical keywords
            if any(keyword in sentence.lower() for keyword in medical_keywords):
                key_phrases.append(sentence)
    
    return key_phrases

def clean_extracted_text(text: str) -> str:
    """Clean and normalize extracted text from OCR/PDF."""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\\s+', ' ', text)
    
    # Remove special characters that might interfere with processing
    text = re.sub(r'[\\x00-\\x08\\x0B\\x0C\\x0E-\\x1F\\x7F-\\x9F]', '', text)
    
    # Normalize line breaks
    text = re.sub(r'\\r\\n', '\\n', text)
    text = re.sub(r'\\r', '\\n', text)
    
    # Remove multiple consecutive newlines
    text = re.sub(r'\\n{3,}', '\\n\\n', text)
    
    return text.strip()

def split_text_by_sections(text: str) -> Dict[str, str]:
    """Split medical document text into logical sections."""
    sections = {}
    
    # Common medical document section headers
    section_patterns = {
        'chief_complaint': r'(chief complaint|cc):\\s*',
        'history': r'(history of present illness|hpi|history):\\s*',
        'medications': r'(medications?|drugs?|prescriptions?):\\s*',
        'allergies': r'(allergies|adverse reactions):\\s*',
        'labs': r'(laboratory|lab results?|lab values?):\\s*',
        'vitals': r'(vital signs?|vitals?):\\s*',
        'physical_exam': r'(physical exam|examination|pe):\\s*',
        'assessment': r'(assessment|impression):\\s*',
        'plan': r'(plan|treatment):\\s*'
    }
    
    current_section = 'other'
    current_text = []
    
    lines = text.split('\\n')
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Check if line matches a section header
        section_found = False
        for section_name, pattern in section_patterns.items():
            if re.match(pattern, line_lower):
                # Save previous section
                if current_text:
                    sections[current_section] = '\\n'.join(current_text).strip()
                
                # Start new section
                current_section = section_name
                current_text = []
                section_found = True
                break
        
        if not section_found:
            current_text.append(line)
    
    # Save last section
    if current_text:
        sections[current_section] = '\\n'.join(current_text).strip()
    
    return sections
'''

# PDF utilities
pdf_utils = '''"""PDF processing utilities."""
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False
    logging.warning("Camelot not available for PDF table extraction")

try:
    import tabula
    TABULA_AVAILABLE = True
except ImportError:
    TABULA_AVAILABLE = False
    logging.warning("Tabula not available for PDF table extraction")

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    try:
        from PyPDF2 import PdfReader
        PYPDF_AVAILABLE = True
    except ImportError:
        PYPDF_AVAILABLE = False
        logging.warning("No PDF reader available")

logger = logging.getLogger(__name__)

class PDFProcessor:
    """PDF processing utility class."""
    
    def __init__(self):
        self.supported_extractors = []
        if CAMELOT_AVAILABLE:
            self.supported_extractors.append("camelot")
        if TABULA_AVAILABLE:
            self.supported_extractors.append("tabula")
    
    def extract_text(self, pdf_path: str) -> str:
        """Extract text from PDF."""
        if not PYPDF_AVAILABLE:
            raise RuntimeError("No PDF reader available")
        
        try:
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\\n"
                return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def extract_tables_camelot(self, pdf_path: str) -> List[pd.DataFrame]:
        """Extract tables using Camelot."""
        if not CAMELOT_AVAILABLE:
            return []
        
        try:
            tables = camelot.read_pdf(pdf_path, pages='all')
            return [table.df for table in tables]
        except Exception as e:
            logger.error(f"Error extracting tables with Camelot: {str(e)}")
            return []
    
    def extract_tables_tabula(self, pdf_path: str) -> List[pd.DataFrame]:
        """Extract tables using Tabula."""
        if not TABULA_AVAILABLE:
            return []
        
        try:
            tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
            return tables if isinstance(tables, list) else [tables]
        except Exception as e:
            logger.error(f"Error extracting tables with Tabula: {str(e)}")
            return []
    
    def extract_tables(self, pdf_path: str, method: str = "auto") -> List[pd.DataFrame]:
        """Extract tables from PDF using specified method."""
        if method == "auto":
            # Try Camelot first, fallback to Tabula
            if CAMELOT_AVAILABLE:
                tables = self.extract_tables_camelot(pdf_path)
                if tables:
                    return tables
            
            if TABULA_AVAILABLE:
                return self.extract_tables_tabula(pdf_path)
            
            return []
        
        elif method == "camelot":
            return self.extract_tables_camelot(pdf_path)
        elif method == "tabula":
            return self.extract_tables_tabula(pdf_path)
        else:
            raise ValueError(f"Unknown table extraction method: {method}")
    
    def classify_tables(self, tables: List[pd.DataFrame]) -> Dict[str, List[pd.DataFrame]]:
        """Classify tables by likely content type."""
        classified = {
            "lab_results": [],
            "medications": [],
            "billing": [],
            "other": []
        }
        
        for table in tables:
            if table.empty:
                continue
            
            # Convert to string for analysis
            table_text = table.to_string().lower()
            
            # Lab results indicators
            lab_indicators = [
                "test", "result", "value", "reference", "range", "normal", "abnormal",
                "hemoglobin", "glucose", "cholesterol", "creatinine", "bun",
                "sodium", "potassium", "chloride", "co2"
            ]
            
            # Medication indicators
            med_indicators = [
                "medication", "drug", "prescription", "dose", "dosage", "frequency",
                "mg", "mcg", "ml", "tablet", "capsule", "daily", "twice"
            ]
            
            # Billing indicators
            billing_indicators = [
                "charge", "cost", "price", "cpt", "icd", "procedure", "billing",
                "insurance", "copay", "deductible", "$"
            ]
            
            # Count indicators
            lab_count = sum(1 for indicator in lab_indicators if indicator in table_text)
            med_count = sum(1 for indicator in med_indicators if indicator in table_text)
            billing_count = sum(1 for indicator in billing_indicators if indicator in table_text)
            
            # Classify based on highest count
            max_count = max(lab_count, med_count, billing_count)
            
            if max_count == 0:
                classified["other"].append(table)
            elif lab_count == max_count:
                classified["lab_results"].append(table)
            elif med_count == max_count:
                classified["medications"].append(table)
            else:
                classified["billing"].append(table)
        
        return classified
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Comprehensive PDF processing."""
        result = {
            "text": "",
            "tables": [],
            "classified_tables": {},
            "metadata": {
                "num_pages": 0,
                "file_size": 0,
                "processing_errors": []
            }
        }
        
        try:
            # Get file metadata
            pdf_file = Path(pdf_path)
            result["metadata"]["file_size"] = pdf_file.stat().st_size
            
            # Extract text
            text = self.extract_text(pdf_path)
            result["text"] = text
            
            # Count pages
            if PYPDF_AVAILABLE:
                try:
                    with open(pdf_path, 'rb') as file:
                        reader = PdfReader(file)
                        result["metadata"]["num_pages"] = len(reader.pages)
                except Exception as e:
                    result["metadata"]["processing_errors"].append(f"Page count error: {str(e)}")
            
            # Extract tables
            tables = self.extract_tables(pdf_path)
            result["tables"] = tables
            
            # Classify tables
            if tables:
                result["classified_tables"] = self.classify_tables(tables)
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            result["metadata"]["processing_errors"].append(str(e))
        
        return result

# Create global PDF processor instance
pdf_processor = PDFProcessor()
'''

# Write utility files
with open("ai_medical_diagnostics/utils/retries.py", "w") as f:
    f.write(retries_util)

with open("ai_medical_diagnostics/utils/json_schemas.py", "w") as f:
    f.write(json_utils)

with open("ai_medical_diagnostics/utils/text_utils.py", "w") as f:
    f.write(text_utils)

with open("ai_medical_diagnostics/utils/pdf_utils.py", "w") as f:
    f.write(pdf_utils)

print("Utility files created successfully!")