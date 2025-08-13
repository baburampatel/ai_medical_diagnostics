# Create core schema files

# Lab schema
lab_schema = '''"""Pydantic models for laboratory test results."""
from typing import Optional, Union, List
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum

class LabFlag(str, Enum):
    """Laboratory result flag."""
    LOW = "low"
    HIGH = "high"
    NORMAL = "normal"
    CRITICAL = "critical"
    ABNORMAL = "abnormal"

class LabResult(BaseModel):
    """Model for individual laboratory test result."""
    test_name: str = Field(..., description="Name of the laboratory test")
    value: Union[float, str] = Field(..., description="Test result value")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    reference_range: Optional[str] = Field(None, description="Normal reference range")
    flag: Optional[LabFlag] = Field(None, description="Result flag (low/high/normal/critical)")
    collected_at: Optional[datetime] = Field(None, description="Sample collection timestamp")
    doc_id: Optional[str] = Field(None, description="Source document ID")
    
    @validator('test_name')
    def normalize_test_name(cls, v):
        """Normalize test names to common aliases."""
        aliases = {
            "hb": "hemoglobin",
            "hgb": "hemoglobin", 
            "wbc": "white_blood_cells",
            "rbc": "red_blood_cells",
            "plt": "platelets",
            "glucose": "blood_glucose",
            "bun": "blood_urea_nitrogen",
            "cr": "creatinine",
            "na": "sodium",
            "k": "potassium",
            "cl": "chloride"
        }
        return aliases.get(v.lower(), v)

class LabPanel(BaseModel):
    """Collection of laboratory results."""
    panel_name: str = Field(..., description="Name of the lab panel")
    results: List[LabResult] = Field(default_factory=list)
    ordered_by: Optional[str] = Field(None, description="Ordering physician")
    lab_name: Optional[str] = Field(None, description="Laboratory name")
    doc_id: Optional[str] = Field(None, description="Source document ID")
'''

# Medications schema  
meds_schema = '''"""Pydantic models for medication information."""
from typing import Optional, List
from datetime import datetime, date
from pydantic import BaseModel, Field, validator
from enum import Enum

class MedicationRoute(str, Enum):
    """Route of medication administration."""
    ORAL = "oral"
    IV = "intravenous"
    IM = "intramuscular"
    TOPICAL = "topical"
    INHALED = "inhaled"
    SUBLINGUAL = "sublingual"
    RECTAL = "rectal"
    NASAL = "nasal"

class MedicationFrequency(str, Enum):
    """Medication frequency."""
    ONCE_DAILY = "once_daily"
    TWICE_DAILY = "twice_daily" 
    THREE_TIMES_DAILY = "three_times_daily"
    FOUR_TIMES_DAILY = "four_times_daily"
    AS_NEEDED = "as_needed"
    WEEKLY = "weekly"
    MONTHLY = "monthly"

class Medication(BaseModel):
    """Model for medication information."""
    drug_name: str = Field(..., description="Medication name")
    generic_name: Optional[str] = Field(None, description="Generic drug name")
    strength: Optional[str] = Field(None, description="Medication strength")
    dose: Optional[str] = Field(None, description="Prescribed dose")
    frequency: Optional[MedicationFrequency] = Field(None, description="Frequency of administration")
    route: Optional[MedicationRoute] = Field(None, description="Route of administration")
    duration: Optional[str] = Field(None, description="Treatment duration")
    start_date: Optional[date] = Field(None, description="Start date")
    end_date: Optional[date] = Field(None, description="End date")
    prescriber: Optional[str] = Field(None, description="Prescribing physician")
    indication: Optional[str] = Field(None, description="Reason for prescription")
    doc_id: Optional[str] = Field(None, description="Source document ID")
    
    @validator('drug_name')
    def normalize_drug_name(cls, v):
        """Normalize drug names."""
        # Common drug name normalizations
        return v.strip().title()

class Prescription(BaseModel):
    """Collection of prescribed medications."""
    prescription_id: Optional[str] = Field(None, description="Prescription ID")
    medications: List[Medication] = Field(default_factory=list)
    prescriber: Optional[str] = Field(None, description="Prescribing physician")
    issue_date: Optional[date] = Field(None, description="Prescription date")
    pharmacy: Optional[str] = Field(None, description="Dispensing pharmacy")
    doc_id: Optional[str] = Field(None, description="Source document ID")
'''

# Radiology schema
radiology_schema = '''"""Pydantic models for radiology reports."""
from typing import Optional, List
from datetime import datetime, date
from pydantic import BaseModel, Field
from enum import Enum

class RadiologyModality(str, Enum):
    """Imaging modality."""
    XRAY = "x-ray"
    CT = "ct_scan"
    MRI = "mri"
    ULTRASOUND = "ultrasound"
    MAMMOGRAPHY = "mammography"
    NUCLEAR = "nuclear_medicine"
    PET = "pet_scan"

class BodyRegion(str, Enum):
    """Body regions for imaging."""
    CHEST = "chest"
    ABDOMEN = "abdomen"
    PELVIS = "pelvis"
    HEAD = "head"
    SPINE = "spine"
    EXTREMITIES = "extremities"
    CARDIAC = "cardiac"

class RadiologyFinding(BaseModel):
    """Individual radiology finding."""
    finding: str = Field(..., description="Description of the finding")
    body_part: Optional[BodyRegion] = Field(None, description="Body part/region")
    laterality: Optional[str] = Field(None, description="Left/right/bilateral")
    severity: Optional[str] = Field(None, description="Severity assessment")

class RadiologyReport(BaseModel):
    """Model for radiology report."""
    study_type: str = Field(..., description="Type of imaging study")
    modality: Optional[RadiologyModality] = Field(None, description="Imaging modality")
    body_part: Optional[BodyRegion] = Field(None, description="Primary body region")
    study_date: Optional[date] = Field(None, description="Study date")
    findings: List[RadiologyFinding] = Field(default_factory=list, description="Imaging findings")
    impression: Optional[str] = Field(None, description="Radiologist impression")
    radiologist: Optional[str] = Field(None, description="Interpreting radiologist")
    technique: Optional[str] = Field(None, description="Imaging technique")
    clinical_history: Optional[str] = Field(None, description="Clinical indication")
    doc_id: Optional[str] = Field(None, description="Source document ID")
'''

# Case bundle schema
case_bundle_schema = '''"""Pydantic models for medical case data bundle."""
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from pydantic import BaseModel, Field

from .lab import LabPanel, LabResult
from .meds import Prescription, Medication
from .radiology import RadiologyReport

class Demographics(BaseModel):
    """Patient demographic information."""
    age: Optional[int] = Field(None, description="Patient age")
    gender: Optional[str] = Field(None, description="Patient gender")
    medical_record_number: Optional[str] = Field(None, description="MRN")

class ClinicalNote(BaseModel):
    """Clinical note or free text."""
    note_type: str = Field(..., description="Type of clinical note")
    content: str = Field(..., description="Note content")
    author: Optional[str] = Field(None, description="Note author")
    date: Optional[datetime] = Field(None, description="Note date")
    doc_id: Optional[str] = Field(None, description="Source document ID")

class DocumentMetadata(BaseModel):
    """Metadata for source documents."""
    doc_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="File type/format")
    upload_timestamp: datetime = Field(default_factory=datetime.now)
    file_size: Optional[int] = Field(None, description="File size in bytes")
    checksum: Optional[str] = Field(None, description="File checksum")

class CaseBundle(BaseModel):
    """Complete medical case data bundle."""
    case_id: str = Field(..., description="Unique case identifier")
    demographics: Optional[Demographics] = Field(None, description="Patient demographics")
    lab_results: List[LabResult] = Field(default_factory=list, description="Laboratory results")
    lab_panels: List[LabPanel] = Field(default_factory=list, description="Laboratory panels")
    medications: List[Medication] = Field(default_factory=list, description="Medications")
    prescriptions: List[Prescription] = Field(default_factory=list, description="Prescriptions")
    radiology_reports: List[RadiologyReport] = Field(default_factory=list, description="Radiology reports")
    clinical_notes: List[ClinicalNote] = Field(default_factory=list, description="Clinical notes")
    documents: List[DocumentMetadata] = Field(default_factory=list, description="Source documents")
    symptoms: List[str] = Field(default_factory=list, description="Reported symptoms")
    chief_complaint: Optional[str] = Field(None, description="Chief complaint")
    created_at: datetime = Field(default_factory=datetime.now)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the case bundle."""
        return {
            "case_id": self.case_id,
            "num_lab_results": len(self.lab_results),
            "num_medications": len(self.medications),
            "num_radiology_reports": len(self.radiology_reports),
            "num_clinical_notes": len(self.clinical_notes),
            "num_documents": len(self.documents),
            "has_demographics": self.demographics is not None,
            "chief_complaint": self.chief_complaint
        }
'''

# Agent output schema
agent_output_schema = '''"""Pydantic models for agent output."""
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum

class Priority(str, Enum):
    """Priority levels for recommendations."""
    IMMEDIATE = "immediate"
    SOON = "soon"
    DISCUSS = "discuss"
    ROUTINE = "routine"

class ConditionConfidence(str, Enum):
    """Confidence levels for suspected conditions."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class SuspectedCondition(BaseModel):
    """Suspected medical condition."""
    name: str = Field(..., description="Condition name")
    rationale: str = Field(..., description="Clinical reasoning")
    evidence_citations: List[str] = Field(default_factory=list, description="Supporting evidence citations")
    confidence: ConditionConfidence = Field(..., description="Confidence level")
    confidence_pct: int = Field(..., ge=0, le=100, description="Confidence percentage")
    
    @validator('confidence_pct')
    def validate_confidence_pct(cls, v, values):
        """Validate confidence percentage matches confidence level."""
        confidence = values.get('confidence')
        if confidence == ConditionConfidence.HIGH and v < 70:
            raise ValueError("High confidence should be >= 70%")
        elif confidence == ConditionConfidence.MEDIUM and (v < 40 or v >= 70):
            raise ValueError("Medium confidence should be 40-69%")
        elif confidence == ConditionConfidence.LOW and v >= 40:
            raise ValueError("Low confidence should be < 40%")
        return v

class Recommendation(BaseModel):
    """Medical recommendation."""
    action: str = Field(..., description="Recommended action")
    reason: str = Field(..., description="Rationale for recommendation")
    priority: Priority = Field(..., description="Priority level")
    evidence_citations: List[str] = Field(default_factory=list, description="Supporting evidence")

class AgentOutput(BaseModel):
    """Output from a medical specialist agent."""
    agent_name: str = Field(..., description="Name of the specialist agent")
    specialty: str = Field(..., description="Medical specialty")
    suspected_conditions: List[SuspectedCondition] = Field(default_factory=list)
    risks_and_red_flags: List[str] = Field(default_factory=list, description="Identified risks and red flags")
    recommended_next_steps: List[Recommendation] = Field(default_factory=list)
    data_gaps: List[str] = Field(default_factory=list, description="Missing data that would help diagnosis")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence in assessment")
    processing_time: float = Field(..., description="Time taken to process (seconds)")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
'''

# Aggregate schema
aggregate_schema = '''"""Pydantic models for aggregated medical analysis."""
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

from .agent_output import SuspectedCondition, Recommendation, Priority
from .lab import LabResult
from .meds import Medication

class ConsolidatedCondition(BaseModel):
    """Consolidated condition from multiple agents."""
    name: str = Field(..., description="Condition name")
    supporting_agents: List[str] = Field(..., description="Agents that identified this condition")
    consensus_confidence: float = Field(..., ge=0.0, le=1.0, description="Consensus confidence")
    rationale: str = Field(..., description="Consolidated rationale")
    evidence_citations: List[str] = Field(default_factory=list)

class AbnormalLab(BaseModel):
    """Abnormal laboratory result."""
    test_name: str
    value: str
    reference_range: Optional[str]
    flag: str
    clinical_significance: Optional[str] = Field(None, description="Clinical interpretation")

class MedicationNote(BaseModel):
    """Medication-related note or concern."""
    medication: str
    note_type: str = Field(..., description="Type: interaction, dose_concern, etc.")
    description: str
    severity: str = Field(..., description="low, medium, high")

class ActionPlan(BaseModel):
    """Categorized action plan."""
    immediate: List[Recommendation] = Field(default_factory=list)
    short_term: List[Recommendation] = Field(default_factory=list) 
    discuss_with_doctor: List[Recommendation] = Field(default_factory=list)

class AggregateReport(BaseModel):
    """Final aggregated medical analysis report."""
    case_id: str = Field(..., description="Case identifier")
    executive_summary: List[str] = Field(..., description="Key findings summary (3-6 bullets)")
    conditions_ranked: List[ConsolidatedCondition] = Field(default_factory=list)
    abnormal_labs: List[AbnormalLab] = Field(default_factory=list)
    medication_notes: List[MedicationNote] = Field(default_factory=list)
    red_flag_warnings: List[str] = Field(default_factory=list)
    action_plan: ActionPlan = Field(default_factory=ActionPlan)
    data_quality_notes: List[str] = Field(default_factory=list, description="Data quality issues")
    participating_agents: List[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.now)
    processing_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def get_priority_actions(self) -> List[Recommendation]:
        """Get all immediate and high-priority actions."""
        priority_actions = self.action_plan.immediate.copy()
        for action in self.action_plan.short_term:
            if action.priority == Priority.IMMEDIATE:
                priority_actions.append(action)
        return priority_actions
'''

# Write all schema files
with open("ai_medical_diagnostics/schemas/lab.py", "w") as f:
    f.write(lab_schema)

with open("ai_medical_diagnostics/schemas/meds.py", "w") as f:
    f.write(meds_schema)

with open("ai_medical_diagnostics/schemas/radiology.py", "w") as f:
    f.write(radiology_schema)

with open("ai_medical_diagnostics/schemas/case_bundle.py", "w") as f:
    f.write(case_bundle_schema)

with open("ai_medical_diagnostics/schemas/agent_output.py", "w") as f:
    f.write(agent_output_schema)

with open("ai_medical_diagnostics/schemas/aggregate.py", "w") as f:
    f.write(aggregate_schema)

print("Schema files created successfully!")