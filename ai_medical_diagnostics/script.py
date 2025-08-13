import os
import json
from pathlib import Path

# Create the complete project structure
project_structure = {
    "ai_medical_diagnostics": {
        "__init__.py": "",
        "run.py": "",
        "ui": {
            "__init__.py": "",
            "app.py": "",
        },
        "agents": {
            "__init__.py": "",
            "base.py": "",
            "cardiology.py": "",
            "pulmonology.py": "",
            "infectious_disease.py": "",
            "endocrinology.py": "",
        },
        "orchestrator": {
            "__init__.py": "",
            "router.py": "",
            "aggregator.py": "",
        },
        "llm": {
            "__init__.py": "",
            "providers": {
                "__init__.py": "",
                "base.py": "",
                "chatgpt_api_free.py": "",
                "openai.py": "",
            }
        },
        "io": {
            "__init__.py": "",
            "ingest.py": "",
            "parsers.py": "",
            "vision.py": "",
        },
        "schemas": {
            "__init__.py": "",
            "case_bundle.py": "",
            "lab.py": "",
            "meds.py": "",
            "radiology.py": "",
            "agent_output.py": "",
            "aggregate.py": "",
        },
        "config": {
            "models.yaml": "",
            "features.yaml": "",
            "prompts": {
                "cardiology.txt": "",
                "pulmonology.txt": "",
                "infectious_disease.txt": "",
                "endocrinology.txt": "",
            }
        },
        "utils": {
            "__init__.py": "",
            "json_schemas.py": "",
            "text_utils.py": "",
            "retries.py": "",
            "pdf_utils.py": "",
        },
        "samples": {
            "case1": {},
            "example_outputs": {},
        },
        "tests": {
            "__init__.py": "",
            "test_parsers.py": "",
            "test_schemas.py": "",
        },
        "requirements.txt": "",
        "README.md": "",
    }
}

# Function to create directory structure
def create_structure(base_path, structure):
    for name, content in structure.items():
        current_path = base_path / name
        if isinstance(content, dict):
            current_path.mkdir(exist_ok=True)
            create_structure(current_path, content)
        else:
            current_path.parent.mkdir(parents=True, exist_ok=True)
            current_path.write_text(content, encoding='utf-8')

# Create the project structure
base_path = Path(".")
create_structure(base_path, project_structure)

print("Project structure created successfully!")
print("\nProject tree:")
for root, dirs, files in os.walk("ai_medical_diagnostics"):
    level = root.replace("ai_medical_diagnostics", "").count(os.sep)
    indent = " " * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = " " * 2 * (level + 1)
    for file in files:
        print(f"{subindent}{file}")