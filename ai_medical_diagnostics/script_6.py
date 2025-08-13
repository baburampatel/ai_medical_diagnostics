# Create run.py CLI skeleton
run_py = '''"""CLI entry point for batch processing."""
import argparse
import asyncio
from pathlib import Path
import logging
from typing import List

from ai_medical_diagnostics.orchestrator.router import RouteManager
from ai_medical_diagnostics.orchestrator.aggregator import AggregateManager
from ai_medical_diagnostics.io.ingest import IngestManager
from ai_medical_diagnostics.llm.providers import get_provider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_cli")

async def process_case(inputs: List[Path], provider_name: str, enable_vision: bool):
    """Process a single medical case folder."""
    ingest_manager = IngestManager(enable_vision=enable_vision)
    route_manager = RouteManager()
    aggregate_manager = AggregateManager()
    
    case_bundle = await ingest_manager.ingest_files(inputs)
    logger.info(f"Loaded case bundle {case_bundle.case_id}")
    
    # Initialize LLM provider
    provider = get_provider(provider_name)
    async with provider as llm:
        agent_outputs = await route_manager.run_all_agents(case_bundle, llm)
        report = aggregate_manager.aggregate_outputs(case_bundle, agent_outputs)
        
        # Print executive summary
        print("\nEXECUTIVE SUMMARY:\n")
        for bullet in report.executive_summary:
            print(f"• {bullet}")
        
        print("\nTop conditions:")
        for cond in report.conditions_ranked[:5]:
            print(f"- {cond.name} (confidence {cond.consensus_confidence:.0%})")

        # Optionally save JSON
        out_path = Path.cwd() / f"{case_bundle.case_id}_aggregate.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(report.model_dump_json(indent=2))
        logger.info(f"Saved aggregate report to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Medical Diagnostics CLI")
    parser.add_argument("--inputs", type=str, required=True, help="Path to folder with case files (PDFs/images)")
    parser.add_argument("--provider", type=str, default="chatgpt_api_free", help="LLM provider to use")
    parser.add_argument("--enable-vision", action="store_true", help="Enable vision path (default OFF)")

    args = parser.parse_args()
    input_path = Path(args.inputs)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path {input_path} does not exist")
    
    # Gather all file paths
    file_paths = [p for p in input_path.rglob('*') if p.is_file()]
    
    asyncio.run(process_case(file_paths, args.provider, args.enable_vision))
'''

with open("ai_medical_diagnostics/run.py", "w") as f:
    f.write(run_py)

print("run.py CLI written!")