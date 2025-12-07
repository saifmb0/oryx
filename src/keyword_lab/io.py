import json
import logging
import os
from pathlib import Path
from typing import List, Dict
import pandas as pd

# Try to import openpyxl for Excel support
try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False


def write_json(items: List[Dict], output_path: str) -> None:
    if output_path == "-":
        # print formatted JSON array to stdout only
        print(json.dumps(items, ensure_ascii=False, indent=2))
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
        f.write("\n")
    logging.info(f"Saved JSON to {path}")


def write_csv(items: List[Dict], csv_path: str) -> None:
    if not csv_path:
        return
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(items)
    df.to_csv(path, index=False)
    logging.info(f"Saved CSV to {path}")


def write_excel(items: List[Dict], xlsx_path: str) -> None:
    """
    Write results to an Excel file with multiple sheets.
    
    Creates two sheets:
    - 'Keywords': Raw data with all keywords
    - 'Cluster Summary': Pivot-style summary by cluster
    
    Requires openpyxl: pip install keyword-lab[excel]
    
    Args:
        items: List of keyword result dictionaries
        xlsx_path: Path to output .xlsx file
    """
    if not xlsx_path:
        return
    
    if not HAS_OPENPYXL:
        logging.warning(
            "Excel export requires openpyxl. Install with: pip install keyword-lab[excel]"
        )
        return
    
    path = Path(xlsx_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(items)
    
    # Create cluster summary
    if not df.empty and 'cluster' in df.columns:
        summary = df.groupby('cluster').agg({
            'keyword': 'count',
            'search_volume': 'mean',
            'difficulty': 'mean',
            'opportunity_score': 'mean',
        }).round(3)
        summary.columns = ['keyword_count', 'avg_volume', 'avg_difficulty', 'avg_opportunity']
        summary = summary.sort_values('avg_opportunity', ascending=False)
    else:
        summary = pd.DataFrame()
    
    # Write to Excel with multiple sheets
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Keywords', index=False)
        if not summary.empty:
            summary.to_excel(writer, sheet_name='Cluster Summary')
    
    logging.info(f"Saved Excel to {path}")


def write_output(items: List[Dict], output_path: str, save_csv: str = None) -> None:
    """
    Write output in the appropriate format based on file extension.
    
    Supports: .json, .csv, .xlsx
    
    Args:
        items: List of keyword result dictionaries
        output_path: Path to output file (or '-' for stdout)
        save_csv: Optional additional CSV path
    """
    if output_path == "-":
        write_json(items, output_path)
        return
    
    path = Path(output_path)
    ext = path.suffix.lower()
    
    if ext == ".xlsx":
        write_excel(items, output_path)
    elif ext == ".csv":
        write_csv(items, output_path)
    else:
        # Default to JSON
        write_json(items, output_path)
    
    # Also write CSV if requested
    if save_csv:
        write_csv(items, save_csv)
