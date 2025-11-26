from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import os
from dotenv import load_dotenv
import json
import io
import csv
from typing import Optional, List

# Try to import openpyxl for Excel generation
try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
except ImportError:
    Workbook = None

# Load env vars (reuses the same environment variables as bug_classifier)
load_dotenv()

# Import classifier (should be import-safe now)
try:
    # batch_classify is the new async batch classifier; keep classify_bug for single-item fallback
    from bug_classifier import batch_classify, classify_bug
except ImportError:
    # fallback to relative import if running as package
    try:
        from .bug_classifier import batch_classify, classify_bug
    except ImportError as e:
        raise RuntimeError(f"Failed to import bug_classifier: {e}")

# Try to import pandas and openpyxl for Excel support
try:
    import pandas as pd
except ImportError:
    pd = None

app = FastAPI(title="BugClassifier API", version="0.1")

# Store latest classification results in memory for download (session-based)
latest_results = None

# Allow the Web frontend (served from localhost:5173) to call this API during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

class ClassifyRequest(BaseModel):
    text: str


class ClassifyItem(BaseModel):
    text: str
    label: str
    raw: str
    team: str | None = None
    severity: str | None = None


class ClassifyResponse(BaseModel):
    results: list[ClassifyItem]


class UploadResponse(BaseModel):
    filename: str
    total_rows: int
    classified_rows: int
    results: list[ClassifyItem]


class DownloadExcelRequest(BaseModel):
    results: list[ClassifyItem]

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text is required")

    # Split input into non-empty lines; treat each line as a separate bug report
    lines = [l.strip() for l in req.text.splitlines() if l.strip()]
    if not lines:
        raise HTTPException(status_code=400, detail="no bug lines found in text")

    try:
        classified = await batch_classify(lines)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"classification error: {e}")

    results = []
    for i, l in enumerate(lines):
        res = classified[i] if i < len(classified) else None
        if isinstance(res, dict):
            label = res.get('label') or ''
            raw = res.get('reason') or json.dumps(res)
            team = res.get('team')
            severity = res.get('severity')
        else:
            label = res or ''
            raw = json.dumps(res)
            team = None
            severity = None
        results.append({"text": l, "label": label, "raw": raw, "team": team, "severity": severity})

    return {"results": results}


@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload CSV/XLSX file với columns: No, Nội dung bug
    Trả về kết quả phân loại: No, Nội dung bug, Label, Reason, Team, Severity
    """
    global latest_results
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="no file provided")
    
    try:
        content = await file.read()
        
        # Parse CSV or Excel
        bugs_with_no = []
        
        if file.filename.endswith('.csv'):
            # Parse CSV
            stream = io.StringIO(content.decode('utf-8'))
            reader = csv.DictReader(stream)
            rows = list(reader)
            
            for row in rows:
                no = row.get('No') or row.get('no') or ''
                bug = row.get('Nội dung bug') or row.get('noi dung bug') or row.get('Bug') or ''
                if bug.strip():
                    bugs_with_no.append({'no': no, 'bug': bug.strip()})
        
        elif file.filename.endswith(('.xlsx', '.xls')):
            # Parse Excel
            if pd is None:
                raise HTTPException(status_code=400, detail="pandas not installed for Excel support")
            
            df = pd.read_excel(io.BytesIO(content))
            
            # Look for columns with 'No' and 'Nội dung bug' (flexible matching)
            no_col = None
            bug_col = None
            for col in df.columns:
                col_lower = str(col).lower()
                if 'no' in col_lower and no_col is None:
                    no_col = col
                if 'nội dung' in col_lower or 'bug' in col_lower:
                    bug_col = col
            
            if bug_col is None:
                raise HTTPException(status_code=400, detail="Excel file must have 'Nội dung bug' column")
            
            for idx, row in df.iterrows():
                no = str(row[no_col]) if no_col else str(idx + 1)
                bug = str(row[bug_col]).strip()
                if bug:
                    bugs_with_no.append({'no': no, 'bug': bug})
        
        else:
            raise HTTPException(status_code=400, detail="only CSV and Excel files are supported")
        
        if not bugs_with_no:
            raise HTTPException(status_code=400, detail="no valid bug entries found in file")
        
        # Extract just the bug texts for classification
        bug_texts = [item['bug'] for item in bugs_with_no]
        
        # Classify using batch
        classified = await batch_classify(bug_texts)
        
        # Build results with No column
        results = []
        for i, (bug_entry, classification) in enumerate(zip(bugs_with_no, classified)):
            if isinstance(classification, dict):
                label = classification.get('label') or ''
                reason = classification.get('reason') or ''
                team = classification.get('team')
                severity = classification.get('severity')
            else:
                label = str(classification) or ''
                reason = ''
                team = None
                severity = None
            
            results.append({
                "text": f"{bug_entry['no']} | {bug_entry['bug']}",
                "label": label,
                "raw": reason,
                "team": team,
                "severity": severity
            })
        
        # Store results for download
        latest_results = {
            "filename": file.filename,
            "results": results,
            "bugs_with_no": bugs_with_no,
            "classifications": classified
        }
        
        return {
            "filename": file.filename,
            "total_rows": len(bugs_with_no),
            "classified_rows": len(results),
            "results": results
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"file processing error: {str(e)}")


@app.get("/download-result")
async def download_result():
    """
    Download kết quả phân loại dưới dạng CSV
    Columns: No, Nội dung bug, Label, Giải thích, Team, Severity
    """
    global latest_results
    
    if not latest_results:
        raise HTTPException(status_code=400, detail="no classification results available")
    
    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['No', 'Nội dung bug', 'Label', 'Giải thích', 'Team', 'Severity'])
    
    # Write data rows
    for i, (bug_entry, classification) in enumerate(zip(latest_results['bugs_with_no'], latest_results['classifications'])):
        if isinstance(classification, dict):
            label = classification.get('label') or ''
            reason = classification.get('reason') or ''
            team = classification.get('team') or ''
            severity = classification.get('severity') or ''
        else:
            label = str(classification) or ''
            reason = ''
            team = ''
            severity = ''
        
        writer.writerow([
            bug_entry['no'],
            bug_entry['bug'],
            label,
            reason,
            team,
            severity
        ])
    
    # Return as file download
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=classification_result.csv"}
    )


@app.post("/download-excel")
async def download_excel(req: DownloadExcelRequest):
    """
    Generate Excel file từ classification results
    """
    if not Workbook:
        raise HTTPException(status_code=400, detail="openpyxl not installed")
    
    if not req.results:
        raise HTTPException(status_code=400, detail="no results to download")
    
    # Create Excel workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Classification Results"
    
    # Define styles
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    header_font = Font(bold=True, color="FFFFFF")
    
    # Write headers
    headers = ['No', 'Nội dung bug', 'Label', 'Giải thích', 'Team', 'Severity']
    for col, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col)
        cell.value = header
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
    
    # Sắp xếp kết quả theo label
    sorted_results = sorted(req.results, key=lambda r: (r.label or '').lower())
    for row_idx, result in enumerate(sorted_results, 2):
        # Parse text to extract No and Bug
        parts = result.text.split(' | ')
        no = parts[0] if parts else str(row_idx - 1)
        bug = ' | '.join(parts[1:]) if len(parts) > 1 else result.text
        row_data = [
            no,
            bug,
            result.label,
            result.raw,
            result.team or '',
            result.severity or ''
        ]
        for col, value in enumerate(row_data, 1):
            cell = ws.cell(row=row_idx, column=col)
            cell.value = value
            cell.alignment = Alignment(horizontal='left', vertical='top', wrap_text=True)
    
    # Auto-adjust column widths
    ws.column_dimensions['A'].width = 8
    ws.column_dimensions['B'].width = 40
    ws.column_dimensions['C'].width = 15
    ws.column_dimensions['D'].width = 35
    ws.column_dimensions['E'].width = 20
    ws.column_dimensions['F'].width = 12
    
    # Save to bytes
    excel_buffer = io.BytesIO()
    wb.save(excel_buffer)
    excel_buffer.seek(0)
    
    return StreamingResponse(
        iter([excel_buffer.getvalue()]),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=classification_results.xlsx"}
    )

# uvicorn entrypoint hint
# run with: uvicorn Server.api:app --reload --port 8000

