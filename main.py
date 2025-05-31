from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from together import Together
import fitz  # PyMuPDF
import json
import re
import os
import uuid
import tempfile
from difflib import SequenceMatcher
from typing import List, Optional, Dict, Tuple
from pydantic import BaseModel
from datetime import datetime
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    logger.error("TOGETHER_API_KEY not found in environment variables")
    raise ValueError("TOGETHER_API_KEY is required")

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 10485760))  # 10MB
SIMILARITY_THRESHOLD = 0.55  # Lowered for better matching
MAX_PAGES_TO_PROCESS = 50  # Prevent infinite processing

# Create directories
Path(UPLOAD_DIR).mkdir(exist_ok=True)
Path("static").mkdir(exist_ok=True)
Path("templates").mkdir(exist_ok=True)

# Models
class DocumentContext(BaseModel):
    document_type: str = "Contract"
    jurisdiction: str = "United States"
    business_type: str = "General Business"
    user_concerns: List[str] = ["liability", "termination", "payment terms"]

class RiskClause(BaseModel):
    text: str
    risk_level: str
    category: str
    page_number: int
    confidence: float
    amendment: Optional[str] = None

# --- Helper Functions ---
def normalize(text: str) -> str:
    """Normalize text for comparison - improved version"""
    if not text:
        return ""
    # Remove extra whitespace, convert to lowercase, remove special chars
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def similar(a: str, b: str) -> float:
    """Calculate similarity between two strings"""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()

def extract_text_by_pages(pdf_path: str) -> Dict[int, str]:
    """Extract text from PDF page by page"""
    doc = fitz.open(pdf_path)
    pages_text = {}
    
    for page_num in range(min(doc.page_count, MAX_PAGES_TO_PROCESS)):
        page = doc[page_num]
        text = page.get_text()
        pages_text[page_num + 1] = text
        logger.info(f"Extracted {len(text)} characters from page {page_num + 1}")
    
    doc.close()
    return pages_text

def chunk_text_for_analysis(text: str, max_chars: int = 4000) -> List[str]:
    """Split large text into chunks for LLM analysis"""
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    sentences = re.split(r'[.!?]+', text)
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk + sentence) <= max_chars:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def analyze_legal_risks(pages_text: Dict[int, str], context: DocumentContext) -> List[RiskClause]:
    """Analyze text for legal risks using LLM with improved processing"""
    logger.info("Starting comprehensive LLM risk analysis...")
    
    try:
        client = Together(api_key=TOGETHER_API_KEY)
        all_risk_clauses = []
        
        # Combine all text for initial analysis
        full_text = "\n".join(pages_text.values())
        text_chunks = chunk_text_for_analysis(full_text, 3500)
        
        for chunk_idx, chunk in enumerate(text_chunks):
            logger.info(f"Analyzing chunk {chunk_idx + 1}/{len(text_chunks)}")
            
            prompt = f'''
            Analyze this legal document section and identify 2-4 specific risk clauses that could be problematic. 
            Extract EXACT phrases from the document text that represent risks.

            DOCUMENT TYPE: {context.document_type}
            JURISDICTION: {context.jurisdiction}
            BUSINESS TYPE: {context.business_type}
            USER CONCERNS: {', '.join(context.user_concerns)}

            IMPORTANT: Return ONLY a JSON array. Each clause text should be 5-30 words from the actual document.

            [
                {{
                    "text": "exact phrase from document",
                    "risk_level": "High|Medium|Low",
                    "category": "liability|termination|payment|confidentiality|intellectual_property|dispute_resolution|indemnification|limitation_of_liability|other",
                    "explanation": "brief risk explanation"
                }}
            ]

            Document section: {chunk}
            '''
            
            try:
                response = client.chat.completions.create(
                    model="meta-llama/Llama-Vision-Free",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0.2,
                    top_p=0.8
                )
                
                result = response.choices[0].message.content.strip()
                
                # Extract JSON from response
                json_match = re.search(r'\[.*?\]', result, re.DOTALL)
                if json_match:
                    try:
                        clauses_data = json.loads(json_match.group())
                        for clause in clauses_data:
                            if clause.get("text") and len(clause.get("text", "").strip()) > 10:
                                # Try to find which page this clause is on
                                clause_text = clause.get("text", "")
                                found_page = 1
                                
                                for page_num, page_text in pages_text.items():
                                    if clause_text.lower() in page_text.lower():
                                        found_page = page_num
                                        break
                                
                                all_risk_clauses.append(RiskClause(
                                    text=clause.get("text", ""),
                                    risk_level=clause.get("risk_level", "Medium"),
                                    category=clause.get("category", "other"),
                                    page_number=found_page,
                                    confidence=0.8
                                ))
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON parsing error for chunk {chunk_idx}: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error analyzing chunk {chunk_idx}: {e}")
                continue
        
        # Remove duplicates based on similarity
        unique_clauses = []
        for clause in all_risk_clauses:
            is_duplicate = False
            for existing in unique_clauses:
                if similar(clause.text, existing.text) > 0.8:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_clauses.append(clause)
        
        logger.info(f"Found {len(unique_clauses)} unique risk clauses")
        return unique_clauses[:8]  # Limit to top 8 risks
        
    except Exception as e:
        logger.error(f"Error in LLM analysis: {e}")
        return []

def get_llm_amendment(client: Together, clause_text: str, risk_level: str, category: str, context: DocumentContext) -> str:
    """Generate amendment suggestion for a clause using LLM"""
    prompt = f'''
    Provide a clear, improved version of this legal clause to reduce risk. 
    Keep it concise and professional.

    CONTEXT:
    - Document Type: {context.document_type}
    - Risk Level: {risk_level}
    - Category: {category}

    ORIGINAL CLAUSE: "{clause_text}"

    IMPROVED VERSION:
    '''
    
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-Vision-Free",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0.3
        )
        amendment = response.choices[0].message.content.strip()
        return amendment[:500] if amendment else f"Consider revising this {category} clause to reduce {risk_level.lower()} risk."
    except Exception as e:
        logger.error(f"Amendment generation failed: {e}")
        return f"Review this {category} clause to reduce {risk_level.lower()} risk."

def find_text_in_page(page, target_text: str) -> Tuple[List[fitz.Rect], float]:
    """Find and return rectangles for text matching in a page"""
    words = page.get_text("words")
    if not words:
        return [], 0.0
    
    word_texts = [w[4] for w in words]
    target_words = normalize(target_text).split()
    
    if len(target_words) == 0:
        return [], 0.0
    
    best_rects = []
    best_ratio = 0.0
    
    # Try different window sizes
    for window_size in range(max(1, len(target_words) - 2), len(target_words) + 3):
        if window_size > len(word_texts):
            continue
            
        for i in range(len(word_texts) - window_size + 1):
            window_words = word_texts[i:i + window_size]
            window_text = " ".join(window_words)
            
            ratio = similar(target_text, window_text)
            
            if ratio > best_ratio and ratio > SIMILARITY_THRESHOLD:
                best_ratio = ratio
                word_rects = [fitz.Rect(words[j][:4]) for j in range(i, i + window_size)]
                best_rects = word_rects
    
    return best_rects, best_ratio

def create_annotation(page, rects: List[fitz.Rect], clause: RiskClause, amendment: str) -> bool:
    """Create highlight and popup annotation"""
    if not rects:
        return False
    
    try:
        # Color mapping
        color_map = {
            "High": (1, 0, 0),      # Red
            "Medium": (1, 0.5, 0),  # Orange  
            "Low": (1, 1, 0)        # Yellow
        }
        color = color_map.get(clause.risk_level, (0.7, 0.7, 0.7))
        
        # Add highlight
        highlight = page.add_highlight_annot(rects)
        highlight.set_colors(stroke=color)
        highlight.update()
        
        # Create popup annotation - positioned near the highlight
        popup_rect = fitz.Rect(rects[-1].x1 + 10, rects[0].y0, rects[-1].x1 + 350, rects[0].y0 + 120)
        
        # Adjust if popup goes off page
        page_rect = page.rect
        if popup_rect.x1 > page_rect.x1:
            popup_rect = fitz.Rect(rects[0].x0 - 350, rects[0].y0, rects[0].x0 - 10, rects[0].y0 + 120)
        
        # Create text annotation with popup
        popup_point = fitz.Point(rects[-1].x1 + 5, rects[0].y0 + 5)
        text_annot = page.add_text_annot(popup_point, "Click for suggested improvement")
        
        # Set annotation properties for better visibility
        text_annot.set_info(
            title=f"⚠️ {clause.category.title()} Risk ({clause.risk_level})",
            content=f"ORIGINAL CLAUSE:\n{clause.text}\n\n{'='*50}\n\nSUGGESTED IMPROVEMENT:\n{amendment}\n\n{'='*50}\n\nRisk Level: {clause.risk_level}\nCategory: {clause.category.title()}"
        )
        
        # Set colors and icon
        text_annot.set_colors(stroke=(0, 0, 1), fill=(1, 1, 0.8))
        
        # Set popup rectangle
        text_annot.set_popup(popup_rect)
        text_annot.set_open(False)  # Start closed
        
        # Update annotation
        text_annot.update()
        
        logger.info(f"✅ Created annotation for {clause.category} risk")
        return True
        
    except Exception as e:
        logger.error(f"Error creating annotation: {e}")
        return False

def highlight_pdf_with_risks(pdf_path: str, risk_clauses: List[RiskClause], context: DocumentContext) -> BytesIO:
    """Generate highlighted PDF with risk annotations - optimized version"""
    logger.info(f"Highlighting PDF with {len(risk_clauses)} risk clauses")
    
    try:
        doc = fitz.open(pdf_path)
        client = Together(api_key=TOGETHER_API_KEY)
        
        # Pre-generate all amendments to avoid repeated API calls
        amendments = {}
        logger.info("Pre-generating amendments...")
        
        for clause in risk_clauses:
            amendment_key = f"{clause.text}_{clause.category}_{clause.risk_level}"
            if amendment_key not in amendments:
                amendment = get_llm_amendment(client, clause.text, clause.risk_level, clause.category, context)
                amendments[amendment_key] = amendment
                logger.info(f"Generated amendment for: {clause.text[:30]}...")
        
        # Process each page
        annotations_added = 0
        pages_processed = min(doc.page_count, MAX_PAGES_TO_PROCESS)
        
        for page_num in range(pages_processed):
            page = doc[page_num]
            logger.info(f"Processing page {page_num + 1}/{pages_processed}")
            
            page_clauses = [c for c in risk_clauses if c.page_number == page_num + 1]
            if not page_clauses:
                # Also check clauses that might be on any page
                page_clauses = [c for c in risk_clauses if c.page_number == 1]
            
            for clause in page_clauses:
                try:
                    # Find text in page
                    rects, ratio = find_text_in_page(page, clause.text)
                    
                    if rects and ratio > SIMILARITY_THRESHOLD:
                        amendment_key = f"{clause.text}_{clause.category}_{clause.risk_level}"
                        amendment = amendments.get(amendment_key, "Review this clause for potential risks.")
                        
                        if create_annotation(page, rects, clause, amendment):
                            annotations_added += 1
                            logger.info(f" Added annotation on page {page_num + 1}: {clause.text[:30]}... (Match: {ratio:.2f})")
                        
                        # Limit annotations per page to avoid clutter
                        if annotations_added >= 15:
                            logger.info("Reached maximum annotations limit")
                            break
                    else:
                        logger.info(f"❌ No good match on page {page_num + 1} for: {clause.text[:30]}... (Best ratio: {ratio:.2f})")
                        
                except Exception as e:
                    logger.error(f"Error processing clause on page {page_num + 1}: {e}")
                    continue
            
            if annotations_added >= 15:
                break
        
        logger.info(f"Total annotations added: {annotations_added}")
        
        # Save to BytesIO
        pdf_buffer = BytesIO()
        doc.save(pdf_buffer)
        pdf_buffer.seek(0)
        doc.close()
        
        logger.info("PDF highlighting completed successfully")
        return pdf_buffer
        
    except Exception as e:
        logger.error(f"Error in PDF highlighting: {e}")
        raise

# --- FastAPI App Setup ---
app = FastAPI(
    title="Legal Document Risk Analyzer",
    description="Upload legal documents and get risk analysis with highlighted PDFs",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with upload form"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze/")
async def analyze_document(
    request: Request,
    file: UploadFile = File(...),
    document_type: str = Form("Contract"),
    jurisdiction: str = Form("United States"),
    business_type: str = Form("General Business"),
    user_concerns: str = Form("liability,termination,payment terms")
):
    """Analyze uploaded legal document with optimized processing"""
    logger.info(f"Starting analysis for file: {file.filename}")
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
        
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Read file content
    try:
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB")
        
        logger.info(f"File size: {len(content)} bytes")
        
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        raise HTTPException(status_code=400, detail="Error reading uploaded file")
    
    # Save uploaded file temporarily
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")
    
    try:
        with open(file_path, "wb") as f:
            f.write(content)
        logger.info(f"File saved to: {file_path}")
        
        # Create context
        context = DocumentContext(
            document_type=document_type,
            jurisdiction=jurisdiction,
            business_type=business_type,
            user_concerns=[concern.strip() for concern in user_concerns.split(",")]
        )
        logger.info(f"Analysis context: {context}")
        
        # Extract text by pages
        logger.info("Extracting text from PDF by pages...")
        pages_text = extract_text_by_pages(file_path)
        total_chars = sum(len(text) for text in pages_text.values())
        logger.info(f"Extracted {total_chars} characters from {len(pages_text)} pages")
        
        if total_chars < 100:
            raise HTTPException(status_code=400, detail="Could not extract sufficient text from PDF. Please ensure the PDF contains readable text.")
        
        # Analyze risks with improved processing
        logger.info("Analyzing risks with optimized LLM processing...")
        risk_clauses = analyze_legal_risks(pages_text, context)
        logger.info(f"Found {len(risk_clauses)} risk clauses")
        
        if not risk_clauses:
            raise HTTPException(status_code=422, detail="No significant risks were identified in this document. The document may be too short or contain standard clauses.")
        
        # Generate highlighted PDF with optimized processing
        logger.info("Generating optimized highlighted PDF...")
        highlighted_pdf = highlight_pdf_with_risks(file_path, risk_clauses, context)
        logger.info("Highlighted PDF generated successfully")
        
        # Clean up original file
        try:
            os.remove(file_path)
            logger.info("Temporary file cleaned up")
        except:
            pass
        
        # Return highlighted PDF
        filename = f"risk_analysis_{file.filename}"
        return StreamingResponse(
            BytesIO(highlighted_pdf.getvalue()),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Clean up on error
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except:
            pass
            
        return JSONResponse(
            status_code=500,
            content={"error": f"Analysis failed: {str(e)}", "details": "Please check the server logs for more information"}
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# --- Run Application ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)