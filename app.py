import re
import io
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import streamlit as st

from pypdf import PdfReader
import docx
from sentence_transformers import SentenceTransformer
import faiss

# Google
import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload

# OCR deps (optional)
OCR_AVAILABLE = True
try:
    import pytesseract
    from PIL import Image
    from pdf2image import convert_from_bytes
except Exception:
    OCR_AVAILABLE = False


# =========================
# Config
# =========================
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DOC_TYPES = ["Invoice", "Tax Exempt"]

FAISS_FILENAME = "faiss.index"
MAP_FILENAME = "faiss_doc_ids.npy"

# For 500–1000 docs: keep smaller preview in Sheets for speed
MAX_STORE_CHARS = 15000

DRIVE_FOLDER_ID = st.secrets["DRIVE_FOLDER_ID"]
SHEET_URL = st.secrets["SHEET_URL"]
WORKSHEET_NAME = st.secrets.get("WORKSHEET_NAME", "Sheet1")

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


# =========================
# Session-state helpers
# =========================
def reset_upload_fields():
    """Clear form fields when switching doc_type and reset uploader by changing its key."""
    st.session_state.pop("supplier_name", None)
    st.session_state.pop("amount", None)
    st.session_state.pop("customer_name", None)
    st.session_state.pop("business_name", None)
    st.session_state.pop("resale_number", None)
    st.session_state.uploader_key = st.session_state.get("uploader_key", 0) + 1


# =========================
# Google Clients
# =========================
@st.cache_resource
def get_gcp_clients():
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=SCOPES
    )
    gc = gspread.authorize(creds)
    drive = build("drive", "v3", credentials=creds)
    ws = gc.open_by_url(SHEET_URL).worksheet(WORKSHEET_NAME)
    return drive, ws

drive, ws = get_gcp_clients()


# =========================
# Google Sheets DB (Metadata + extracted text preview)
# =========================
def ensure_sheet_header():
    header = ws.row_values(1)
    if header:
        return
    ws.append_row([
        "id", "doc_type",
        "supplier_name", "amount",
        "customer_name", "business_name", "resale_number",
        "filename", "drive_file_id", "drive_link",
        "extracted_text", "uploaded_at"
    ])

@st.cache_data(ttl=60)
def get_all_sheet_records_cached() -> List[Dict[str, Any]]:
    # cached 60 seconds to avoid repeated API calls
    return ws.get_all_records()

def _sheet_records() -> List[Dict[str, Any]]:
    return get_all_sheet_records_cached()

def insert_doc_sheet(row: dict) -> int:
    # Create incremental id based on current row count (header is row 1)
    current_values = ws.get_all_values()
    new_id = len(current_values)  # row 1 header, so next row index == len(values)

    extracted_store = (row.get("extracted_text") or "")[:MAX_STORE_CHARS]

    ws.append_row([
        int(new_id),
        row["doc_type"],
        row.get("supplier_name") or "",
        row.get("amount") if row.get("amount") is not None else "",
        row.get("customer_name") or "",
        row.get("business_name") or "",
        row.get("resale_number") or "",
        row["filename"],
        row["drive_file_id"],
        row["drive_link"],
        extracted_store,
        row["uploaded_at"],
    ], value_input_option="USER_ENTERED")

    return int(new_id)

def fetch_all_docs_for_rebuild_sheet():
    recs = _sheet_records()
    rows = []
    for r in recs:
        try:
            rows.append({"id": int(r.get("id")), "extracted_text": r.get("extracted_text", "")})
        except Exception:
            continue
    rows.sort(key=lambda x: x["id"])
    return rows

ensure_sheet_header()


# =========================
# Google Drive storage (documents + FAISS + mapping)
# =========================
def drive_find_by_name(name: str) -> Optional[str]:
    q = f"'{DRIVE_FOLDER_ID}' in parents and name='{name}' and trashed=false"
    res = drive.files().list(q=q, fields="files(id,name)").execute()
    files = res.get("files", [])
    return files[0]["id"] if files else None

def drive_upload_document(filename: str, content: bytes) -> tuple[str, str]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = filename.replace("/", "_").replace("\\", "_")
    final_name = f"{ts}__{safe}"

    media = MediaIoBaseUpload(io.BytesIO(content), mimetype="application/octet-stream", resumable=False)
    meta = {"name": final_name, "parents": [DRIVE_FOLDER_ID]}

    created = drive.files().create(body=meta, media_body=media, fields="id, webViewLink").execute()
    return created["id"], created["webViewLink"]

def drive_download_file(file_id: str) -> bytes:
    req = drive.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, req)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return fh.getvalue()

@st.cache_data(ttl=600)
def drive_download_cached(file_id: str) -> bytes:
    # cache downloads for 10 minutes
    return drive_download_file(file_id)

def drive_upsert_bytes(name: str, data_bytes: bytes):
    existing_id = drive_find_by_name(name)
    media = MediaIoBaseUpload(io.BytesIO(data_bytes), mimetype="application/octet-stream", resumable=False)
    meta = {"name": name, "parents": [DRIVE_FOLDER_ID]}

    if existing_id:
        drive.files().update(fileId=existing_id, media_body=media).execute()
    else:
        drive.files().create(body=meta, media_body=media).execute()

def drive_read_bytes_by_name(name: str) -> Optional[bytes]:
    file_id = drive_find_by_name(name)
    if not file_id:
        return None
    return drive_download_file(file_id)


# =========================
# Text extraction + OCR
# =========================
def clean_text(t: str) -> str:
    t = t or ""
    t = re.sub(r"\s+", " ", t).strip()
    return t

def extract_text_pdf_bytes(pdf_bytes: bytes) -> str:
    parts = []
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        for page in reader.pages:
            parts.append(page.extract_text() or "")
    except Exception:
        return ""
    return "\n".join(parts)

def extract_text_docx_bytes(docx_bytes: bytes) -> str:
    try:
        d = docx.Document(io.BytesIO(docx_bytes))
        return "\n".join(p.text for p in d.paragraphs if p.text)
    except Exception:
        return ""

def ocr_image_bytes(img_bytes: bytes) -> str:
    if not OCR_AVAILABLE:
        return ""
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return pytesseract.image_to_string(img) or ""
    except Exception:
        return ""

def ocr_pdf_bytes(pdf_bytes: bytes, max_pages: int = 6) -> str:
    if not OCR_AVAILABLE:
        return ""
    try:
        images = convert_from_bytes(pdf_bytes, first_page=1, last_page=max_pages)
        texts = []
        for img in images:
            texts.append(pytesseract.image_to_string(img) or "")
        return "\n".join(texts)
    except Exception:
        return ""

def extract_all_text(filename: str, file_bytes: bytes) -> Tuple[str, str]:
    ext = Path(filename).suffix.lower()

    if ext == ".txt":
        try:
            return file_bytes.decode("utf-8", errors="ignore"), "txt"
        except Exception:
            return "", "none"

    if ext == ".docx":
        t = extract_text_docx_bytes(file_bytes)
        return t, "docx_text" if t.strip() else "none"

    if ext == ".pdf":
        t = extract_text_pdf_bytes(file_bytes)
        if t.strip():
            return t, "pdf_text"
        t2 = ocr_pdf_bytes(file_bytes)
        return t2, "ocr_pdf" if t2.strip() else "none"

    if ext in [".png", ".jpg", ".jpeg"]:
        t = ocr_image_bytes(file_bytes)
        return t, "ocr_image" if t.strip() else "none"

    return "", "none"


# =========================
# Embeddings + FAISS (persisted to Google Drive)
# =========================
@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

def embed_text(model, text: str) -> np.ndarray:
    v = model.encode([text], normalize_embeddings=True)
    return v.astype("float32")

def load_index_from_drive(dim: int):
    data = drive_read_bytes_by_name(FAISS_FILENAME)
    if not data:
        return faiss.IndexFlatIP(dim)  # cosine via normalized vectors

    tmp_path = "faiss_tmp.index"
    with open(tmp_path, "wb") as f:
        f.write(data)
    return faiss.read_index(tmp_path)

def save_index_to_drive(idx):
    tmp_path = "faiss_tmp.index"
    faiss.write_index(idx, tmp_path)
    with open(tmp_path, "rb") as f:
        data = f.read()
    drive_upsert_bytes(FAISS_FILENAME, data)

def load_mapping_from_drive():
    data = drive_read_bytes_by_name(MAP_FILENAME)
    if not data:
        return []
    arr = np.load(io.BytesIO(data))
    return arr.tolist()

def save_mapping_to_drive(doc_ids):
    buf = io.BytesIO()
    np.save(buf, np.array(doc_ids, dtype=np.int64))
    buf.seek(0)
    drive_upsert_bytes(MAP_FILENAME, buf.getvalue())

def rebuild_index_from_sheet(model):
    rows = fetch_all_docs_for_rebuild_sheet()

    dim = model.get_sentence_embedding_dimension()
    idx = faiss.IndexFlatIP(dim)
    doc_ids = []
    vectors = []

    for r in rows:
        t = clean_text(r.get("extracted_text") or "")
        if not t:
            continue
        vectors.append(model.encode([t], normalize_embeddings=True)[0])
        doc_ids.append(int(r["id"]))

    if vectors:
        vecs = np.array(vectors, dtype="float32")
        idx.add(vecs)

    save_index_to_drive(idx)
    save_mapping_to_drive(doc_ids)
    return idx, doc_ids


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Doc Upload + AI Search (Google Drive + Sheets)", layout="wide")
st.title("Document Upload + AI Search (Google Drive + Sheets)")

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

model = load_model()
dim = model.get_sentence_embedding_dimension()

index = load_index_from_drive(dim)
doc_id_map = load_mapping_from_drive()

# Keep FAISS + mapping consistent
if index.ntotal != len(doc_id_map):
    index, doc_id_map = rebuild_index_from_sheet(model)

page = st.sidebar.radio("Menu", ["Upload", "AI Search"], index=0)

# ---------------- Upload ----------------
if page == "Upload":
    st.subheader("Upload")

    doc_type = st.selectbox(
        "Document Type",
        DOC_TYPES,
        key="doc_type_selector",
        on_change=reset_upload_fields
    )

    with st.form("upload_form", clear_on_submit=True):

        if doc_type == "Invoice":
            supplier_name = st.text_input("Supplier Name *", key="supplier_name")
            amount = st.number_input(
                "Amount *",
                min_value=0.0,
                step=0.01,
                format="%.2f",
                key="amount"
            )

            customer_name = None
            business_name = None
            resale_number = None

        else:
            customer_name = st.text_input("Customer Name *", key="customer_name")
            business_name = st.text_input("Business Name *", key="business_name")
            resale_number = st.text_input("Resale Number *", key="resale_number")

            supplier_name = None
            amount = None

        uploaded = st.file_uploader(
            "Attach File * (pdf/txt/docx; images searchable with OCR)",
            type=["pdf", "txt", "docx", "png", "jpg", "jpeg"],
            key=f"file_{st.session_state.uploader_key}"
        )

        submitted = st.form_submit_button("Submit")

    if submitted:
        errors = []

        if doc_type == "Invoice":
            if not (supplier_name or "").strip():
                errors.append("Supplier Name is required for Invoice.")
            if amount is None or amount <= 0:
                errors.append("Amount must be greater than 0 for Invoice.")
        else:
            if not (customer_name or "").strip():
                errors.append("Customer Name is required for Tax Exempt.")
            if not (business_name or "").strip():
                errors.append("Business Name is required for Tax Exempt.")
            if not (resale_number or "").strip():
                errors.append("Resale Number is required for Tax Exempt.")

        if uploaded is None:
            errors.append("File attachment is required.")

        if errors:
            for e in errors:
                st.error(e)
        else:
            file_bytes = uploaded.getvalue()

            # 1) Save file to Google Drive (permanent)
            drive_file_id, drive_link = drive_upload_document(uploaded.name, file_bytes)

            # 2) Extract text (OCR fallback)
            extracted, method = extract_all_text(uploaded.name, file_bytes)
            extracted = clean_text(extracted)

            # 3) Save metadata + extracted text preview to Google Sheets
            doc_id = insert_doc_sheet({
                "doc_type": doc_type,
                "supplier_name": supplier_name.strip() if supplier_name else None,
                "amount": float(amount) if doc_type == "Invoice" else None,
                "customer_name": customer_name.strip() if customer_name else None,
                "business_name": business_name.strip() if business_name else None,
                "resale_number": resale_number.strip() if resale_number else None,
                "filename": uploaded.name,
                "drive_file_id": drive_file_id,
                "drive_link": drive_link,
                "extracted_text": extracted,
                "uploaded_at": datetime.now().isoformat(timespec="seconds"),
            })

            # IMPORTANT: clear sheet cache so search sees the new doc immediately
            st.cache_data.clear()

            # 4) Add to FAISS index + persist to Drive
            if extracted:
                v = embed_text(model, extracted)
                index.add(v)
                doc_id_map.append(doc_id)
                save_index_to_drive(index)
                save_mapping_to_drive(doc_id_map)
                st.success(f"Saved to Google Drive + indexed! (extraction: {method})")
                st.write("Drive link:", drive_link)
            else:
                st.warning(
                    f"Saved to Google Drive, but no text extracted (extraction: {method}). "
                    "If it’s scanned PDF/image, OCR must be available."
                )
                st.write("Drive link:", drive_link)

            reset_upload_fields()
            st.rerun()

    if not OCR_AVAILABLE:
        st.info("OCR not available in this environment. Text PDFs work; scanned docs need OCR.")

# ---------------- AI Search ----------------
else:
    st.subheader("AI Semantic Search")

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        q = st.text_input("Search (example: 'invoice from sysco 1200' or 'tax exempt resale number')")

    with c2:
        doc_type_filter = st.selectbox("Type filter", ["All"] + DOC_TYPES, index=0)

    with c3:
        top_k = st.selectbox("Top results", [5, 10, 20, 50], index=1)

    if st.button("Search", disabled=not bool(q.strip())):
        if index.ntotal == 0:
            st.warning("No indexed documents yet. Upload documents first.")
        else:
            # Pull sheet once, then use dict for instant lookups
            records = _sheet_records()
            doc_by_id = {}
            for r in records:
                try:
                    doc_by_id[int(r["id"])] = r
                except Exception:
                    continue

            qv = embed_text(model, clean_text(q))
            D, I = index.search(qv, int(top_k))

            shown = 0
            for score, pos in zip(D[0], I[0]):
                if pos < 0 or pos >= len(doc_id_map):
                    continue

                doc_id = int(doc_id_map[pos])
                r = doc_by_id.get(doc_id)
                if not r:
                    continue

                if doc_type_filter != "All" and r.get("doc_type") != doc_type_filter:
                    continue

                shown += 1
                st.markdown(f"### #{r['id']} • {r['doc_type']} • {r['filename']}")
                cols = st.columns(4)

                if r["doc_type"] == "Invoice":
                    cols[0].write(f"**Supplier:** {r.get('supplier_name') or '-'}")
                    cols[1].write(f"**Amount:** {r.get('amount') or '-'}")
                else:
                    cols[0].write(f"**Customer:** {r.get('customer_name') or '-'}")
                    cols[1].write(f"**Business:** {r.get('business_name') or '-'}")
                    cols[2].write(f"**Resale #:** {r.get('resale_number') or '-'}")

                cols[3].write(f"**Score:** {float(score):.3f}")

                preview = (r.get("extracted_text") or "")[:350]
                if preview:
                    st.caption(preview)

                # Download from Google Drive (cached 10 minutes)
                file_bytes = drive_download_cached(r["drive_file_id"])
                st.download_button(
                    label="Download file",
                    data=file_bytes,
                    file_name=r["filename"],
                    mime="application/octet-stream",
                    key=f"dl_{r['id']}"
                )

                st.write("Drive link:", r.get("drive_link", ""))
                st.code(r["drive_file_id"], language="text")
                st.divider()

            st.info(f"Displayed {shown} result(s).")
