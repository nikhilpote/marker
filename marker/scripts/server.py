import base64
import asyncio
import os
import traceback
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Optional, Annotated, Dict, List
from uuid import uuid4
import io

import click
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from pydantic import BaseModel, Field
from starlette.responses import HTMLResponse

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.settings import settings
from marker.config.parser import ConfigParser
from marker.output import text_from_rendered, save_output
from marker.scripts.render_from_json import render_json_document
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app_data = {}


UPLOAD_DIRECTORY = "./uploads"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
JOB_DIRECTORY = "./job_outputs"
os.makedirs(JOB_DIRECTORY, exist_ok=True)


class JobStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class JobResult(BaseModel):
    json_path: Optional[str] = None
    html_path: Optional[str] = None
    metadata_path: Optional[str] = None
    assets: List[str] = Field(default_factory=list)


class JobResponse(BaseModel):
    job_id: str
    filename: str
    status: str
    error: Optional[str] = None
    result: Optional[JobResult] = None
    output_dir: Optional[str] = None
    progress: int = 0
    stage: Optional[str] = None
    detail: Optional[str] = None
    processed_pages: Optional[int] = None
    total_pages: Optional[int] = None


def _job_lock() -> Lock:
    return app_data["job_lock"]


def _job_store() -> Dict[str, Dict]:
    return app_data["jobs"]


def _executor() -> ThreadPoolExecutor:
    return app_data["executor"]


def _update_job(job_id: str, **updates):
    with _job_lock():
        _job_store()[job_id].update(updates)


def _get_job(job_id: str) -> Dict:
    with _job_lock():
        return _job_store().get(job_id)


def _process_job(job_id: str):
    job = _get_job(job_id)
    if not job:
        return

    _update_job(job_id, status=JobStatus.processing, detail=None)
    last_progress = int(job.get("progress", 0) or 0)
    total_pages_state = {"value": job.get("total_pages")}

    def set_progress(progress=None, stage=None, detail=None, processed=None, total=None):
        nonlocal last_progress
        updates = {}
        if progress is not None:
            clamped = max(0, min(99, int(progress)))
            if clamped < last_progress:
                clamped = last_progress
            updates["progress"] = clamped
            last_progress = clamped
        if stage is not None:
            updates["stage"] = stage
        if detail is not None:
            updates["detail"] = detail
        if processed is not None:
            updates["processed_pages"] = processed
        if total is not None:
            updates["total_pages"] = total
        if updates:
            _update_job(job_id, **updates)

    set_progress(progress=5, stage="Initializing GPU pipeline")

    input_path = job["input_path"]
    output_dir = job["output_dir"]

    stage_progress_map = {
        "document_preparation": (12, "Preparing document"),
        "layout_detection": (25, "Detecting layout regions"),
        "line_extraction": (38, "Extracting text lines"),
        "ocr_processing": (55, "Running OCR"),
        "document_ready": (60, "Compiling pages"),
        "structure_analysis": (68, "Analyzing structure"),
        "processors_complete": (88, "Refinements complete"),
        "rendering_start": (92, "Rendering outputs"),
        "rendering_complete": (96, "Rendering complete"),
    }
    processor_progress_range = (68, 88)

    def progress_callback(event):
        if not isinstance(event, dict):
            return
        stage = event.get("stage")
        if event.get("total_pages") is not None:
            total_pages_state["value"] = event["total_pages"]
        processed_pages = event.get("processed_pages")

        if stage == "processors_progress":
            total_steps = max(1, int(event.get("total", 1)))
            current = int(min(max(event.get("current", 0), 0), total_steps))
            start, end = processor_progress_range
            fraction = current / total_steps
            progress_value = start + fraction * (end - start)
            detail_text = f"Processor {current}/{total_steps} applied"
            set_progress(
                progress=progress_value,
                stage="Applying refinements",
                detail=detail_text,
                processed=processed_pages,
                total=total_pages_state["value"],
            )
            return

        if stage in stage_progress_map:
            progress_value, label = stage_progress_map[stage]
            detail_text = event.get("detail")
            if stage == "document_preparation" and detail_text is None and total_pages_state["value"]:
                detail_text = f"Detected {total_pages_state['value']} page(s)"
            set_progress(
                progress=progress_value,
                stage=label,
                detail=detail_text,
                processed=processed_pages,
                total=total_pages_state["value"],
            )
            return

        if stage is not None:
            set_progress(
                progress=event.get("progress"),
                stage=stage,
                detail=event.get("detail"),
                processed=processed_pages,
                total=total_pages_state["value"],
            )

    try:
        options = {
            "output_format": "json",
            "output_dir": output_dir,
            "layout_batch_size": 1,
            "detection_batch_size": 1,
            "table_rec_batch_size": 1,
            "ocr_error_batch_size": 1,
            "recognition_batch_size": 4,
            "equation_batch_size": 1,
        }
        config_parser = ConfigParser(options)
        config_dict = config_parser.generate_config_dict()
        config_dict["disable_tqdm"] = True
        converter_cls = config_parser.get_converter_cls()
        converter = converter_cls(
            config=config_dict,
            artifact_dict=app_data["models"],
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service(),
        )
        rendered = converter(input_path, progress_callback=progress_callback)
        base_name = config_parser.get_base_filename(input_path)
        set_progress(stage="Saving outputs", progress=97)
        save_output(rendered, output_dir, base_name)

        json_path = os.path.join(output_dir, f"{base_name}.json")
        html_path = None
        if os.path.exists(json_path):
            html_path = str(render_json_document(Path(json_path)))

        metadata_path = os.path.join(output_dir, f"{base_name}_meta.json")
        assets = []
        for asset_name in sorted(os.listdir(output_dir)):
            if asset_name.endswith((".jpeg", ".jpg", ".png")):
                full_asset_path = os.path.join(output_dir, asset_name)
                assets.append(os.path.relpath(full_asset_path, start=output_dir))

        json_rel = (
            os.path.relpath(json_path, start=output_dir)
            if os.path.exists(json_path)
            else None
        )
        html_rel = (
            os.path.relpath(html_path, start=output_dir)
            if html_path and os.path.exists(html_path)
            else None
        )
        metadata_rel = (
            os.path.relpath(metadata_path, start=output_dir)
            if os.path.exists(metadata_path)
            else None
        )

        _update_job(
            job_id,
            status=JobStatus.completed,
            result=JobResult(
                json_path=json_rel,
                html_path=html_rel,
                metadata_path=metadata_rel,
                assets=assets,
            ),
            progress=100,
            stage="Completed",
            detail="Conversion successful",
            processed_pages=total_pages_state["value"],
            total_pages=total_pages_state["value"],
        )
    except Exception as exc:
        traceback.print_exc()
        _update_job(
            job_id,
            status=JobStatus.failed,
            error=str(exc),
            stage="Failed",
            detail=str(exc),
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_data["models"] = create_model_dict()
    app_data["jobs"] = {}
    app_data["job_lock"] = Lock()
    worker_count = int(os.getenv("MARKER_SERVER_WORKERS", "1"))
    app_data["executor"] = ThreadPoolExecutor(max_workers=max(1, worker_count))

    yield

    if "models" in app_data:
        del app_data["models"]
    if "executor" in app_data:
        app_data["executor"].shutdown(wait=True)


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return HTMLResponse(
        """
<h1>Marker API</h1>
<ul>
    <li><a href="/docs">API Documentation</a></li>
    <li><a href="/marker">Run marker (post request only)</a></li>
</ul>
"""
    )


class CommonParams(BaseModel):
    filepath: Annotated[
        Optional[str], Field(description="The path to the PDF file to convert.")
    ]
    page_range: Annotated[
        Optional[str],
        Field(
            description="Page range to convert, specify comma separated page numbers or ranges.  Example: 0,5-10,20",
            example=None,
        ),
    ] = None
    force_ocr: Annotated[
        bool,
        Field(
            description="Force OCR on all pages of the PDF.  Defaults to False.  This can lead to worse results if you have good text in your PDFs (which is true in most cases)."
        ),
    ] = False
    paginate_output: Annotated[
        bool,
        Field(
            description="Whether to paginate the output.  Defaults to False.  If set to True, each page of the output will be separated by a horizontal rule that contains the page number (2 newlines, {PAGE_NUMBER}, 48 - characters, 2 newlines)."
        ),
    ] = False
    output_format: Annotated[
        str,
        Field(
            description="The format to output the text in.  Can be 'markdown', 'json', or 'html'.  Defaults to 'markdown'."
        ),
    ] = "markdown"


async def _convert_pdf(params: CommonParams):
    assert params.output_format in ["markdown", "json", "html", "chunks"], (
        "Invalid output format"
    )
    try:
        options = params.model_dump()
        config_parser = ConfigParser(options)
        config_dict = config_parser.generate_config_dict()
        config_dict["pdftext_workers"] = 1
        converter_cls = PdfConverter
        converter = converter_cls(
            config=config_dict,
            artifact_dict=app_data["models"],
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service(),
        )
        rendered = converter(params.filepath)
        text, _, images = text_from_rendered(rendered)
        metadata = rendered.metadata
    except Exception as e:
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
        }

    encoded = {}
    for k, v in images.items():
        byte_stream = io.BytesIO()
        v.save(byte_stream, format=settings.OUTPUT_IMAGE_FORMAT)
        encoded[k] = base64.b64encode(byte_stream.getvalue()).decode(
            settings.OUTPUT_ENCODING
        )

    return {
        "format": params.output_format,
        "output": text,
        "images": encoded,
        "metadata": metadata,
        "success": True,
    }


@app.post("/marker")
async def convert_pdf(params: CommonParams):
    return await _convert_pdf(params)


@app.post("/marker/upload")
async def convert_pdf_upload(
    page_range: Optional[str] = Form(default=None),
    force_ocr: Optional[bool] = Form(default=False),
    paginate_output: Optional[bool] = Form(default=False),
    output_format: Optional[str] = Form(default="markdown"),
    file: UploadFile = File(
        ..., description="The PDF file to convert.", media_type="application/pdf"
    ),
):
    upload_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
    with open(upload_path, "wb+") as upload_file:
        file_contents = await file.read()
        upload_file.write(file_contents)

    params = CommonParams(
        filepath=upload_path,
        page_range=page_range,
        force_ocr=force_ocr,
        paginate_output=paginate_output,
        output_format=output_format,
    )
    results = await _convert_pdf(params)
    os.remove(upload_path)
    return results


async def _run_job(job_id: str):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_executor(), lambda: _process_job(job_id))


@app.post("/jobs", response_model=JobResponse)
async def create_job(file: UploadFile = File(...)):
    job_id = uuid4().hex
    job_dir = os.path.join(JOB_DIRECTORY, job_id)
    os.makedirs(job_dir, exist_ok=True)

    input_path = os.path.join(job_dir, file.filename)
    file_contents = await file.read()
    with open(input_path, "wb") as f:
        f.write(file_contents)

    job_record = {
        "job_id": job_id,
        "filename": file.filename,
        "status": JobStatus.queued,
        "error": None,
        "result": None,
        "input_path": input_path,
        "output_dir": job_dir,
        "progress": 0,
        "stage": "Queued",
        "detail": None,
        "processed_pages": 0,
        "total_pages": None,
    }

    with _job_lock():
        _job_store()[job_id] = job_record

    asyncio.create_task(_run_job(job_id))

    return JobResponse(
        job_id=job_id,
        filename=file.filename,
        status=JobStatus.queued,
        output_dir=job_dir,
        progress=0,
        stage="Queued",
        processed_pages=0,
        total_pages=None,
    )


@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobResponse(
        job_id=job["job_id"],
        filename=job["filename"],
        status=job["status"],
        error=job.get("error"),
        result=job.get("result"),
        output_dir=job.get("output_dir"),
        progress=job.get("progress", 0),
        stage=job.get("stage"),
        detail=job.get("detail"),
        processed_pages=job.get("processed_pages"),
        total_pages=job.get("total_pages"),
    )


@app.post("/jobs/import-json", response_model=JobResponse)
async def import_json(file: UploadFile = File(...)):
    filename = file.filename or "document.json"
    if not filename.lower().endswith(".json"):
        raise HTTPException(status_code=400, detail="Only .json files are supported.")

    job_id = uuid4().hex
    job_dir = os.path.join(JOB_DIRECTORY, job_id)
    os.makedirs(job_dir, exist_ok=True)

    dest_path = os.path.join(job_dir, filename)
    contents = await file.read()
    with open(dest_path, "wb") as f:
        f.write(contents)

    html_path = None
    try:
        html_path = render_json_document(Path(dest_path))
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Failed to render JSON: {exc}")

    json_rel = os.path.relpath(dest_path, start=job_dir)
    html_rel = (
        os.path.relpath(html_path, start=job_dir) if html_path and html_path.exists() else None
    )

    job_record = {
        "job_id": job_id,
        "filename": filename,
        "status": JobStatus.completed,
        "error": None,
        "result": JobResult(
            json_path=json_rel,
            html_path=html_rel,
            metadata_path=None,
            assets=[],
        ),
        "input_path": dest_path,
        "output_dir": job_dir,
        "progress": 100,
        "stage": "Completed",
        "detail": None,
        "processed_pages": None,
        "total_pages": None,
    }

    with _job_lock():
        _job_store()[job_id] = job_record

    return JobResponse(
        job_id=job_id,
        filename=filename,
        status=JobStatus.completed,
        result=job_record["result"],
        output_dir=job_dir,
        progress=100,
        stage="Completed",
    )


@click.command()
@click.option("--port", type=int, default=8000, help="Port to run the server on")
@click.option("--host", type=str, default="127.0.0.1", help="Host to run the server on")
def server_cli(port: int, host: str):
    import uvicorn

    # Run the server
    uvicorn.run(
        app,
        host=host,
        port=port,
    )


@app.post("/api/upload", response_model=JobResponse)
async def api_upload(file: UploadFile = File(...)):
    return await create_job(file)


@app.post("/api/import-json", response_model=JobResponse)
async def api_import_json(file: UploadFile = File(...)):
    return await import_json(file)


@app.get("/api/jobs/{job_id}/status", response_model=JobResponse)
async def api_job_status(job_id: str):
    return await get_job(job_id)


@app.get("/api/jobs/{job_id}/file")
async def api_job_file(job_id: str, type: str, name: Optional[str] = None):
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != JobStatus.completed:
        raise HTTPException(status_code=409, detail="Job is not complete")

    output_dir = job.get("output_dir")
    if not output_dir:
        raise HTTPException(status_code=404, detail="Job output unavailable")

    result = job.get("result")
    relative_path = None

    if type == "json" and result and result.json_path:
        relative_path = result.json_path
    elif type == "html" and result and result.html_path:
        relative_path = result.html_path
    elif type == "metadata" and result and result.metadata_path:
        relative_path = result.metadata_path
    elif type == "asset":
        if not name:
            raise HTTPException(status_code=400, detail="Asset name required")
        relative_path = name
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported type {type}")

    if not relative_path:
        raise HTTPException(status_code=404, detail="Requested file not found")

    absolute_path = os.path.abspath(os.path.join(output_dir, relative_path))
    output_dir_abs = os.path.abspath(output_dir)
    if not absolute_path.startswith(output_dir_abs):
        raise HTTPException(status_code=400, detail="Invalid file path")
    if not os.path.exists(absolute_path):
        raise HTTPException(status_code=404, detail="File missing on server")

    return FileResponse(absolute_path)
