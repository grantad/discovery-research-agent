"""
Web UI for Discovery Call Research Agent
==========================================
FastAPI server with SSE streaming for real-time progress updates.

Usage:
    uv run python server.py
    # Then open http://localhost:8000
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

import markdown
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI

from agent import (
    research_prospect,
    generate_briefing,
    save_briefing,
    BRIEFING_TEMPLATE,
)
from proposal import (
    generate_upwork_proposal,
    generate_client_proposal,
    load_profile,
    save_profile,
    save_proposal,
)
from upwork_scraper import normalize_job_url, fetch_job, format_job_for_proposal

load_dotenv()

app = FastAPI(title="Discovery Research Agent")

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / "index.html").read_text()


@app.get("/api/history")
async def history():
    """List past briefings."""
    briefings = []
    for f in sorted(OUTPUT_DIR.glob("briefing_*.md"), reverse=True):
        content = f.read_text()
        # Extract prospect and company from first lines
        lines = content.split("\n")
        prospect = ""
        company = ""
        date = ""
        for line in lines[:6]:
            if line.startswith("**Prospect:**"):
                prospect = line.replace("**Prospect:**", "").strip()
            elif line.startswith("**Company:**"):
                company = line.replace("**Company:**", "").strip()
            elif line.startswith("**Date:**"):
                date = line.replace("**Date:**", "").strip()
        briefings.append({
            "filename": f.name,
            "prospect": prospect,
            "company": company,
            "date": date,
        })
    return briefings


@app.get("/api/briefing/{filename}")
async def get_briefing(filename: str):
    """Get a specific briefing as HTML."""
    filepath = OUTPUT_DIR / filename
    if not filepath.exists() or not filepath.name.startswith("briefing_"):
        return JSONResponse({"error": "Not found"}, status_code=404)
    md_content = filepath.read_text()
    html = markdown.markdown(md_content, extensions=["extra", "sane_lists"])
    return {"html": html, "markdown": md_content}


@app.post("/api/upwork/fetch")
async def fetch_upwork_job(request: Request):
    """Fetch an Upwork job description from URL or ID."""
    data = await request.json()
    job_input = data.get("url", "").strip()
    if not job_input:
        return JSONResponse({"error": "URL or job ID required"}, status_code=400)

    job_url = normalize_job_url(job_input)
    loop = asyncio.get_event_loop()
    job = await loop.run_in_executor(None, lambda: asyncio.run(fetch_job(job_url)))

    if not job:
        return JSONResponse({"error": "Could not fetch job. You may need to run: uv run python upwork_scraper.py --login"}, status_code=422)

    return {
        "title": job.get("title", ""),
        "description": format_job_for_proposal(job),
        "budget": job.get("budget", ""),
        "url": job.get("url", ""),
    }


@app.post("/api/research")
async def research(request: Request):
    """Run research and stream progress via SSE."""
    data = await request.json()
    name = data.get("name", "").strip()
    company = data.get("company", "").strip()
    linkedin = data.get("linkedin", "").strip() or None

    if not name or not company:
        return JSONResponse({"error": "Name and company are required"}, status_code=400)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return JSONResponse({"error": "OPENAI_API_KEY not configured"}, status_code=500)

    async def event_stream():
        loop = asyncio.get_event_loop()

        # Step 1: Research
        yield f"data: {json.dumps({'stage': 'research', 'message': 'Starting web research...'})}\n\n"

        progress_updates = []

        def on_progress(stage, current, total, detail):
            progress_updates.append({
                "stage": stage,
                "current": current,
                "total": total,
                "detail": detail,
            })

        research_data = await loop.run_in_executor(
            None, lambda: research_prospect(name, company, linkedin, on_progress)
        )

        total_results = sum(len(v) for v in research_data.values())
        yield f"data: {json.dumps({'stage': 'research_done', 'message': f'Found {total_results} results from {len(research_data)} searches'})}\n\n"

        # Step 2: Generate briefing
        yield f"data: {json.dumps({'stage': 'generating', 'message': 'AI is analyzing research and generating briefing...'})}\n\n"

        client = OpenAI(api_key=api_key)
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        analysis = await loop.run_in_executor(
            None, lambda: generate_briefing(name, company, research_data, linkedin, client, model)
        )

        linkedin_line = f"**LinkedIn:** {linkedin}" if linkedin else ""
        briefing_md = BRIEFING_TEMPLATE.format(
            name=name,
            company=company,
            date=datetime.now().strftime("%B %d, %Y"),
            linkedin_line=linkedin_line,
            analysis=analysis,
        )

        # Step 3: Save
        filepath = await loop.run_in_executor(
            None, lambda: save_briefing(briefing_md, name, company)
        )

        briefing_html = markdown.markdown(briefing_md, extensions=["extra", "sane_lists"])

        yield f"data: {json.dumps({'stage': 'complete', 'message': 'Briefing ready!', 'html': briefing_html, 'markdown': briefing_md, 'filename': filepath.name})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/proposals", response_class=HTMLResponse)
async def proposals_page():
    return (STATIC_DIR / "proposals.html").read_text()


@app.get("/api/profile")
async def get_profile():
    return load_profile()


@app.post("/api/profile")
async def update_profile(request: Request):
    data = await request.json()
    save_profile(data)
    return {"status": "ok"}


@app.get("/api/proposals")
async def list_proposals():
    """List past proposals."""
    proposals = []
    for f in sorted(OUTPUT_DIR.glob("proposal_*.md"), reverse=True):
        content = f.read_text()
        lines = content.split("\n")
        title = lines[0].replace("#", "").strip() if lines else f.name
        ptype = "upwork" if "upwork" in f.name else "client"
        proposals.append({
            "filename": f.name,
            "title": title,
            "type": ptype,
            "modified": datetime.fromtimestamp(f.stat().st_mtime).strftime("%B %d, %Y %H:%M"),
        })
    return proposals


@app.get("/api/proposal/{filename}")
async def get_proposal(filename: str):
    filepath = OUTPUT_DIR / filename
    if not filepath.exists() or not filepath.name.startswith("proposal_"):
        return JSONResponse({"error": "Not found"}, status_code=404)
    md_content = filepath.read_text()
    html = markdown.markdown(md_content, extensions=["extra", "sane_lists"])
    return {"html": html, "markdown": md_content}


@app.post("/api/proposal/upwork")
async def create_upwork_proposal(request: Request):
    data = await request.json()
    job_description = data.get("job_description", "").strip()
    if not job_description:
        return JSONResponse({"error": "Job description is required"}, status_code=400)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return JSONResponse({"error": "OPENAI_API_KEY not configured"}, status_code=500)

    async def event_stream():
        loop = asyncio.get_event_loop()
        yield f"data: {json.dumps({'stage': 'generating', 'message': 'Analyzing job description...'})}\n\n"

        profile = load_profile()
        openai_client = OpenAI(api_key=api_key)
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        proposal = await loop.run_in_executor(
            None, lambda: generate_upwork_proposal(job_description, profile, openai_client, model)
        )

        filepath = await loop.run_in_executor(
            None, lambda: save_proposal(proposal, "upwork", job_description[:30])
        )

        html = markdown.markdown(proposal, extensions=["extra", "sane_lists"])
        yield f"data: {json.dumps({'stage': 'complete', 'message': 'Proposal ready!', 'html': html, 'markdown': proposal, 'filename': filepath.name})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/proposal/client")
async def create_client_proposal(request: Request):
    data = await request.json()
    client_name = data.get("client_name", "").strip()
    company = data.get("company", "").strip()
    notes = data.get("notes", "").strip()
    context = data.get("context", "").strip()
    my_business = data.get("my_business", "PalmettoDevs LLC").strip()

    if not notes:
        return JSONResponse({"error": "Call notes are required"}, status_code=400)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return JSONResponse({"error": "OPENAI_API_KEY not configured"}, status_code=500)

    async def event_stream():
        loop = asyncio.get_event_loop()
        yield f"data: {json.dumps({'stage': 'generating', 'message': 'Generating proposal from call notes...'})}\n\n"

        openai_client = OpenAI(api_key=api_key)
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        proposal = await loop.run_in_executor(
            None, lambda: generate_client_proposal(
                client_name, company, notes, context, my_business, openai_client, model
            )
        )

        filepath = await loop.run_in_executor(
            None, lambda: save_proposal(proposal, "client", f"{company}_{client_name}")
        )

        html = markdown.markdown(proposal, extensions=["extra", "sane_lists"])
        yield f"data: {json.dumps({'stage': 'complete', 'message': 'Proposal ready!', 'html': html, 'markdown': proposal, 'filename': filepath.name})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    print("\n  Discovery Research Agent + Proposal Generator")
    print("  http://localhost:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
