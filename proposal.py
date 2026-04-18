"""
Proposal Generator
===================
Two modes:
  1. Upwork Proposal — from a job description, generates a 4-section proposal
  2. Client Proposal — from discovery call notes/transcript, generates a formal proposal

Usage (CLI):
    uv run python proposal.py upwork "Job description text here"
    uv run python proposal.py client --notes "Call notes or transcript"
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# UPWORK PROPOSAL — 4-Section Framework from AIAS+
# ---------------------------------------------------------------------------

UPWORK_SYSTEM_PROMPT = """You are an expert Upwork proposal writer for an AI automation agency.
You generate proposals following a proven 4-section framework that is optimized for Upwork's
compressed preview format (clients only see the first ~150 characters before deciding to click).

The 4 sections:

**Section 1 — Introduction & Social Proof (THE MOST IMPORTANT LINE)**
Format: "[Name] here — [X] automations built in n8n/Make, [Y credential or niche experience]."
This line MUST be under 150 characters and load-bearing. It appears in the Upwork preview.
It should immediately signal: you read the job AND you're confident you can solve it.

**Section 2 — Loom Video Link**
A standalone line with the Loom placeholder. Keep it isolated so it's impossible to miss:
"I recorded a short walkthrough of how I'd approach this — [LOOM_LINK_HERE]"

**Section 3 — Problem-Solving Paragraph**
This is rewritten from scratch for every proposal. It must:
- Restate the client's problem in your own words (proves you read the job)
- Describe the technical approach end-to-end (triggers, data flow, integrations, output)
- Reference a similar past build when possible
- End with ONE smart clarifying question (shows depth of thinking, helps scope the work)

**Section 4 — Clean Closing**
Short, confident, no fluff. No offers for meetings. No pitches for long-term work.
Just a clear signal you're ready to deliver this specific project.

RULES:
- Keep total proposal under 200 words (clients scan, they don't read essays)
- Never apologize for experience level ("I'm new", "I'm still learning" = instant credibility kill)
- Never use filler phrases ("I'd love to help", "I'm excited about this opportunity")
- Be specific to THIS job — generic proposals get ignored
- Sound human, not robotic
"""

UPWORK_USER_TEMPLATE = """Generate an Upwork proposal for the following job description.

**My Profile:**
{profile}

**Job Description:**
{job_description}

Generate the 4-section proposal. Use [LOOM_LINK_HERE] as the Loom placeholder.
Keep the opening line under 150 characters.
Total proposal under 200 words."""

# ---------------------------------------------------------------------------
# CLIENT PROPOSAL — From Discovery Call to Formal Proposal
# ---------------------------------------------------------------------------

CLIENT_SYSTEM_PROMPT = """You are a senior AI automation consultant who creates professional
project proposals after discovery calls. You transform call notes or transcripts into
structured, branded proposals that inspire confidence and clearly define scope.

Your proposals follow this structure:

## Project Overview
1-2 paragraph summary of the client's situation and what they need.

## Objectives
Bullet list of specific, measurable goals this project will achieve.

## Scope of Work
Detailed breakdown of what will be built/delivered. Each item should be specific enough
to serve as "definition of done" — no ambiguity. Group by phase if the project is large.

## Technical Approach
How the solution will work — tools, integrations, data flows, architecture.
Keep it accessible (the client may not be technical).

## Timeline & Milestones
Phase-based timeline with clear deliverables at each milestone.

## Investment
Use the ROI anchoring approach:
- State the estimated value/savings the automation will create
- Present pricing as a percentage of that value (20-30% of first-year ROI)
- Offer tiered packages if appropriate (Starter / Growth / Scale)
- Include maintenance/retainer options

## Next Steps
Clear call-to-action — what happens after they approve.

RULES:
- Frame everything around business outcomes, not technical features
- Use the "Three Profit Levers" lens: more customers, higher customer value, cut costs
- Be specific — vague proposals kill deals
- Include functionality requirements that are realistic and deliverable
- Never over-promise or include KPIs you can't guarantee
"""

CLIENT_USER_TEMPLATE = """Generate a professional project proposal from the following discovery call information.

**Client Name:** {client_name}
**Company:** {company}
**My Business Name:** {my_business}

**Discovery Call Notes / Transcript:**
{notes}

**Additional Context (if any):**
{context}

Generate a complete, professional proposal document in markdown format.
Include realistic timeline estimates and ROI-anchored pricing."""


def load_profile() -> dict:
    """Load saved user profile if it exists."""
    profile_path = Path(__file__).parent / "profile.json"
    if profile_path.exists():
        return json.loads(profile_path.read_text())
    return {}


def save_profile(profile: dict):
    """Save user profile for reuse."""
    profile_path = Path(__file__).parent / "profile.json"
    profile_path.write_text(json.dumps(profile, indent=2))


def generate_upwork_proposal(
    job_description: str,
    profile: dict,
    client: OpenAI,
    model: str,
) -> str:
    """Generate an Upwork proposal from a job description."""
    profile_text = format_profile(profile)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": UPWORK_SYSTEM_PROMPT},
            {"role": "user", "content": UPWORK_USER_TEMPLATE.format(
                profile=profile_text,
                job_description=job_description,
            )},
        ],
        temperature=0.4,
        max_tokens=1500,
    )
    return response.choices[0].message.content


def generate_client_proposal(
    client_name: str,
    company: str,
    notes: str,
    context: str,
    my_business: str,
    client: OpenAI,
    model: str,
) -> str:
    """Generate a formal client proposal from discovery call notes."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CLIENT_SYSTEM_PROMPT},
            {"role": "user", "content": CLIENT_USER_TEMPLATE.format(
                client_name=client_name,
                company=company,
                my_business=my_business,
                notes=notes,
                context=context,
            )},
        ],
        temperature=0.3,
        max_tokens=3000,
    )
    return response.choices[0].message.content


def format_profile(profile: dict) -> str:
    """Format profile dict into readable text."""
    if not profile:
        return "No profile set. Use generic AI automation freelancer positioning."
    parts = []
    if profile.get("name"):
        parts.append(f"Name: {profile['name']}")
    if profile.get("title"):
        parts.append(f"Title: {profile['title']}")
    if profile.get("automations_built"):
        parts.append(f"Automations built: {profile['automations_built']}")
    if profile.get("tools"):
        parts.append(f"Tools: {', '.join(profile['tools'])}")
    if profile.get("industries"):
        parts.append(f"Industry experience: {', '.join(profile['industries'])}")
    if profile.get("proof_points"):
        parts.append(f"Proof points: {'; '.join(profile['proof_points'])}")
    return "\n".join(parts)


def save_proposal(content: str, proposal_type: str, label: str) -> Path:
    """Save proposal to output directory."""
    safe_label = label.lower().replace(" ", "_")[:40]
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"proposal_{proposal_type}_{safe_label}_{date_str}.md"
    filepath = OUTPUT_DIR / filename
    filepath.write_text(content)
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Proposal Generator")
    subparsers = parser.add_subparsers(dest="mode", help="Proposal type")

    # Upwork mode
    up_parser = subparsers.add_parser("upwork", help="Generate Upwork proposal")
    up_parser.add_argument("job", help="Job description (text or file path)")

    # Client mode
    cl_parser = subparsers.add_parser("client", help="Generate client proposal")
    cl_parser.add_argument("--name", required=True, help="Client name")
    cl_parser.add_argument("--company", required=True, help="Client company")
    cl_parser.add_argument("--notes", required=True, help="Call notes (text or file path)")
    cl_parser.add_argument("--context", default="", help="Additional context")
    cl_parser.add_argument("--business", default="PalmettoDevs LLC", help="Your business name")

    # Profile setup
    subparsers.add_parser("setup", help="Set up your profile")

    args = parser.parse_args()

    if not args.mode:
        parser.print_help()
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key and args.mode != "setup":
        print("OPENAI_API_KEY not set.")
        sys.exit(1)

    if args.mode == "setup":
        print("\nProfile Setup")
        print("=" * 40)
        profile = load_profile()
        profile["name"] = input(f"Your name [{profile.get('name', '')}]: ").strip() or profile.get("name", "")
        profile["title"] = input(f"Title [{profile.get('title', 'AI Automation Consultant')}]: ").strip() or profile.get("title", "AI Automation Consultant")
        profile["automations_built"] = input(f"# automations built [{profile.get('automations_built', '')}]: ").strip() or profile.get("automations_built", "")
        tools = input(f"Tools (comma-sep) [{','.join(profile.get('tools', ['n8n']))}]: ").strip()
        profile["tools"] = [t.strip() for t in tools.split(",")] if tools else profile.get("tools", ["n8n"])
        industries = input(f"Industries (comma-sep) [{','.join(profile.get('industries', []))}]: ").strip()
        profile["industries"] = [i.strip() for i in industries.split(",")] if industries else profile.get("industries", [])
        proof = input(f"Proof points (semicolon-sep) [{'; '.join(profile.get('proof_points', []))}]: ").strip()
        profile["proof_points"] = [p.strip() for p in proof.split(";")] if proof else profile.get("proof_points", [])
        save_profile(profile)
        print(f"\nProfile saved to profile.json")
        return

    openai_client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if args.mode == "upwork":
        job_desc = args.job
        if Path(job_desc).exists():
            job_desc = Path(job_desc).read_text()

        profile = load_profile()
        print(f"\nGenerating Upwork proposal...")
        proposal = generate_upwork_proposal(job_desc, profile, openai_client, model)
        filepath = save_proposal(proposal, "upwork", job_desc[:30])
        print(f"\n{proposal}")
        print(f"\nSaved to: {filepath}")

    elif args.mode == "client":
        notes = args.notes
        if Path(notes).exists():
            notes = Path(notes).read_text()

        print(f"\nGenerating client proposal...")
        proposal = generate_client_proposal(
            args.name, args.company, notes, args.context,
            args.business, openai_client, model,
        )
        filepath = save_proposal(proposal, "client", f"{args.company}_{args.name}")
        print(f"\n{proposal}")
        print(f"\nSaved to: {filepath}")


if __name__ == "__main__":
    main()
