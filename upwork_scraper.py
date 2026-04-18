"""
Upwork Job Scraper
===================
Fetches job descriptions from Upwork using a saved browser session.

Usage:
    uv run python upwork_scraper.py --login          # First time: log in to Upwork
    uv run python upwork_scraper.py <job_url_or_id>  # Fetch a job description
"""

import asyncio
import json
import re
import sys
from pathlib import Path

from playwright.async_api import async_playwright

AUTH_STATE_FILE = Path(__file__).parent / "upwork_auth.json"


def normalize_job_url(input_str: str) -> str:
    """Convert a job URL or ID to a full Upwork job URL."""
    input_str = input_str.strip()
    # Already a full URL
    if input_str.startswith("http"):
        return input_str
    # Just the job ID like ~021915024814498495
    if input_str.startswith("~"):
        return f"https://www.upwork.com/jobs/{input_str}"
    # Bare ID without tilde
    if re.match(r'^0\d{17,}$', input_str):
        return f"https://www.upwork.com/jobs/~{input_str}"
    # Slug-style URL path
    return f"https://www.upwork.com/jobs/{input_str}"


async def login():
    """Open browser for manual Upwork login, save auth state."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            viewport={"width": 1440, "height": 900},
        )
        await context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', { get: () => undefined });"
        )

        page = await context.new_page()
        await page.goto("https://www.upwork.com/ab/account-security/login", wait_until="domcontentloaded", timeout=30000)

        print("\n🔐 Please log in to Upwork in the browser window.")
        print("   After logging in, the script will detect your dashboard and save automatically.\n")

        try:
            # Wait for redirect to dashboard/feed after login (up to 5 min)
            await page.wait_for_url("**/nx/**", timeout=300000)
            print("✅ Detected Upwork dashboard!")
            await page.wait_for_timeout(3000)
        except Exception:
            print("⏰ Timed out. Saving current state anyway...")

        storage = await context.storage_state()
        AUTH_STATE_FILE.write_text(json.dumps(storage, indent=2))
        print(f"✅ Auth state saved to {AUTH_STATE_FILE}")

        await browser.close()


async def fetch_job(job_url: str) -> dict | None:
    """Fetch job details from an Upwork job URL using saved auth."""
    if not AUTH_STATE_FILE.exists():
        print("❌ No Upwork auth state. Run with --login first.")
        return None

    auth_state = json.loads(AUTH_STATE_FILE.read_text())

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = await browser.new_context(
            storage_state=auth_state,
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            viewport={"width": 1440, "height": 900},
        )
        await context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', { get: () => undefined });"
        )

        page = await context.new_page()

        try:
            await page.goto(job_url, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(6000)

            # Check if Cloudflare blocked us
            body_text = await page.evaluate("() => document.body.innerText")
            if len(body_text) < 200 or "Cloudflare" in body_text:
                # Try waiting longer for challenge
                print("  Waiting for Cloudflare challenge...")
                await page.wait_for_timeout(10000)
                body_text = await page.evaluate("() => document.body.innerText")

            if len(body_text) < 200:
                print("❌ Cloudflare still blocking. Try running --login again.")
                await browser.close()
                return None

            # Extract job data from the page
            job_data = await page.evaluate("""() => {
                const body = document.body.innerText;

                // Try to find title
                const h1 = document.querySelector('h1');
                const title = h1 ? h1.textContent.trim() : null;

                // Try to get structured data
                const scripts = document.querySelectorAll('script[type="application/ld+json"]');
                let structuredData = null;
                for (const s of scripts) {
                    try {
                        const d = JSON.parse(s.textContent);
                        if (d['@type'] === 'JobPosting' || d.title || d.description) {
                            structuredData = d;
                            break;
                        }
                    } catch {}
                }

                return {
                    title: title,
                    body: body,
                    url: window.location.href,
                    structuredData: structuredData,
                };
            }""")

            await browser.close()

            # Parse the body text to extract key fields
            result = parse_job_page(job_data)
            return result

        except Exception as e:
            print(f"❌ Error: {e}")
            await browser.close()
            return None


def parse_job_page(job_data: dict) -> dict:
    """Parse the raw page data into structured job fields."""
    body = job_data.get("body", "")
    structured = job_data.get("structuredData")

    result = {
        "title": job_data.get("title", ""),
        "url": job_data.get("url", ""),
        "description": "",
        "budget": "",
        "skills": [],
        "client_info": "",
        "raw_text": body,
    }

    # If we have structured data, use it
    if structured:
        result["title"] = structured.get("title", result["title"])
        result["description"] = structured.get("description", "")
        if structured.get("baseSalary"):
            sal = structured["baseSalary"]
            if isinstance(sal, dict):
                val = sal.get("value", {})
                if isinstance(val, dict):
                    result["budget"] = f"${val.get('minValue', '')} - ${val.get('maxValue', '')}"
                else:
                    result["budget"] = str(val)

    # Fallback: extract description from body text
    if not result["description"]:
        result["description"] = body

    return result


def format_job_for_proposal(job: dict) -> str:
    """Format extracted job data as clean text for the proposal generator."""
    parts = []
    if job.get("title"):
        parts.append(f"Job Title: {job['title']}")
    if job.get("budget"):
        parts.append(f"Budget: {job['budget']}")
    if job.get("url"):
        parts.append(f"URL: {job['url']}")
    parts.append("")
    parts.append(job.get("description") or job.get("raw_text", ""))
    return "\n".join(parts)


async def main():
    if "--login" in sys.argv:
        await login()
        print("\n✅ Login complete! Now you can fetch jobs:")
        print('   uv run python upwork_scraper.py "https://www.upwork.com/jobs/~012345"')
        return

    if len(sys.argv) < 2:
        print("Usage:")
        print("  uv run python upwork_scraper.py --login")
        print('  uv run python upwork_scraper.py "https://www.upwork.com/jobs/~012345"')
        print('  uv run python upwork_scraper.py "~012345"')
        return

    job_input = sys.argv[1]
    job_url = normalize_job_url(job_input)
    print(f"📋 Fetching: {job_url}")

    job = await fetch_job(job_url)
    if job:
        print(f"\n{'='*60}")
        print(f"Title: {job.get('title', 'N/A')}")
        print(f"Budget: {job.get('budget', 'N/A')}")
        print(f"URL: {job.get('url', 'N/A')}")
        print(f"{'='*60}")
        desc = job.get("description") or job.get("raw_text", "")
        print(desc[:2000])


if __name__ == "__main__":
    asyncio.run(main())
