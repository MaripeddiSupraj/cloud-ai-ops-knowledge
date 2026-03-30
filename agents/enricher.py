#!/usr/bin/env python3
"""
agents/enricher.py
------------------
CrewAI enrichment agent powered by a local Ollama model.

The agent reads LinkedIn post content and enhances it with:
  - A concise summary
  - Why it matters (broader context for cloud/devops/AI practitioners)
  - Key technical takeaways (bullet points)
  - Reusable ideas, patterns, or tools
  - A confirmed or refined category suggestion

The original post content is NEVER modified or removed.
"""
from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class EnrichmentResult:
    summary: str
    why_it_matters: str
    key_takeaways: list[str]
    reusable_ideas: list[str]
    suggested_category: str
    model_used: str


# ─────────────────────────────────────────────
#  Prompt template
# ─────────────────────────────────────────────

ENRICHMENT_PROMPT = textwrap.dedent("""
You are a Knowledge Enrichment Specialist focused on Cloud, DevOps, AI, and Security.

A LinkedIn post has been captured for a personal knowledge base. Your job is to ENHANCE it
without removing or altering a single word of the original post.

--- ORIGINAL POST ---
{post_text}
--- END POST ---

Post URL: {url}
Auto-detected category: {auto_category}

Respond with EXACTLY the following structured format (use the exact section headers):

## SUMMARY
[Write 2-3 sentences explaining what this post is about. Be specific and technical.]

## WHY IT MATTERS
[Write 2-4 sentences explaining the broader significance. Why should a cloud/devops/AI engineer care about this?]

## KEY TAKEAWAYS
- [Specific actionable takeaway 1]
- [Specific actionable takeaway 2]
- [Specific actionable takeaway 3]
- [Add more if genuinely useful, do not pad]

## REUSABLE IDEAS
- [A pattern, tool, workflow, or concept that can be immediately applied]
- [Another reusable idea]
- [Add more if genuinely useful]

## CATEGORY CONFIRMATION
[Either confirm the auto-detected category "{auto_category}" or suggest a better one from this list:
cloud/aws | cloud/azure | cloud/gcp | cloud/general |
devops/ci-cd | devops/k8s | devops/terraform | devops/obs |
ai/llm | ai/agents | ai/mlops |
security/iam | security/devsecops | security/compliance |
career | productivity | engineering | data | learning]
Respond with ONLY the category slug, nothing else (e.g. "cloud/aws")

Do not add any text outside these sections. Do not repeat the original post.
""")


# ─────────────────────────────────────────────
#  Ollama-based enrichment (no CrewAI overhead)
# ─────────────────────────────────────────────

def _enrich_with_ollama(post_text: str, url: str, auto_category: str, model: str, base_url: str) -> EnrichmentResult:
    """Call Ollama directly using the ollama Python client."""
    try:
        import ollama as ollama_client
    except ImportError:
        raise ImportError(
            "ollama package not installed. Run: pip install ollama"
        )

    prompt = ENRICHMENT_PROMPT.format(
        post_text=post_text,
        url=url,
        auto_category=auto_category,
    )

    client = ollama_client.Client(host=base_url)
    response = client.generate(model=model, prompt=prompt)
    raw = response["response"] if isinstance(response, dict) else response.response
    return _parse_enrichment_response(raw, auto_category, model)


# ─────────────────────────────────────────────
#  CrewAI-based enrichment (richer agent loop)
# ─────────────────────────────────────────────

def _enrich_with_crewai(post_text: str, url: str, auto_category: str, model: str, base_url: str) -> EnrichmentResult:
    """Use CrewAI + Ollama LLM for enrichment."""
    try:
        from crewai import Agent, Task, Crew, Process
        from langchain_ollama import OllamaLLM
    except ImportError:
        raise ImportError(
            "crewai / langchain-ollama not installed. Run: pip install -r requirements.txt"
        )

    llm = OllamaLLM(model=model, base_url=base_url)

    enricher = Agent(
        role="Knowledge Enrichment Specialist",
        goal=(
            "Enhance raw LinkedIn post content with structured insights for a cloud/devops/AI "
            "knowledge base. Never remove or alter the original content."
        ),
        backstory=(
            "You are a senior cloud architect and DevOps engineer with 15 years of experience "
            "across AWS, Azure, GCP, Kubernetes, and AI/ML platforms. You distill technical "
            "content into actionable insights for engineering teams."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    prompt = ENRICHMENT_PROMPT.format(
        post_text=post_text,
        url=url,
        auto_category=auto_category,
    )

    task = Task(
        description=prompt,
        agent=enricher,
        expected_output=(
            "Structured enrichment with SUMMARY, WHY IT MATTERS, KEY TAKEAWAYS, "
            "REUSABLE IDEAS, and CATEGORY CONFIRMATION sections."
        ),
    )

    crew = Crew(agents=[enricher], tasks=[task], process=Process.sequential, verbose=False)
    result = crew.kickoff()
    raw = str(result)
    return _parse_enrichment_response(raw, auto_category, model)


# ─────────────────────────────────────────────
#  Response parser
# ─────────────────────────────────────────────

def _extract_section(text: str, header: str, next_headers: list[str]) -> str:
    """Extract text between a section header and the next header."""
    pattern = f"## {header}"
    start = text.find(pattern)
    if start == -1:
        return ""
    start = text.find("\n", start) + 1
    end = len(text)
    for nxt in next_headers:
        idx = text.find(f"## {nxt}", start)
        if idx != -1 and idx < end:
            end = idx
    return text[start:end].strip()


def _parse_bullets(section_text: str) -> list[str]:
    """Parse bullet list items from a section."""
    lines = section_text.splitlines()
    bullets = []
    for line in lines:
        line = line.strip()
        if line.startswith("- ") or line.startswith("* "):
            bullets.append(line[2:].strip())
        elif line.startswith("• "):
            bullets.append(line[2:].strip())
    return bullets


def _parse_enrichment_response(raw: str, auto_category: str, model: str) -> EnrichmentResult:
    """Parse the structured LLM response into an EnrichmentResult."""
    all_headers = ["SUMMARY", "WHY IT MATTERS", "KEY TAKEAWAYS", "REUSABLE IDEAS", "CATEGORY CONFIRMATION"]

    summary = _extract_section(raw, "SUMMARY", all_headers[1:])
    why = _extract_section(raw, "WHY IT MATTERS", all_headers[2:])
    takeaways_text = _extract_section(raw, "KEY TAKEAWAYS", all_headers[3:])
    reusable_text = _extract_section(raw, "REUSABLE IDEAS", all_headers[4:])
    category_raw = _extract_section(raw, "CATEGORY CONFIRMATION", [])

    # Clean category — take first word/slug-like token
    valid_cats = {
        "cloud/aws", "cloud/azure", "cloud/gcp", "cloud/general",
        "devops/ci-cd", "devops/k8s", "devops/terraform", "devops/obs",
        "ai/llm", "ai/agents", "ai/mlops",
        "security/iam", "security/devsecops", "security/compliance",
        "career", "productivity", "engineering", "data", "learning",
    }
    category_raw = category_raw.lower().strip().split("\n")[0].split(" ")[0].strip("\"'`")
    confirmed_category = category_raw if category_raw in valid_cats else auto_category

    return EnrichmentResult(
        summary=summary or "No summary generated.",
        why_it_matters=why or "No context generated.",
        key_takeaways=_parse_bullets(takeaways_text) or ["See original post for details."],
        reusable_ideas=_parse_bullets(reusable_text) or ["See original post for details."],
        suggested_category=confirmed_category,
        model_used=model,
    )


# ─────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────

def enrich_content(
    post_text: str,
    url: str,
    auto_category: str,
    use_crewai: bool = False,
) -> Optional[EnrichmentResult]:
    """
    Enrich post content using the local Ollama model.

    Args:
        post_text: The full text of the LinkedIn post / article.
        url: Original URL (for context).
        auto_category: Category inferred by keyword matching.
        use_crewai: If True, use the full CrewAI agent loop (slower).
                    If False, call Ollama directly (faster, default).

    Returns:
        EnrichmentResult or None if Ollama is unreachable.
    """
    model = os.getenv("OLLAMA_MODEL", "llama3")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    try:
        if use_crewai:
            return _enrich_with_crewai(post_text, url, auto_category, model, base_url)
        else:
            return _enrich_with_ollama(post_text, url, auto_category, model, base_url)
    except Exception as exc:
        print(f"  ⚠️  Enrichment failed (Ollama unreachable?): {exc}")
        print("  → Tip: make sure Ollama is running: ollama serve")
        return None
