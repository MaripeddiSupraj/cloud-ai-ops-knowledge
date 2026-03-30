#!/usr/bin/env python3
"""
linkedin_ingest.py
------------------
LinkedIn → GitHub Knowledge Hub pipeline.

Usage:
    python3 linkedin_ingest.py "https://www.linkedin.com/posts/..."
    python3 linkedin_ingest.py --dry-run "https://www.linkedin.com/posts/..."
    python3 linkedin_ingest.py --no-enrich "https://www.linkedin.com/posts/..."
    python3 linkedin_ingest.py --use-crewai "https://www.linkedin.com/posts/..."

What it does:
    1. Fetches LinkedIn post/article content using your credentials (.env)
    2. Determines content type: linkedin-post | linkedin-article | external-link
    3. Classifies into fine-grained category (cloud/aws, devops/k8s, ai/llm, etc.)
    4. Enriches content with a local Ollama agent (summary, takeaways, context)
    5. Writes enriched markdown to content/<category>/<slug>.md
    6. Updates registry.json and indexes/
    7. Creates a local git commit (you review, then push manually)

Requirements:
    pip install -r requirements.txt
    cp .env.example .env   # then fill in LINKEDIN_EMAIL and LINKEDIN_PASSWORD
    ollama serve           # in a separate terminal
    ollama pull llama3     # first time only
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
#  Repo paths (same as ingest.py)
# ─────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
CONTENT = ROOT / "content"
RAW = ROOT / "raw"
INDEXES = ROOT / "indexes"
REGISTRY = ROOT / "registry.json"
TAXONOMY = ROOT / "taxonomy.json"

# ─────────────────────────────────────────────
#  Data model
# ─────────────────────────────────────────────

@dataclass
class LinkedInEntry:
    title: str
    slug: str
    created_at: str
    content_type: str          # linkedin-post | linkedin-article | external-link
    category: str              # e.g. cloud/aws
    author: str
    url: str
    post_text: str             # original post — never modified
    topics: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    summary: str = ""
    why_it_matters: str = ""
    key_takeaways: list[str] = field(default_factory=list)
    reusable_ideas: list[str] = field(default_factory=list)
    enriched: bool = False
    entry_path: str = ""
    raw_path: str = ""
    source_kind: str = "url"   # compatibility with existing registry.json
    source_name: str = ""
    source_path: str = ""      # alias for url (registry compat)


# ─────────────────────────────────────────────
#  Taxonomy / keyword loading
# ─────────────────────────────────────────────

def load_taxonomy() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    if not TAXONOMY.exists():
        return {}, {}
    try:
        data = json.loads(TAXONOMY.read_text(encoding="utf-8"))
        return data.get("tools", {}), data.get("topics", {})
    except json.JSONDecodeError:
        return {}, {}


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-{2,}", "-", value)
    return value.strip("-") or "item"


def unique_slug(base_slug: str) -> str:
    existing = {row["slug"] for row in load_registry()}
    slug = base_slug
    counter = 2
    while slug in existing or list(CONTENT.glob(f"**/{slug}.md")):
        slug = f"{base_slug}-{counter}"
        counter += 1
    return slug


def load_registry() -> list[dict]:
    if not REGISTRY.exists():
        return []
    try:
        return json.loads(REGISTRY.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def save_registry(rows: list[dict]) -> None:
    REGISTRY.write_text(json.dumps(rows, indent=2), encoding="utf-8")


# ─────────────────────────────────────────────
#  LinkedIn content fetching
# ─────────────────────────────────────────────

def _linkedin_credentials() -> tuple[str, str]:
    email = os.getenv("LINKEDIN_EMAIL", "")
    password = os.getenv("LINKEDIN_PASSWORD", "")
    if not email or not password:
        print("\n❌  LinkedIn credentials not found.")
        print("   1. Copy .env.example to .env")
        print("   2. Fill in LINKEDIN_EMAIL and LINKEDIN_PASSWORD")
        sys.exit(1)
    return email, password


def _get_post_id_from_url(url: str) -> Optional[str]:
    """Extract the numeric activity/post ID from a LinkedIn URL."""
    # Handle: /posts/username_activity-1234567890-xxxx/
    match = re.search(r"activity-(\d+)", url)
    if match:
        return match.group(1)
    # Handle: /feed/update/urn:li:activity:1234567890
    match = re.search(r"activity:(\d+)", url)
    if match:
        return match.group(1)
    # Handle share id in URL
    match = re.search(r"/(\d{10,})", url)
    if match:
        return match.group(1)
    return None


def fetch_linkedin_content(url: str) -> dict:
    """
    Fetch LinkedIn post/article content using the linkedin-api library.
    Returns a dict with: text, author, content_type, title, timestamp.
    """
    try:
        from linkedin_api import Linkedin
    except ImportError:
        print("❌  linkedin-api not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    email, password = _linkedin_credentials()

    print(f"  🔐 Authenticating with LinkedIn as {email}...")
    try:
        api = Linkedin(email, password)
    except Exception as exc:
        print(f"  ❌ LinkedIn auth failed: {exc}")
        print("  → Check your LINKEDIN_EMAIL and LINKEDIN_PASSWORD in .env")
        sys.exit(1)

    parsed = urlparse(url)
    path = parsed.path.rstrip("/")

    # ── Article ──────────────────────────────
    if "/pulse/" in path:
        return _fetch_article(api, url, path)

    # ── Post (activity) ───────────────────────
    post_id = _get_post_id_from_url(url)
    if post_id:
        return _fetch_post(api, url, post_id)

    # ── Fallback: treat as external link ──────
    print("  ⚠️  Could not identify post/article ID — saving as external link.")
    return {
        "text": f"[External link — content could not be automatically fetched]\n\nURL: {url}",
        "author": "Unknown",
        "content_type": "external-link",
        "title": _title_from_url(url),
        "timestamp": datetime.now().isoformat(),
    }


def _fetch_post(api, url: str, post_id: str) -> dict:
    print(f"  📥 Fetching LinkedIn post (ID: {post_id})...")
    try:
        post = api.get_post_comments(post_id, comment_count=0)
        # The post itself is in the sharesV2 → elements structure
        # Try the direct profile_updates endpoint
        data = api._fetch(f"feed/updates/urn:li:activity:{post_id}")
        elements = data.get("elements", [])
        if elements:
            el = elements[0]
            commentary = (
                el.get("value", {})
                .get("com.linkedin.voyager.feed.render.UpdateV2", {})
                .get("commentary", {})
                .get("text", {})
                .get("text", "")
            )
            author_info = el.get("value", {}).get("com.linkedin.voyager.feed.render.UpdateV2", {}).get("actor", {})
            name = author_info.get("name", {}).get("text", "Unknown")
            if commentary:
                return {
                    "text": commentary,
                    "author": name,
                    "content_type": "linkedin-post",
                    "title": _derive_title(commentary),
                    "timestamp": datetime.now().isoformat(),
                }
    except Exception:
        pass

    # Fallback: use the search-based approach
    try:
        results = api.search_people(keyword_first=post_id, limit=1)
        _ = results  # not useful here
    except Exception:
        pass

    print("  ⚠️  Could not fetch full post text via API.")
    print("      LinkedIn may have blocked the request or the post is private.")
    print(f"      Saving minimal entry for: {url}")
    return {
        "text": (
            f"[Post content could not be fetched automatically]\n\n"
            f"URL: {url}\n\n"
            f"To add content manually, edit the generated markdown file and paste the post text "
            f"into the '## Original Post' section."
        ),
        "author": "Unknown",
        "content_type": "linkedin-post",
        "title": f"LinkedIn Post {post_id}",
        "timestamp": datetime.now().isoformat(),
    }


def _fetch_article(api, url: str, path: str) -> dict:
    print(f"  📥 Fetching LinkedIn article...")
    # Extract article ID from path like /pulse/article-slug-1234567890
    parts = path.split("/")
    slug_part = parts[-1] if parts else ""
    match = re.search(r"(\d{7,})$", slug_part)
    article_id = match.group(1) if match else slug_part

    try:
        article = api.get_article_details(article_id)
        title = article.get("title", {}).get("text", _title_from_url(url))
        content_parts = article.get("content", {}).get("multiLocaleRichText", {})
        text = ""
        if isinstance(content_parts, dict):
            text = content_parts.get("text", "")
        if not text:
            text = article.get("description", {}).get("text", "")
        author = article.get("authorName", {}).get("text", "Unknown")
        return {
            "text": text or f"[Article content not available — URL: {url}]",
            "author": author,
            "content_type": "linkedin-article",
            "title": title,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as exc:
        print(f"  ⚠️  Article fetch failed: {exc}")
        return {
            "text": f"[Article content could not be fetched]\n\nURL: {url}",
            "author": "Unknown",
            "content_type": "linkedin-article",
            "title": _title_from_url(url),
            "timestamp": datetime.now().isoformat(),
        }


def _title_from_url(url: str) -> str:
    parsed = urlparse(url)
    leaf = Path(parsed.path).name or parsed.netloc
    stem = leaf.replace("-", " ").replace("_", " ").strip()
    return stem.title() or url


def _derive_title(text: str, max_length: int = 80) -> str:
    """Derive a title from the first meaningful line of post text."""
    first_line = text.strip().splitlines()[0].strip()
    if len(first_line) <= max_length:
        return first_line
    truncated = first_line[:max_length].rsplit(" ", 1)[0]
    return truncated + "…"


# ─────────────────────────────────────────────
#  Category classification
# ─────────────────────────────────────────────

def classify_category(text: str, url: str, topic_keywords: dict[str, list[str]]) -> str:
    """
    Classify content into a fine-grained category slug.
    Uses keyword matching first; falls back to the most relevant match.
    """
    haystack = (text + " " + url).lower()
    scores: dict[str, int] = defaultdict(int)

    for category, keywords in topic_keywords.items():
        for kw in keywords:
            if kw.lower() in haystack:
                scores[category] += 1

    if not scores:
        return "learning"

    # Return category with highest keyword hit count
    best = max(scores, key=lambda k: scores[k])
    return best


def infer_tools(text: str, tool_keywords: dict[str, list[str]]) -> list[str]:
    haystack = text.lower()
    matched = []
    for tool, keywords in tool_keywords.items():
        if any(kw.lower() in haystack for kw in keywords):
            matched.append(tool)
    return sorted(matched)


def infer_topics_from_category(category: str) -> list[str]:
    """Derive topic tags from the category path."""
    parts = category.replace("/", " ").replace("-", " ").split()
    return [p for p in parts if p]


# ─────────────────────────────────────────────
#  Markdown writer
# ─────────────────────────────────────────────

def build_markdown(entry: LinkedInEntry) -> str:
    topics_yaml = ", ".join(f'"{t}"' for t in entry.topics)
    tools_yaml = ", ".join(f'"{t}"' for t in entry.tools)
    takeaways_md = "\n".join(f"- {t}" for t in entry.key_takeaways) if entry.key_takeaways else "- See original post."
    reusable_md = "\n".join(f"- {r}" for r in entry.reusable_ideas) if entry.reusable_ideas else "- See original post."

    enrichment_note = (
        f"_Enriched by Ollama `{os.getenv('OLLAMA_MODEL', 'llama3')}` · "
        f"added {entry.created_at[:10]}_"
        if entry.enriched else
        "_Enrichment skipped (--no-enrich flag or Ollama unavailable)_"
    )

    return f"""---
title: "{entry.title}"
slug: "{entry.slug}"
created_at: "{entry.created_at}"
content_type: "{entry.content_type}"
category: "{entry.category}"
author: "{entry.author}"
url: "{entry.url}"
topics: [{topics_yaml}]
tools: [{tools_yaml}]
enriched: {str(entry.enriched).lower()}
---

# {entry.title}

> **Author:** {entry.author}  
> **Source:** [{entry.content_type}]({entry.url})  
> **Category:** `{entry.category}`  
> **Added:** {entry.created_at[:10]}

---

## Original Post

> {chr(10).join('> ' + line for line in entry.post_text.splitlines())}

---

## Summary

{entry.summary or '_No summary — run without `--no-enrich` to generate._'}

## Why It Matters

{entry.why_it_matters or '_No context generated._'}

## Key Takeaways

{takeaways_md}

## Reusable Ideas

{reusable_md}

---

## Classification

| Field | Value |
|---|---|
| Category | `{entry.category}` |
| Content type | `{entry.content_type}` |
| Tools | {', '.join(f'`{t}`' for t in entry.tools) or 'unclassified'} |
| Topics | {', '.join(f'`{t}`' for t in entry.topics) or 'general'} |

{enrichment_note}
"""


# ─────────────────────────────────────────────
#  Index rebuilder (reused from ingest.py logic)
# ─────────────────────────────────────────────

def rebuild_indexes(registry_rows: list[dict]) -> None:
    INDEXES.mkdir(parents=True, exist_ok=True)
    rows_sorted = sorted(registry_rows, key=lambda r: r.get("created_at", ""), reverse=True)

    def entry_line(row: dict) -> str:
        ep = Path(row.get("entry_path", ""))
        title = row.get("title", "Untitled")
        ct = row.get("content_type", "?")
        date = row.get("created_at", "")[:10]
        try:
            rel = ep.relative_to(ROOT)
            return f"- [{title}](../{rel.as_posix()}) | `{ct}` | {date}"
        except ValueError:
            return f"- {title} | `{ct}` | {date}"

    # All entries
    all_lines = ["# All Entries\n"] + [entry_line(r) for r in rows_sorted]
    (INDEXES / "all.md").write_text("\n".join(all_lines) + "\n", encoding="utf-8")

    # By topic
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for row in rows_sorted:
        cat = row.get("category", row.get("topics", ["general"]))
        if isinstance(cat, str):
            by_topic[cat].append(row)
        elif isinstance(cat, list):
            for t in cat:
                by_topic[t].append(row)
        else:
            by_topic["general"].append(row)

    topic_lines = ["# Browse By Category\n"]
    for grp in sorted(by_topic):
        topic_lines.append(f"\n## {grp}\n")
        topic_lines.extend(entry_line(r) for r in by_topic[grp])
    (INDEXES / "by-topic.md").write_text("\n".join(topic_lines) + "\n", encoding="utf-8")

    # By tool
    by_tool: dict[str, list[dict]] = defaultdict(list)
    for row in rows_sorted:
        tools = row.get("tools", [])
        if tools:
            for t in tools:
                by_tool[t].append(row)
        else:
            by_tool["unclassified"].append(row)

    tool_lines = ["# Browse By Tool\n"]
    for grp in sorted(by_tool):
        tool_lines.append(f"\n## {grp}\n")
        tool_lines.extend(entry_line(r) for r in by_tool[grp])
    (INDEXES / "by-tool.md").write_text("\n".join(tool_lines) + "\n", encoding="utf-8")

    # By content type
    by_ct: dict[str, list[dict]] = defaultdict(list)
    for row in rows_sorted:
        by_ct[row.get("content_type", "unknown")].append(row)

    ct_lines = ["# Browse By Content Type\n"]
    for grp in sorted(by_ct):
        ct_lines.append(f"\n## {grp}\n")
        ct_lines.extend(entry_line(r) for r in by_ct[grp])
    (INDEXES / "by-content-type.md").write_text("\n".join(ct_lines) + "\n", encoding="utf-8")


# ─────────────────────────────────────────────
#  Git commit
# ─────────────────────────────────────────────

def git_commit(entry: LinkedInEntry) -> None:
    """Stage new/modified files and create a local commit."""
    try:
        subprocess.run(["git", "add", "content/", "indexes/", "registry.json"], cwd=ROOT, check=True, capture_output=True)
        msg = f"add: {entry.content_type} [{entry.category}] — {entry.title[:60]}"
        subprocess.run(["git", "commit", "-m", msg], cwd=ROOT, check=True, capture_output=True)
        print(f"\n  ✅ Local git commit created:")
        print(f"     \"{msg}\"")
        print(f"\n  👀 Review: {entry.entry_path}")
        print(f"  🚀 Push when ready: git push")
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode() if exc.stderr else ""
        if "nothing to commit" in stderr:
            print("  ℹ️  Nothing new to commit (already ingested?).")
        else:
            print(f"  ⚠️  Git commit failed: {stderr}")


# ─────────────────────────────────────────────
#  Main pipeline
# ─────────────────────────────────────────────

def run_pipeline(url: str, dry_run: bool, no_enrich: bool, use_crewai: bool) -> None:
    print(f"\n{'='*60}")
    print(f"🔗  URL: {url}")
    print(f"{'='*60}")

    tool_keywords, topic_keywords = load_taxonomy()

    # ── Step 1: Fetch LinkedIn content ─────────────────────────
    print("\n[1/5] Fetching LinkedIn content...")
    fetched = fetch_linkedin_content(url)
    post_text = fetched["text"]
    author = fetched["author"]
    content_type = fetched["content_type"]
    raw_title = fetched["title"]

    print(f"  ✓ Type: {content_type}")
    print(f"  ✓ Author: {author}")
    print(f"  ✓ Text preview: {post_text[:120].replace(chr(10), ' ')}…")

    # ── Step 2: Classify category ───────────────────────────────
    print("\n[2/5] Classifying category...")
    auto_category = classify_category(post_text, url, topic_keywords)
    tools = infer_tools(post_text, tool_keywords)
    topics = infer_topics_from_category(auto_category)
    print(f"  ✓ Category: {auto_category}")
    print(f"  ✓ Tools detected: {tools or ['none']}")

    # ── Step 3: AI enrichment ───────────────────────────────────
    enrichment = None
    if not no_enrich and not dry_run:
        print(f"\n[3/5] Enriching with Ollama ({os.getenv('OLLAMA_MODEL', 'llama3')})...")
        from agents.enricher import enrich_content
        enrichment = enrich_content(
            post_text=post_text,
            url=url,
            auto_category=auto_category,
            use_crewai=use_crewai,
        )
        if enrichment:
            # Use agent-confirmed category if different
            if enrichment.suggested_category != auto_category:
                print(f"  🔄 Category refined: {auto_category} → {enrichment.suggested_category}")
                auto_category = enrichment.suggested_category
                topics = infer_topics_from_category(auto_category)
            print(f"  ✓ Summary: {enrichment.summary[:100]}…")
            print(f"  ✓ Takeaways: {len(enrichment.key_takeaways)} items")
        else:
            print("  ⚠️  Enrichment skipped — will save with basic classification only.")
    else:
        print("\n[3/5] Skipping enrichment (--no-enrich or --dry-run).")

    # ── Step 4: Build entry ─────────────────────────────────────
    print("\n[4/5] Building knowledge entry...")
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    date_prefix = datetime.now().strftime("%Y%m%d")
    base_slug = f"{date_prefix}-{slugify(raw_title)[:50]}"
    slug = unique_slug(base_slug)

    category_dir = CONTENT / auto_category.replace("/", os.sep)
    entry_path = category_dir / f"{slug}.md"
    raw_path = RAW / datetime.now().strftime("%Y/%m") / f"{slug}.json"

    entry = LinkedInEntry(
        title=raw_title,
        slug=slug,
        created_at=created_at,
        content_type=content_type,
        category=auto_category,
        author=author,
        url=url,
        post_text=post_text,
        topics=topics,
        tools=tools,
        summary=enrichment.summary if enrichment else "",
        why_it_matters=enrichment.why_it_matters if enrichment else "",
        key_takeaways=enrichment.key_takeaways if enrichment else [],
        reusable_ideas=enrichment.reusable_ideas if enrichment else [],
        enriched=enrichment is not None,
        entry_path=str(entry_path),
        raw_path=str(raw_path),
        source_kind="url",
        source_name=urlparse(url).netloc,
        source_path=url,
    )

    if dry_run:
        print("\n  ── DRY RUN MODE ── (nothing written)")
        print(f"  Would write to: content/{auto_category}/{slug}.md")
        print(f"  Category: {auto_category}")
        print(f"  Tools: {tools}")
        print(f"  Enriched: {enrichment is not None}")
        print(f"\n  Preview of markdown frontmatter:")
        preview = build_markdown(entry).split("---")[1]
        for line in preview.strip().splitlines()[:10]:
            print(f"    {line}")
        return

    # ── Step 5: Write files ─────────────────────────────────────
    category_dir.mkdir(parents=True, exist_ok=True)
    entry_path.write_text(build_markdown(entry), encoding="utf-8")
    print(f"  ✓ Written: content/{auto_category}/{slug}.md")

    # Save raw JSON snapshot
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps({
        "url": url,
        "fetched_at": created_at,
        **fetched,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # Update registry
    registry_rows = load_registry()
    if not any(r.get("source_path") == url for r in registry_rows):
        registry_rows.append(asdict(entry))
        save_registry(registry_rows)
    else:
        print("  ℹ️  URL already in registry — skipping duplicate registry entry.")
        registry_rows_before = len(registry_rows)

    # Rebuild indexes
    registry_rows = load_registry()
    rebuild_indexes(registry_rows)
    print(f"  ✓ Indexes rebuilt")

    # ── Step 5: Git commit (local) ──────────────────────────────
    print("\n[5/5] Creating local git commit...")
    git_commit(entry)

    print(f"\n{'='*60}")
    print(f"✅  Done! New entry added to your knowledge hub.")
    print(f"{'='*60}")


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest a LinkedIn post/article into your GitHub knowledge hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 linkedin_ingest.py "https://www.linkedin.com/posts/..."
  python3 linkedin_ingest.py --dry-run "https://www.linkedin.com/posts/..."
  python3 linkedin_ingest.py --no-enrich "https://www.linkedin.com/posts/..."
  python3 linkedin_ingest.py --use-crewai "https://www.linkedin.com/posts/..."

After ingestion:
  Review the generated file in content/<category>/
  Then push: git push
        """,
    )
    parser.add_argument("url", help="LinkedIn post or article URL")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Classify and preview without writing any files",
    )
    parser.add_argument(
        "--no-enrich",
        action="store_true",
        help="Skip AI enrichment (faster — just classify and save)",
    )
    parser.add_argument(
        "--use-crewai",
        action="store_true",
        help="Use full CrewAI agent loop instead of direct Ollama call (slower but more agentic)",
    )

    args = parser.parse_args()

    if not args.url.startswith("http"):
        parser.error("URL must start with http:// or https://")

    run_pipeline(
        url=args.url,
        dry_run=args.dry_run,
        no_enrich=args.no_enrich,
        use_crewai=args.use_crewai,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
