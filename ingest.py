#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
INBOX = ROOT / "inbox"
RAW = ROOT / "raw"
CONTENT = ROOT / "content"
INDEXES = ROOT / "indexes"
REGISTRY = ROOT / "registry.json"

TEXT_EXTENSIONS = {
    ".md",
    ".txt",
    ".html",
    ".htm",
    ".rst",
    ".csv",
    ".json",
    ".yaml",
    ".yml",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".go",
    ".java",
    ".sh",
}

CONTENT_TYPE_MAP = {
    ".md": "note",
    ".txt": "note",
    ".rst": "note",
    ".html": "article",
    ".htm": "article",
    ".pdf": "document",
    ".doc": "document",
    ".docx": "document",
    ".ppt": "slides",
    ".pptx": "slides",
    ".key": "slides",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".gif": "image",
    ".webp": "image",
    ".svg": "image",
    ".mp4": "video",
    ".mov": "video",
    ".mkv": "video",
    ".avi": "video",
    ".mp3": "audio",
    ".wav": "audio",
    ".m4a": "audio",
    ".csv": "dataset",
    ".json": "dataset",
    ".yaml": "config",
    ".yml": "config",
    ".py": "code",
    ".js": "code",
    ".ts": "code",
    ".tsx": "code",
    ".jsx": "code",
    ".go": "code",
    ".java": "code",
    ".sh": "code",
}

DEFAULT_TOOL_KEYWORDS = {
    "github": ["github", "actions", "pull request", "pr", "repo"],
    "gitlab": ["gitlab", "gitlab ci", "gitlab-ci"],
    "docker": ["docker", "container", "dockerfile"],
    "kubernetes": ["k8s", "kubernetes", "helm", "kubectl"],
    "terraform": ["terraform", "iac", "hcl"],
    "aws": ["aws", "ec2", "s3", "lambda", "cloudformation"],
    "azure": ["azure", "aks", "arm template"],
    "gcp": ["gcp", "gke", "bigquery"],
    "python": ["python", ".py", "pandas", "fastapi", "django"],
    "javascript": ["javascript", "node", "react", "next.js", "vue"],
    "ai": ["openai", "llm", "prompt", "rag", "agent", "gpt"],
    "devops": ["cicd", "ci/cd", "pipeline", "deployment", "sre"],
}

DEFAULT_TOPIC_KEYWORDS = {
    "ai": ["ai", "llm", "rag", "prompt", "agent", "gpt"],
    "devops": ["devops", "cicd", "deployment", "infra", "sre"],
    "cloud": ["cloud", "aws", "azure", "gcp", "saas"],
    "kubernetes": ["kubernetes", "k8s", "helm", "cluster"],
    "career": ["career", "job", "interview", "resume", "leadership"],
    "productivity": ["productivity", "workflow", "focus", "time management"],
    "engineering": ["engineering", "architecture", "backend", "frontend", "api"],
    "security": ["security", "vulnerability", "iam", "secrets", "compliance"],
    "data": ["data", "analytics", "sql", "dashboard", "etl"],
    "learning": ["guide", "tutorial", "explainer", "roadmap", "tips"],
}


@dataclass
class Entry:
    title: str
    slug: str
    created_at: str
    content_type: str
    source_kind: str
    source_name: str
    source_path: str
    raw_path: str
    topics: list[str]
    tools: list[str]
    summary: str
    entry_path: str


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-{2,}", "-", value)
    return value.strip("-") or "item"


def titleize(path: Path) -> str:
    stem = path.stem.replace("_", " ").replace("-", " ").strip()
    acronyms = {
        "ai": "AI",
        "api": "API",
        "aws": "AWS",
        "ci": "CI",
        "cd": "CD",
        "cicd": "CI/CD",
        "devops": "DevOps",
        "gcp": "GCP",
        "github": "GitHub",
        "gitlab": "GitLab",
        "k8s": "K8s",
        "llm": "LLM",
        "pr": "PR",
    }
    words = []
    for word in stem.split():
        lowered = word.lower()
        words.append(acronyms.get(lowered, lowered.capitalize()))
    return " ".join(words) or path.name


def read_snippet(path: Path, limit: int = 1200) -> str:
    if path.suffix.lower() not in TEXT_EXTENSIONS:
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def infer_labels(text: str, mapping: dict[str, list[str]]) -> list[str]:
    haystack = text.lower()
    matches = [label for label, keywords in mapping.items() if any(keyword in haystack for keyword in keywords)]
    return sorted(matches)


def classify_content_type(path: Path) -> str:
    return CONTENT_TYPE_MAP.get(path.suffix.lower(), "asset")


def classify_url_content_type(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()

    if "youtube.com" in host or "youtu.be" in host or "vimeo.com" in host:
        return "video"
    if "linkedin.com" in host:
        return "post"
    if "github.com" in host:
        if re.fullmatch(r"/[^/]+/[^/]+/?", path):
            return "repo"
        return "document"
    if "/blog/" in path or "/article/" in path:
        return "article"
    if "/docs/" in path or host.startswith("docs."):
        return "document"
    return "link"


def make_summary(title: str, content_type: str, topics: list[str], tools: list[str], snippet: str) -> str:
    topic_text = ", ".join(topics[:3]) if topics else "general knowledge"
    tool_text = ", ".join(tools[:3]) if tools else "mixed tools"
    if snippet:
        sentence = snippet[:180].rstrip()
        return f"{content_type.capitalize()} entry about {topic_text}, likely involving {tool_text}. Seed text: {sentence}"
    return f"{content_type.capitalize()} entry about {topic_text}, likely involving {tool_text}."


def ensure_dirs() -> None:
    for directory in (INBOX, RAW, CONTENT, INDEXES):
        directory.mkdir(parents=True, exist_ok=True)


def load_taxonomy() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    path = ROOT / "taxonomy.json"
    if not path.exists():
        return DEFAULT_TOOL_KEYWORDS, DEFAULT_TOPIC_KEYWORDS
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return DEFAULT_TOOL_KEYWORDS, DEFAULT_TOPIC_KEYWORDS
    tools = payload.get("tools", DEFAULT_TOOL_KEYWORDS)
    topics = payload.get("topics", DEFAULT_TOPIC_KEYWORDS)
    return tools, topics


def load_registry() -> list[dict]:
    if not REGISTRY.exists():
        return []
    try:
        return json.loads(REGISTRY.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def save_registry(entries: list[dict]) -> None:
    REGISTRY.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def copy_to_raw(path: Path, slug: str) -> Path:
    stamp = datetime.now().strftime("%Y/%m")
    destination_dir = RAW / stamp
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{slug}{path.suffix.lower()}"
    shutil.copy2(path, destination)
    return destination


def write_json_to_raw(payload: dict, slug: str) -> Path:
    stamp = datetime.now().strftime("%Y/%m")
    destination_dir = RAW / stamp
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{slug}.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def clean_text(value: str) -> str:
    value = html.unescape(value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def extract_title_from_html(payload: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", payload, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return clean_text(match.group(1))


def extract_meta_description(payload: str) -> str:
    patterns = [
        r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']',
        r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\'](.*?)["\']',
    ]
    for pattern in patterns:
        match = re.search(pattern, payload, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return clean_text(match.group(1))
    return ""


def title_from_url(url: str) -> str:
    parsed = urlparse(url)
    leaf = Path(parsed.path).name or parsed.netloc
    return titleize(Path(leaf))


def fetch_url_metadata(url: str) -> dict:
    parsed = urlparse(url)
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 Codex Knowledge Ingestor"})
    metadata = {
        "url": url,
        "final_url": url,
        "domain": parsed.netloc,
        "title": title_from_url(url),
        "description": "",
        "snippet": "",
        "fetch_status": "unfetched",
    }
    try:
        with urlopen(request, timeout=15) as response:
            body = response.read(200000).decode("utf-8", errors="ignore")
            final_url = response.geturl()
    except (HTTPError, URLError, TimeoutError) as exc:
        metadata["fetch_status"] = f"error: {exc}"
        return metadata

    title = extract_title_from_html(body)
    description = extract_meta_description(body)
    visible_text = clean_text(body)[:1200]

    metadata.update(
        {
            "final_url": final_url,
            "domain": urlparse(final_url).netloc,
            "title": title or metadata["title"],
            "description": description,
            "snippet": description or visible_text,
            "fetch_status": "ok",
        }
    )
    return metadata


def build_entry_markdown(entry: Entry) -> str:
    topics = ", ".join(entry.topics) if entry.topics else "general"
    tools = ", ".join(entry.tools) if entry.tools else "unclassified"
    return f"""---
title: "{entry.title}"
slug: "{entry.slug}"
created_at: "{entry.created_at}"
content_type: "{entry.content_type}"
source_kind: "{entry.source_kind}"
topics: [{", ".join(f'"{item}"' for item in entry.topics)}]
tools: [{", ".join(f'"{item}"' for item in entry.tools)}]
source_name: "{entry.source_name}"
source_path: "{entry.source_path}"
raw_path: "{entry.raw_path}"
---

# {entry.title}

## Snapshot

{entry.summary}

## Classification

- Content type: `{entry.content_type}`
- Source kind: `{entry.source_kind}`
- Topics: `{topics}`
- Tools: `{tools}`

## Why it matters

Add the practical value here after reviewing the source.

## Key takeaways

- Add takeaway 1
- Add takeaway 2
- Add takeaway 3

## Reusable ideas

- What can someone apply immediately?
- Which workflow, tool, or pattern is worth reusing?

## Source

- Original upload: `{entry.source_path}`
- Archived asset: `{entry.raw_path}`
"""


def write_entry(entry: Entry) -> None:
    primary_topic = entry.topics[0] if entry.topics else "general"
    destination_dir = CONTENT / primary_topic
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{entry.slug}.md"
    destination.write_text(build_entry_markdown(entry), encoding="utf-8")


def rebuild_indexes(registry_rows: list[dict]) -> None:
    INDEXES.mkdir(parents=True, exist_ok=True)
    entries = [Entry(**row) for row in registry_rows]
    entries.sort(key=lambda item: item.created_at, reverse=True)

    write_index(
        INDEXES / "all.md",
        "# All Entries\n\n" + "\n".join(render_entry_line(entry) for entry in entries),
    )

    by_type: dict[str, list[Entry]] = defaultdict(list)
    by_topic: dict[str, list[Entry]] = defaultdict(list)
    by_tool: dict[str, list[Entry]] = defaultdict(list)

    for entry in entries:
        by_type[entry.content_type].append(entry)
        if entry.topics:
            for topic in entry.topics:
                by_topic[topic].append(entry)
        else:
            by_topic["general"].append(entry)
        if entry.tools:
            for tool in entry.tools:
                by_tool[tool].append(entry)
        else:
            by_tool["unclassified"].append(entry)

    write_group_index(INDEXES / "by-content-type.md", "# Browse By Content Type", by_type)
    write_group_index(INDEXES / "by-topic.md", "# Browse By Topic", by_topic)
    write_group_index(INDEXES / "by-tool.md", "# Browse By Tool", by_tool)


def render_entry_line(entry: Entry) -> str:
    rel = Path(entry.entry_path).relative_to(ROOT)
    return f"- [{entry.title}](../{rel.as_posix()}) | `{entry.content_type}` | {entry.created_at[:10]}"


def write_index(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n", encoding="utf-8")


def write_group_index(path: Path, heading: str, groups: dict[str, list[Entry]]) -> None:
    lines = [heading, ""]
    for group in sorted(groups):
        lines.append(f"## {group}")
        lines.append("")
        lines.extend(render_entry_line(entry) for entry in groups[group])
        lines.append("")
    write_index(path, "\n".join(lines))


def candidate_file_paths(explicit_paths: Iterable[str]) -> list[Path]:
    if explicit_paths:
        paths = []
        for item in explicit_paths:
            if is_url(item):
                continue
            paths.append(Path(item).expanduser().resolve())
        return paths
    return sorted(path.resolve() for path in INBOX.iterdir() if path.is_file() and path.name != ".gitkeep")


def candidate_urls(explicit_inputs: Iterable[str], url_args: Iterable[str]) -> list[str]:
    urls = [value for value in explicit_inputs if is_url(value)]
    urls.extend(url_args)
    deduped = []
    seen = set()
    for url in urls:
        if url not in seen:
            seen.add(url)
            deduped.append(url)
    return deduped


def unique_slug(base_slug: str, registry_rows: list[dict]) -> str:
    known = {row["slug"] for row in registry_rows}
    slug = base_slug
    counter = 2
    while slug in known or list(CONTENT.glob(f"**/{slug}.md")):
        slug = f"{base_slug}-{counter}"
        counter += 1
    return slug


def ingest_file(
    path: Path,
    dry_run: bool,
    registry_rows: list[dict],
    tool_keywords: dict[str, list[str]],
    topic_keywords: dict[str, list[str]],
) -> Entry:
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    title = titleize(path)
    slug = unique_slug(f"{datetime.now().strftime('%Y%m%d')}-{slugify(title)}", registry_rows)
    snippet = read_snippet(path)
    inference_base = f"{path.name} {title} {snippet}"
    content_type = classify_content_type(path)
    topics = infer_labels(inference_base, topic_keywords)
    tools = infer_labels(inference_base, tool_keywords)
    raw_path = RAW / datetime.now().strftime("%Y/%m") / f"{slug}{path.suffix.lower()}"
    primary_topic = topics[0] if topics else "general"
    entry_path = CONTENT / primary_topic / f"{slug}.md"

    entry = Entry(
        title=title,
        slug=slug,
        created_at=created_at,
        content_type=content_type,
        source_kind="file",
        source_name=path.name,
        source_path=str(path),
        raw_path=str(raw_path),
        topics=topics,
        tools=tools,
        summary=make_summary(title, content_type, topics, tools, snippet),
        entry_path=str(entry_path),
    )

    if not dry_run:
        copied_path = copy_to_raw(path, slug)
        entry.raw_path = str(copied_path)
        entry.entry_path = str(entry_path)
        write_entry(entry)
    return entry


def ingest_url(
    url: str,
    dry_run: bool,
    registry_rows: list[dict],
    tool_keywords: dict[str, list[str]],
    topic_keywords: dict[str, list[str]],
) -> Entry:
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    metadata = fetch_url_metadata(url)
    title = metadata["title"] or title_from_url(url)
    slug = unique_slug(f"{datetime.now().strftime('%Y%m%d')}-{slugify(title)}", registry_rows)
    snippet = metadata["snippet"][:1200]
    inference_base = f"{url} {title} {metadata['description']} {snippet}"
    content_type = classify_url_content_type(metadata["final_url"])
    topics = infer_labels(inference_base, topic_keywords)
    tools = infer_labels(inference_base, tool_keywords)
    raw_path = RAW / datetime.now().strftime("%Y/%m") / f"{slug}.json"
    primary_topic = topics[0] if topics else "general"
    entry_path = CONTENT / primary_topic / f"{slug}.md"

    entry = Entry(
        title=title,
        slug=slug,
        created_at=created_at,
        content_type=content_type,
        source_kind="url",
        source_name=metadata["domain"],
        source_path=url,
        raw_path=str(raw_path),
        topics=topics,
        tools=tools,
        summary=make_summary(title, content_type, topics, tools, snippet),
        entry_path=str(entry_path),
    )

    if not dry_run:
        payload = {
            "url": url,
            "fetched_at": created_at,
            **metadata,
        }
        saved = write_json_to_raw(payload, slug)
        entry.raw_path = str(saved)
        entry.entry_path = str(entry_path)
        write_entry(entry)
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize loose uploads into repo-ready knowledge entries.")
    parser.add_argument("inputs", nargs="*", help="Files or URLs to ingest. Defaults to everything in inbox/.")
    parser.add_argument("--url", action="append", default=[], help="URL to ingest. Repeat for multiple links.")
    parser.add_argument("--dry-run", action="store_true", help="Classify files without writing output.")
    parser.add_argument("--keep-inbox", action="store_true", help="Keep source files in inbox after processing.")
    args = parser.parse_args()

    ensure_dirs()
    tool_keywords, topic_keywords = load_taxonomy()
    paths = candidate_file_paths(args.inputs)
    urls = candidate_urls(args.inputs, args.url)
    if not paths and not urls:
        print("No files or URLs found to ingest.")
        return 0

    registry_rows = load_registry()
    ingested: list[Entry] = []

    for path in paths:
        if not path.exists() or not path.is_file():
            print(f"Skipping missing file: {path}")
            continue
        entry = ingest_file(
            path,
            dry_run=args.dry_run,
            registry_rows=registry_rows,
            tool_keywords=tool_keywords,
            topic_keywords=topic_keywords,
        )
        ingested.append(entry)
        print(f"{entry.source_name} -> {entry.content_type} | topics={entry.topics or ['general']} | tools={entry.tools or ['unclassified']}")

        if not args.dry_run and path.parent == INBOX and not args.keep_inbox:
            path.unlink()

    for url in urls:
        entry = ingest_url(
            url,
            dry_run=args.dry_run,
            registry_rows=registry_rows,
            tool_keywords=tool_keywords,
            topic_keywords=topic_keywords,
        )
        ingested.append(entry)
        print(f"{entry.source_path} -> {entry.content_type} | topics={entry.topics or ['general']} | tools={entry.tools or ['unclassified']}")

    if args.dry_run:
        return 0

    known_slugs = {row["slug"] for row in registry_rows}
    for entry in ingested:
        if entry.slug not in known_slugs:
            registry_rows.append(asdict(entry))
    save_registry(registry_rows)
    rebuild_indexes(registry_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
