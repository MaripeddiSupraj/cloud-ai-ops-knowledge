# Cloud AI Ops Knowledge

Cloud AI Ops Knowledge is a curated GitHub knowledge base for DevOps, cloud, and AI content that should not disappear in the scroll.

The operating model is simple:

1. Pass a URL to the script or drop anything useful into `inbox/`.
2. Run `python3 ingest.py`.
3. Commit the generated knowledge entries and indexes.

## What the script does

- accepts loose uploads such as `.md`, `.txt`, `.html`, `.pdf`, images, videos, slides, and code files
- accepts direct URLs for repos, blogs, docs, posts, and videos
- classifies each upload by content type
- infers likely tools and topics from filenames and text snippets
- copies the original asset into `raw/`
- creates a normalized markdown entry in `content/`
- rebuilds repo indexes in `indexes/`

For large binary uploads like long videos, prefer storing an external link or using Git LFS. Regular notes, screenshots, short PDFs, and code snippets fit this repo model well.

## Repo Structure

```text
cloud-ai-ops-knowledge/
  inbox/      # drop uploads here
  raw/        # copied originals, managed by the script
  content/    # generated markdown knowledge entries
  indexes/    # generated browse pages
  ingest.py   # ingestion CLI
```

## Recommended workflow

- push first, polish later
- treat `inbox/` as a capture area
- let the script normalize structure
- only improve summaries and tags for the entries worth highlighting

## Commands

Process everything in `inbox/`:

```bash
python3 ingest.py
```

Process a single URL:

```bash
python3 ingest.py --url "https://github.com/openai/openai-python"
```

Process mixed local files and links:

```bash
python3 ingest.py notes.md --url "https://kubernetes.io/docs/concepts/overview/"
```

Process specific files:

```bash
python3 ingest.py /path/to/file1.pdf /path/to/file2.md
```

Keep processed files in `inbox/`:

```bash
python3 ingest.py --keep-inbox
```

Preview classification only:

```bash
python3 ingest.py --dry-run
```

## Content model

Each generated entry includes:

- title
- added date
- content type
- source kind
- inferred topics
- inferred tools
- source asset path
- short summary seed
- next-step sections you can refine later

## Best way to use this repo

Do not wait for perfect curation. Capture aggressively, then let the repo become useful through:

- `indexes/by-topic.md` for subject browsing
- `indexes/by-tool.md` for tool-centric browsing
- `indexes/by-content-type.md` for format browsing
- `indexes/all.md` for a full stream of entries

This gives you a durable system instead of a random dump folder.
