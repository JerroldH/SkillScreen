"""Package input recipe copied from the released SkillScreen Package model.

Only input formatting and token coverage are retained; see the HF v1.1 model.
"""


from bisect import bisect_left, bisect_right
from pathlib import PurePosixPath
import re


CODE_SUFFIXES = frozenset(".py .sh .bash .zsh .fish .js .mjs .cjs .jsx .ts .tsx .rb .pl .php .ps1 .psm1 .bat .cmd .go .rs .java .c .h .cc .cpp .hpp .cs .swift .lua .r .jl .sql".split())


CONFIG_SUFFIXES = frozenset(".json .jsonc .yaml .yml .toml .ini .cfg .conf .env .xml .properties".split())


BULK_SUFFIXES = frozenset(".csv .tsv .jsonl .ndjson .geojson .log .svg .map".split())


BULK_DIRECTORIES = frozenset(("data", "assets", "output", "outputs", "reports"))


def safe_relative_path(value):
    """Validate a serialized path. Never resolve, open or execute that path."""
    if (not isinstance(value, str) or not value or "\\" in value or
            value.startswith("/") or re.match(r"^[A-Za-z]:", value) or
            any(ord(c) < 32 or ord(c) == 127 for c in value) or
            any(part in ("", ".", "..") for part in value.split("/"))):
        raise ValueError("Expected an inert, unambiguous relative package path")
    value.encode("utf-8")
    return value


def file_priority(item):
    path = PurePosixPath(safe_relative_path(item["path"]))
    suffix = path.suffix.lower()
    parts = {p.lower() for p in path.parts[:-1]}
    if parts & BULK_DIRECTORIES or suffix in BULK_SUFFIXES:
        priority = 3
    elif suffix in CODE_SUFFIXES or path.name.lower() in ("makefile", "dockerfile"):
        priority = 0
    elif suffix in CONFIG_SUFFIXES or path.name.startswith(".env"):
        priority = 1
    else:
        priority = 2
    return priority, item["path"]


def ordered_files(row):
    """Order only explicit companion text files using label-independent rules."""
    if not isinstance(row.get("skill_text"), str) or not isinstance(row.get("files"), list):
        raise ValueError("Package records require original Markdown and companion files")
    files = row["files"]
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "text"} or not isinstance(item["text"], str):
            raise ValueError("Companions must contain only path and inert text")
        safe_relative_path(item["path"])
        if item["path"].lower() == "skill.md":
            raise ValueError("SKILL.md must come from the original skill_text field")
    if len({item["path"] for item in files}) != len(files):
        raise ValueError("Duplicate companion path")
    return sorted(files, key=file_priority)


def _render(row):
    chunks = ["[SKILL.md]\n", row["skill_text"]]
    cursor = len(chunks[0])
    spans = [{"path": "SKILL.md", "start": cursor, "end": cursor + len(row["skill_text"]), "header_start": 0}]
    cursor += len(row["skill_text"])
    for item in ordered_files(row):
        header = "\n\n[FILE " + item["path"] + "]\n"
        header_start = cursor
        cursor += len(header)
        spans.append({"path": item["path"], "start": cursor, "end": cursor + len(item["text"]),
                      "header_start": header_start})
        chunks.extend((header, item["text"]))
        cursor += len(item["text"])
    return "".join(chunks), spans


def encode_package(tokenizer, row, hypothesis, max_length):
    """Return IDs, original token count and exact retained file-token coverage.

    A fast Hugging Face tokenizer is required for Unicode character offsets. The
    premise and fixed hypothesis are encoded at an explicit segment boundary.
    Truncation removes only a suffix of the premise. File counts concern content
    tokens, excluding headers. Boundary-overlapping tokens count in both spans.
    """
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Package coverage requires a fast tokenizer with character offsets")
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        raise ValueError("A fixed hypothesis is required")
    ordered_files(row)
    premise, spans = _render(row)
    suffix_ids = list(tokenizer.encode("\nHypothesis: " + hypothesis, add_special_tokens=False))
    if type(max_length) is not int or max_length <= len(suffix_ids) + 8:
        raise ValueError("Input budget is too small for the premise and fixed hypothesis")
    prefix = "Premise: "
    encoding = tokenizer(prefix + premise, add_special_tokens=False, truncation=False,
                         return_offsets_mapping=True, return_attention_mask=False,
                         return_token_type_ids=False)
    premise_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]
    if len(premise_ids) != len(offsets) or any(start > end for start, end in offsets):
        raise ValueError("Invalid tokenizer offsets")
    used = min(len(premise_ids), max_length - len(suffix_ids))
    ids = list(premise_ids[:used]) + suffix_ids
    starts = [start for start, _ in offsets]
    ends = [end for _, end in offsets]
    if any(a > b for a, b in zip(starts, starts[1:])) or any(a > b for a, b in zip(ends, ends[1:])):
        raise ValueError("Tokenizer offsets must be monotone for retained file coverage")
    files = []
    for span in spans:
        start, end = len(prefix) + span["start"], len(prefix) + span["end"]
        first = bisect_right(ends, start)
        last = bisect_left(starts, end)
        original_tokens = max(0, last - first) if start != end else 0
        used_tokens = max(0, min(last, used) - first) if original_tokens else 0
        if original_tokens:
            status = "full" if used_tokens == original_tokens else ("partial" if used_tokens else "omitted")
        else:
            header_end = bisect_left(starts, start)
            header_first = bisect_right(ends, len(prefix) + span["header_start"])
            status = "full" if used >= header_end else ("partial" if used > header_first else "omitted")
        files.append({"path": span["path"], "original_tokens": original_tokens,
                      "used_tokens": used_tokens, "status": status})
    original = len(premise_ids) + len(suffix_ids)
    return ids, original, {"original_tokens": original, "used_tokens": len(ids),
        "premise_original_tokens": len(premise_ids), "premise_used_tokens": used,
        "hypothesis_tokens": len(suffix_ids), "truncated": used < len(premise_ids),
        "files_full": sum(f["status"] == "full" for f in files),
        "files_partial": sum(f["status"] == "partial" for f in files),
        "files_omitted": sum(f["status"] == "omitted" for f in files), "files": files}
