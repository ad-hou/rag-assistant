from src.ingest import chunk_document, clean_markdown, parse_sections

DOC = """---
hide: [toc]
---

# Query Parameters { #query-parameters }

{* ../../docs_src/query_params/tutorial001.py *}

/// note | Technical Details
Ignored marker lines only.
///

![img](a.png)

Intro paragraph about query parameters in FastAPI applications and how they work.

## Required { #required }

```python
@app.get("/items/")
async def read(skip: int = 0):
    # ## not a heading
    return {"skip": skip}
```

Another paragraph that follows the code block and explains the default value.
"""


def test_clean_removes_noise_but_keeps_text():
    c = clean_markdown(DOC)
    assert "docs_src" not in c and "![img]" not in c and "///" not in c
    assert "hide: [toc]" not in c
    assert "Ignored marker lines only." in c


def test_heading_path_and_anchor_removed():
    sections = parse_sections(clean_markdown(DOC))
    paths = [p for p, _ in sections]
    assert paths == ["Query Parameters", "Query Parameters > Required"]


def test_code_block_never_split_and_comment_not_heading():
    title, items = chunk_document(DOC, max_chars=1000, overlap=100)
    assert title == "Query Parameters"
    code = [t for _, t in items if "# ## not a heading" in t]
    assert code and "```python" in code[0] and code[0].count("```") >= 2


def test_max_size_and_overlap():
    paras = "\n\n".join(f"Paragraph number {i} " + "word " * 30 for i in range(12))
    _, items = chunk_document("# T\n\n" + paras, max_chars=500, overlap=200)
    texts = [t for _, t in items]
    assert len(texts) > 1
    assert all(len(t) <= 500 for t in texts)
    last_par_first = texts[0].split("\n\n")[-1]
    assert last_par_first in texts[1]


def test_oversized_block_is_cut():
    big = "```python\n" + "\n".join(f"x{i} = {i}" for i in range(400)) + "\n```"
    _, items = chunk_document("# T\n\n" + big, max_chars=500, overlap=0)
    assert len(items) > 1 and all(len(t) <= 500 for _, t in items)
