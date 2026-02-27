from app.application.chunking.legal_chunker import split_legal_text_hierarchical


def test_chunker_preserves_hierarchy_and_splits():
    text = """TITLE I GENERAL

SECTION 1 Scope

1 This is paragraph one. It is short.
2 This is paragraph two. """ + ("A" * 2000)

    chunks = split_legal_text_hierarchical(text=text, max_chunk_chars=500, overlap_chars=50)

    assert len(chunks) > 1
    assert any(c.hierarchy_path for c in chunks)
    assert all(len(c.text) <= 500 for c in chunks)
