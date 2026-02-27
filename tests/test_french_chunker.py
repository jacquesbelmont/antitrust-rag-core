from app.application.chunking.legal_chunker import split_legal_text_hierarchical


def test_french_arret_becomes_hierarchy_root():
    text = """ARRÊT du 15 janvier 2024

TITRE I Dispositions générales

SECTION 1 Champ d'application

Attendu que la société X a conclu des accords. """

    chunks = split_legal_text_hierarchical(text=text)

    paths = [c.hierarchy_path for c in chunks]
    assert any("ARRÊT du 15 janvier 2024" in p for path in paths for p in path)
    assert any("TITRE I Dispositions générales" in p for path in paths for p in path)


def test_french_decision_number_detected():
    text = """DÉCISION N° 24-D-01 du 12 mars 2024

SECTION 1 Faits

Considérant que l'entreprise Y a abusé de sa position dominante."""

    chunks = split_legal_text_hierarchical(text=text)

    assert len(chunks) >= 1
    paths_flat = [p for c in chunks for p in c.hierarchy_path]
    assert any("DÉCISION" in p for p in paths_flat)


def test_attendu_que_triggers_paragraph_split():
    text = """SECTION 1 Analyse

Attendu que le marché pertinent est celui de la distribution.
Attendu que les parts de marché dépassent 40 pour cent."""

    chunks = split_legal_text_hierarchical(text=text, max_chunk_chars=500, overlap_chars=0)

    # Both "attendu que" paragraphs should produce separate blocks
    assert len(chunks) >= 1
    combined = " ".join(c.text for c in chunks)
    assert "marché pertinent" in combined
    assert "40 pour cent" in combined


def test_section_paragraph_symbol():
    text = """CHAPITRE I Marché concerné

§ 1 Définition du marché de produits

Le marché comprend la fourniture de services juridiques spécialisés.

§ 2 Marché géographique

Le marché géographique est de dimension nationale."""

    chunks = split_legal_text_hierarchical(text=text)

    assert any("CHAPITRE I" in p for c in chunks for p in c.hierarchy_path)
