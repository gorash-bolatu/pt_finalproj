import pytest

# Skip tests gracefully if spacy not installed
spacy = pytest.importorskip("spacy")

from src.aspect_extractor import extract_aspects  # expects src/aspect_extractor.py

@pytest.mark.skipif(not spacy.util.is_package("en_core_web_sm"),
                    reason="spaCy 'en_core_web_sm' model not installed")
def test_extract_aspects_simple():
    text = "The battery life is excellent but the shipping time was terrible. Screen quality is fantastic."
    aspects = extract_aspects(text, topn=5)
    # aspects should include 'battery life' or 'battery' and 'shipping' and 'screen quality'
    joined = " ".join(aspects).lower()
    assert "battery" in joined or "battery life" in joined
    assert "shipping" in joined
    assert "screen" in joined or "screen quality" in joined
