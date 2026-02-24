from utils.wikipedia import build_species_wikipedia_url


def test_build_species_wikipedia_url_prefers_scientific_name():
    url = build_species_wikipedia_url(
        common_name="Kohlmeise", scientific_name="Parus_major"
    )
    assert url is not None
    assert "de.wikipedia.org" in url
    assert "Parus+major" in url


def test_build_species_wikipedia_url_falls_back_to_common_name():
    url = build_species_wikipedia_url(common_name="Blaumeise", scientific_name=None)
    assert url is not None
    assert "Blaumeise" in url


def test_build_species_wikipedia_url_returns_none_for_empty_values():
    assert build_species_wikipedia_url(common_name="", scientific_name=None) is None
    assert build_species_wikipedia_url(common_name=None, scientific_name=None) is None


def test_build_species_wikipedia_url_honors_locale():
    url = build_species_wikipedia_url(
        common_name="Great tit", scientific_name="Parus major", locale="en"
    )
    assert url is not None
    assert url.startswith("https://en.wikipedia.org/")
