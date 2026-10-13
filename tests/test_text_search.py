import sqlite3

import pytest

from geo_search_env.experiment import text_search as ts


def test_domains_and_phones_are_recognised_and_normalised():
    tlds = ts.GENERIC_TLDS | {"lu", "uk", "sk"}
    assert ts.domain_of("https://www.porsche.lu/fr/", tlds) == "porsche.lu"
    assert ts.domain_of("shop.example.co.uk", tlds) == "example.co.uk"
    assert ts.domain_of("St.Mary", tlds) is None  # read text, not a domain
    assert ts.phone_digits("+352 45 11 451") == "+3524511451"
    assert ts.phone_digits("0044 20 7946 0958") == "+442079460958"
    assert ts.phone_digits("2010") is None
    assert ts.norm("Galéria Nedbalka — Straße") == "galeria nedbalka strasse"


@pytest.fixture
def tool(tmp_path, monkeypatch):
    places, yfcc = tmp_path / "places.db", tmp_path / "yfcc.db"
    con = sqlite3.connect(places)
    con.executescript("""
        CREATE TABLE places (rowid INTEGER PRIMARY KEY, src INTEGER, name TEXT, lat REAL, lon REAL, cc TEXT, kind TEXT, chain INTEGER);
        CREATE TABLE domains (domain TEXT, place INTEGER);
        CREATE TABLE phones (tail TEXT, digits TEXT, place INTEGER);
        CREATE TABLE towns (lat REAL, lon REAL, name TEXT, cc TEXT, pop INTEGER);
        INSERT INTO places VALUES (1, 1, 'Galéria Nedbalka', 48.144, 17.112, 'SK', 'art_gallery', 0);
        INSERT INTO places VALUES (2, 1, 'Nedbalka Café', 40.0, -3.7, 'ES', 'cafe', 0);
        INSERT INTO domains VALUES ('galerianedbalka.sk', 1);
        INSERT INTO phones VALUES ('4010101', '421254010101', 1);
        INSERT INTO towns VALUES (48.148, 17.107, 'Bratislava', 'SK', 400000);
        INSERT INTO towns VALUES (40.417, -3.704, 'Madrid', 'ES', 3000000);
        CREATE VIRTUAL TABLE names USING fts5 (name, content = 'places', content_rowid = 'rowid', tokenize = 'unicode61 remove_diacritics 2');
        INSERT INTO names (names) VALUES ('rebuild');""")
    con.commit()
    con = sqlite3.connect(yfcc)
    con.executescript("""
        CREATE TABLE meta (rowid INTEGER PRIMARY KEY, photo_id INTEGER, uid TEXT, lat REAL, lon REAL, accuracy INTEGER);
        CREATE VIRTUAL TABLE titles USING fts5 (text, content = '', tokenize = 'unicode61 remove_diacritics 2');""")
    rows = [(1, 100, "me@N00", 40.0, -3.7, 16)] + [(i, 100 + i, "me@N00", 40.0, -3.7, 16) for i in range(2, 30)]  # the query's own photographer
    rows += [(30, 200, "other@N01", 48.145, 17.11, 16)]
    con.executemany("INSERT INTO meta VALUES (?, ?, ?, ?, ?, ?)", rows)
    con.executemany("INSERT INTO titles (rowid, text) VALUES (?, ?)", [(r[0], "galeria nedbalka") for r in rows])
    con.commit()
    monkeypatch.setattr(ts, "PLACES_DB", places)
    monkeypatch.setattr(ts, "YFCC_DB", yfcc)
    return ts.TextSearch()


def test_name_search_excludes_the_photo_and_its_photographer(tool):
    r = tool.search("Galeria Nedbalka", exclude=(100, "me@N00"))
    assert r.matches["Flickr photos"] == 1  # 29 photos by the query's photographer never count
    assert r.clusters[0].label == "Bratislava, SK"
    assert r.clusters[0].places == 1 and r.clusters[0].users == 1


def test_one_photographer_counts_once(tool):
    r = tool.search("Galeria Nedbalka")
    madrid = next(c for c in r.clusters if c.label == "Madrid, ES")
    assert madrid.photos == 29 and madrid.users == 1
    assert r.clusters[0].label == "Bratislava, SK"  # 1 place + 1 user outweighs 29 photos of one user (whose cluster weighs 1)


def test_domain_phone_and_near(tool):
    assert tool.search("www.galerianedbalka.sk").clusters[0].label == "Bratislava, SK"
    r = tool.search("+421 2 5401 0101")
    assert r.clusters[0].label == "Bratislava, SK" and "Slovakia" in r.hint
    r = tool.search("Nedbalka", near=(40.4, -3.7, 50))
    assert [c.label for c in r.clusters] == ["Madrid, ES"]
