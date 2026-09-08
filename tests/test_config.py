from src import config


def test_generation_size_edges_are_multiples_of_16():
    """gpt-image-2 rejects any edge that is not a multiple of 16."""
    w, h = (int(v) for v in config.GEN_SIZE.split("x"))
    assert w % 16 == 0
    assert h % 16 == 0


def test_generation_size_is_16_by_9():
    w, h = (int(v) for v in config.GEN_SIZE.split("x"))
    assert round(w / h, 4) == round(16 / 9, 4)


def test_final_dimensions_are_1920_by_1080():
    assert (config.FINAL_W, config.FINAL_H) == (1920, 1080)


def test_generation_size_is_larger_than_final_so_we_downscale():
    w, h = (int(v) for v in config.GEN_SIZE.split("x"))
    assert w > config.FINAL_W and h > config.FINAL_H


def test_max_bytes_is_youtube_thumbnail_cap():
    assert config.MAX_BYTES == 2 * 1024 * 1024


def test_drive_scope_allows_creating_folders_beside_the_source_ad():
    """drive.file cannot create a subfolder in a folder the app did not create."""
    assert "https://www.googleapis.com/auth/drive" in config.GOOGLE_SCOPES


def test_the_headline_font_ships_in_the_repo_and_opens():
    from PIL import ImageFont
    from src import config
    assert config.HEADLINE_FONT.exists(), config.HEADLINE_FONT
    assert config.HEADLINE_FALLBACK_FONT.exists()
    assert (config.FONTS_DIR / "OFL.txt").exists(), "the licence travels with the font"
    font = ImageFont.truetype(str(config.HEADLINE_FONT), 100)
    assert font.getlength("SAME AI") > 0


def test_typesetting_constants_match_the_spec():
    from src import config
    assert config.TEXT_FLOOR_FRACTION == 0.07
    assert config.TEXT_BOTTOM_RESERVE == 0.14
    assert config.TEXT_SUPERSAMPLE == 2
    assert config.NAVY == (12, 20, 43)
    assert config.CREAM == (255, 248, 238)
