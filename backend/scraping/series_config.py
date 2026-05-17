from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SeriesConfig:
    series_slug: str
    series_title: str
    wiki_origin: str
    season_page_template: str


THE_OFFICE = SeriesConfig(
    series_slug="the_office",
    series_title="The Office",
    wiki_origin="https://theoffice.fandom.com",
    season_page_template="Season_{n}",
)
