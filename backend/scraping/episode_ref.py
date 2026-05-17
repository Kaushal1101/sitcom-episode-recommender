from __future__ import annotations

from dataclasses import dataclass


def make_episode_id(series_slug: str, season: int, episode: int) -> str:
    return f"{series_slug}_s{season:02d}_e{episode:02d}"


@dataclass(frozen=True, slots=True)
class EpisodeRef:
    series_slug: str
    season_number: int
    episode_number: int
    wiki_origin: str
    wiki_page_title: str
    linked_episode_numbers: tuple[int, ...] = ()

    @property
    def episode_id(self) -> str:
        return make_episode_id(self.series_slug, self.season_number, self.episode_number)

    def all_episode_numbers(self) -> tuple[int, ...]:
        nums = [self.episode_number, *self.linked_episode_numbers]
        return tuple(sorted(set(nums)))

    def all_episode_ids(self) -> tuple[str, ...]:
        return tuple(
            make_episode_id(self.series_slug, self.season_number, n)
            for n in self.all_episode_numbers()
        )


THE_OFFICE_COMBINED_OVERRIDES: dict[tuple[str, int, int], EpisodeRef] = {
    ("the_office", 4, 1): EpisodeRef(
        series_slug="the_office",
        season_number=4,
        episode_number=1,
        wiki_origin="https://theoffice.fandom.com",
        wiki_page_title="Fun_Run",
        linked_episode_numbers=(2,),
    ),
    ("the_office", 4, 2): EpisodeRef(
        series_slug="the_office",
        season_number=4,
        episode_number=2,
        wiki_origin="https://theoffice.fandom.com",
        wiki_page_title="Fun_Run",
        linked_episode_numbers=(1,),
    ),
}

COMBINED_OVERRIDES: dict[tuple[str, int, int], EpisodeRef] = {
    **THE_OFFICE_COMBINED_OVERRIDES,
}
