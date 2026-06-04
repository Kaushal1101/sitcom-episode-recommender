from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

from backend.recommender.episode_feature_builder import TONE_DIMENSIONS


@dataclass
class ConversationMessage:
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MoodPreferences:
    energy: float = 0.0
    humor: float = 0.0
    sadness: float = 0.0
    comfort: float = 0.0


@dataclass
class EpisodeStylePreferences:
    specialness: float = 0.0
    novelty: float = 0.0
    setting_variation: float = 0.0
    continuity_heaviness: float = 0.0


def _default_tone_preferences() -> dict[str, float]:
    return {tone: 0.0 for tone in TONE_DIMENSIONS}


@dataclass
class Preferences:
    mood: MoodPreferences = field(default_factory=MoodPreferences)
    episode_style: EpisodeStylePreferences = field(
        default_factory=EpisodeStylePreferences
    )
    tone_preferences: dict[str, float] = field(
        default_factory=_default_tone_preferences
    )


@dataclass
class HardConstraints:
    excluded_shows: list[str] = field(default_factory=list)
    excluded_episodes: list[str] = field(default_factory=list)
    excluded_tags: list[str] = field(default_factory=list)
    max_runtime_minutes: int | None = None
    min_runtime_minutes: int | None = None


@dataclass
class RecommendationContext:
    previously_recommended: list[str] = field(default_factory=list)
    accepted_recommendations: list[str] = field(default_factory=list)
    rejected_recommendations: list[str] = field(default_factory=list)


@dataclass
class ClarificationState:
    awaiting_clarification: bool = False
    pending_question: str | None = None


@dataclass
class Timestamps:
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SessionState:
    session_id: str = field(default_factory=lambda: str(uuid4()))
    conversation_history: list[ConversationMessage] = field(default_factory=list)
    current_preferences: Preferences = field(default_factory=Preferences)
    hard_constraints: HardConstraints = field(default_factory=HardConstraints)
    recommendation_context: RecommendationContext = field(
        default_factory=RecommendationContext
    )
    clarification_state: ClarificationState = field(default_factory=ClarificationState)
    timestamps: Timestamps = field(default_factory=Timestamps)
