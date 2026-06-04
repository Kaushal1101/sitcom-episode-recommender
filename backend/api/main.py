from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.csm.decision import InteractionDecision
from backend.csm.interaction_types import InteractionType
from backend.csm.preference_updater import apply_answer
from backend.csm.strategy import ConversationStrategy
from backend.models.session_state import SessionState
from backend.recommender.engine import RecommendationEngine
from backend.recommender.episode_feature_builder import load_synopses


@dataclass
class Session:
    state: SessionState
    strategy: ConversationStrategy
    last_decision: InteractionDecision | None = None


_sessions: dict[str, Session] = {}
_engine: RecommendationEngine | None = None
_db_path: Path | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _db_path
    _db_path = Path(os.getenv("SITCOM_DB_PATH", "data/app.sqlite3"))
    chroma_path = Path(os.getenv("SITCOM_CHROMA_PATH", "data/chroma"))
    _engine = RecommendationEngine(db_path=_db_path, chroma_path=chroma_path)
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnswerRequest(BaseModel):
    answer: str
    is_final_selection: bool = False


class DecisionResponse(BaseModel):
    session_id: str
    question: str
    interaction_type: str
    options: list[str] | None
    candidate_synopses: list[str] | None
    confidence: float | None
    preferences: dict | None
    done: bool
    farewell: str | None


def _to_response(
    session_id: str,
    decision: InteractionDecision,
    done: bool = False,
    farewell: str | None = None,
    candidate_synopses: list[str] | None = None,
    preferences: dict | None = None,
) -> DecisionResponse:
    return DecisionResponse(
        session_id=session_id,
        question=decision.question,
        interaction_type=decision.interaction_type.name,
        options=decision.options,
        candidate_synopses=candidate_synopses,
        confidence=decision.confidence,
        preferences=preferences,
        done=done,
        farewell=farewell,
    )


def _synopses_for_decision(decision: InteractionDecision) -> list[str] | None:
    """Return an ordered list of synopses matching decision.candidates, or None."""
    if decision.interaction_type not in (
        InteractionType.ITEM_MULTI,
        InteractionType.ITEM_YN,
    ):
        return None
    if not decision.candidates:
        return None
    synopsis_map = load_synopses(
        [c.episode_id for c in decision.candidates], _db_path
    )
    return [synopsis_map.get(c.episode_id, "") for c in decision.candidates]


def _serialize_preferences(state: SessionState) -> dict:
    mood = state.current_preferences.mood
    return {
        "mood": {
            "humor": round(mood.humor, 3),
            "energy": round(mood.energy, 3),
            "comfort": round(mood.comfort, 3),
            "sadness": round(mood.sadness, 3),
        },
        "tone": {
            k: round(v, 3)
            for k, v in state.current_preferences.tone_preferences.items()
        },
    }


def _apply_state_update(session: Session, answer: str) -> None:
    state = session.state
    decision = session.last_decision
    apply_answer(state, decision, answer, _db_path)

    if decision.interaction_type is InteractionType.RECOMMEND:
        episode_id = decision.episode_id
        if episode_id is not None:
            ctx = state.recommendation_context
            ctx.previously_recommended.append(episode_id)
            if answer == "Yes":
                ctx.accepted_recommendations.append(episode_id)
            elif answer == "No":
                ctx.rejected_recommendations.append(episode_id)

    state.timestamps.last_updated = datetime.utcnow()


@app.post("/session", response_model=DecisionResponse)
def create_session() -> DecisionResponse:
    session_id = str(uuid4())
    state = SessionState()
    strategy = ConversationStrategy(engine=_engine)
    try:
        decision = strategy.next_action(state)
    except ValueError:
        raise HTTPException(status_code=503, detail="No episodes available.")
    session = Session(state=state, strategy=strategy, last_decision=decision)
    _sessions[session_id] = session
    return _to_response(
        session_id,
        decision,
        candidate_synopses=_synopses_for_decision(decision),
        preferences=_serialize_preferences(state),
    )


@app.post("/session/{session_id}/answer", response_model=DecisionResponse)
def submit_answer(session_id: str, request: AnswerRequest) -> DecisionResponse:
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    if session.last_decision is None:
        raise HTTPException(status_code=400, detail="No pending question.")

    if request.is_final_selection:
        decision = session.last_decision
        episode_title: str | None = None
        if decision.interaction_type is InteractionType.ITEM_MULTI:
            for c in decision.candidates or []:
                if request.answer == f"{c.episode_title} ({c.series_slug})":
                    episode_title = c.episode_title
                    break
            short_circuit = True
        elif decision.interaction_type is InteractionType.ITEM_YN:
            if decision.candidates:
                episode_title = decision.candidates[0].episode_title
            short_circuit = True
        else:
            short_circuit = False

        if short_circuit:
            _apply_state_update(session, request.answer)
            farewell = (
                f"Great choice! Enjoy '{episode_title}'."
                if episode_title
                else "Great choice! Enjoy it."
            )
            interaction_type_name = decision.interaction_type.name
            del _sessions[session_id]
            return DecisionResponse(
                session_id=session_id,
                question="",
                interaction_type=interaction_type_name,
                options=None,
                candidate_synopses=None,
                confidence=None,
                preferences=None,
                done=True,
                farewell=farewell,
            )

    _apply_state_update(session, request.answer)

    if (
        session.last_decision.interaction_type is InteractionType.SOFT_CRITIQUE
        and session.last_decision.episode_id is not None
        and session.last_decision.episode_id
            in session.state.recommendation_context.previously_recommended
    ):
        session.state.recommendation_context.previously_recommended.remove(
            session.last_decision.episode_id
        )

    if session.last_decision.interaction_type is InteractionType.RATING:
        prefs = _serialize_preferences(session.state)
        del _sessions[session_id]
        return DecisionResponse(
            session_id=session_id,
            question="",
            interaction_type="RATING",
            options=None,
            candidate_synopses=None,
            confidence=None,
            preferences=prefs,
            done=True,
            farewell="Thanks for the rating! Hope you enjoy it.",
        )

    try:
        decision = session.strategy.next_action(session.state)
    except ValueError:
        prefs = _serialize_preferences(session.state)
        del _sessions[session_id]
        return DecisionResponse(
            session_id=session_id,
            question="",
            interaction_type="",
            options=None,
            candidate_synopses=None,
            confidence=None,
            preferences=prefs,
            done=True,
            farewell="No more episodes to suggest. See you next time!",
        )

    session.last_decision = decision
    return _to_response(
        session_id,
        decision,
        candidate_synopses=_synopses_for_decision(decision),
        preferences=_serialize_preferences(session.state),
    )


@app.delete("/session/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(session_id: str) -> Response:
    _sessions.pop(session_id, None)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
