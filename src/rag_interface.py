"""RAG-style retrieval and intent parsing for VibeFinder.

The retrieval step is intentionally lightweight: it extracts structured intent
from a user prompt, applies deterministic similarity heuristics to the CSV
catalog, and passes a candidate pool to the existing ranking engine.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .recommender import (
    DEFAULT_FEATURE_SIGMAS,
    DEFAULT_FEATURE_WEIGHTS,
    recommend_songs,
)


GENRE_KEYWORDS: Sequence[str] = (
    "indie pop",
    "hip hop",
    "synthwave",
    "classical",
    "cinematic",
    "ambient",
    "acoustic",
    "rock",
    "metal",
    "reggae",
    "jazz",
    "blues",
    "folk",
    "house",
    "lofi",
    "funk",
    "pop",
)

MOOD_KEYWORDS: Sequence[str] = (
    "euphoric",
    "reflective",
    "laid-back",
    "nostalgic",
    "confident",
    "playful",
    "focused",
    "relaxed",
    "intense",
    "moody",
    "smoky",
    "ominous",
    "chill",
    "happy",
    "defiant",
)

ENERGY_HINTS: Dict[str, float] = {
    "very low": 0.15,
    "low": 0.25,
    "calm": 0.25,
    "chill": 0.35,
    "relaxed": 0.35,
    "mellow": 0.35,
    "focused": 0.40,
    "mid": 0.55,
    "medium": 0.55,
    "balanced": 0.55,
    "upbeat": 0.72,
    "energetic": 0.80,
    "high": 0.85,
    "intense": 0.90,
}

ACOUSTIC_HINTS: Dict[str, float] = {
    "acoustic": 0.85,
    "organic": 0.80,
    "instrumental": 0.75,
    "lofi": 0.70,
    "electronic": 0.25,
    "synthetic": 0.20,
    "digital": 0.20,
    "not acoustic": 0.20,
}

TEMPO_HINTS: Dict[str, float] = {
    "slow": 72.0,
    "steady": 90.0,
    "mid tempo": 110.0,
    "medium tempo": 110.0,
    "fast": 136.0,
    "very fast": 150.0,
}

DANCEABILITY_HINTS: Dict[str, float] = {
    "danceable": 0.80,
    "groovy": 0.82,
    "rhythmic": 0.75,
    "laid-back": 0.55,
}

VALENCE_HINTS: Dict[str, float] = {
    "happy": 0.82,
    "positive": 0.78,
    "bright": 0.75,
    "sad": 0.25,
    "melancholy": 0.30,
    "moody": 0.40,
}

WEIGHT_LEVEL_VALUES: Dict[str, float] = {
    "low": 0.5,
    "medium": 1.0,
    "high": 1.5,
}

CLARIFICATION_TYPE_QUESTIONS: Dict[str, List[str]] = {
    "genre_vs_mood": [
        "Do you want me to prioritize genre, mood, or both?",
        "Should I focus more on the genre side, the mood side, or balance both?",
    ],
    "energy_vs_acoustic": [
        "Should I lean calmer and more acoustic, or more energetic and upbeat?",
        "Do you want the next pass to focus on calm acoustic vibes or higher energy?",
    ],
    "genre": [
        "Is there a genre you want me to prioritize?",
        "Which genre should I lean toward?",
    ],
    "mood": [
        "What mood should I prioritize?",
        "Which mood best fits what you want?",
    ],
    "energy": [
        "Should this be low-energy, mid-energy, or high-energy?",
        "Do you want this to feel calmer or more energetic?",
    ],
    "tempo": [
        "Should the songs be slower, medium tempo, or faster?",
        "Do you want a slower or quicker tempo?",
    ],
    "general": [
        "What should I adjust most: mood, energy, acousticness, genre, or tempo?",
        "Which direction should I refine first?",
    ],
}

CLARIFICATION_TYPES: Sequence[str] = tuple(CLARIFICATION_TYPE_QUESTIONS.keys())

# Local guardrails to prevent accidental request bursts during interactive use.
MAX_LLM_CALLS_PER_SESSION: int = int(os.getenv("VIBEFINDER_MAX_LLM_CALLS", "40"))
MIN_SECONDS_BETWEEN_LLM_CALLS: float = float(
    os.getenv("VIBEFINDER_LLM_COOLDOWN_SECONDS", "0.75")
)

_LLM_CALL_COUNT: int = 0
_LAST_LLM_CALL_MONOTONIC: float = 0.0
_INTENT_CACHE: Dict[str, SearchIntent] = {}


@dataclass
class SearchIntent:
    query: str
    genre: Optional[str] = None
    mood: Optional[str] = None
    energy: Optional[float] = None
    acousticness: Optional[float] = None
    tempo_bpm: Optional[float] = None
    danceability: Optional[float] = None
    valence: Optional[float] = None
    avoid_intense: bool = False
    prefer_chill: bool = False
    prefer_acoustic: bool = False
    confidence: float = 0.0
    clarification_type: Optional[str] = None
    clarifying_question: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def to_user_prefs(self) -> Dict[str, object]:
        prefs: Dict[str, object] = {
            "genre": self.genre or "",
            "mood": self.mood or "",
            "energy": self.energy if self.energy is not None else 0.50,
            "likes_acoustic": self.prefer_acoustic
            or (self.acousticness is not None and self.acousticness >= 0.5),
            "weights": dict(DEFAULT_FEATURE_WEIGHTS),
            "sigmas": dict(DEFAULT_FEATURE_SIGMAS),
        }
        if self.acousticness is not None:
            prefs["acousticness"] = self.acousticness
        if self.tempo_bpm is not None:
            prefs["tempo_bpm"] = self.tempo_bpm
        if self.danceability is not None:
            prefs["danceability"] = self.danceability
        if self.valence is not None:
            prefs["valence"] = self.valence
        return prefs


@dataclass
class RecommendationState:
    query: str
    genre: Optional[str] = None
    mood: Optional[str] = None
    energy: Optional[float] = None
    acousticness: Optional[float] = None
    tempo_bpm: Optional[float] = None
    danceability: Optional[float] = None
    valence: Optional[float] = None
    avoid_intense: bool = False
    prefer_chill: bool = False
    prefer_acoustic: bool = False
    weights: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_FEATURE_WEIGHTS)
    )

    def to_dict(self) -> Dict[str, object]:
        return {
            "query": self.query,
            "genre": self.genre,
            "mood": self.mood,
            "energy": self.energy,
            "acousticness": self.acousticness,
            "tempo_bpm": self.tempo_bpm,
            "danceability": self.danceability,
            "valence": self.valence,
            "avoid_intense": self.avoid_intense,
            "prefer_chill": self.prefer_chill,
            "prefer_acoustic": self.prefer_acoustic,
            "weights": dict(self.weights),
        }


@dataclass
class RefinementPlan:
    action: str = "refine"
    suggested_updates: Dict[str, object] = field(default_factory=dict)
    weight_hints: Dict[str, str] = field(default_factory=dict)
    clarification_type: Optional[str] = None
    confidence: float = 0.0
    clarifying_question: Optional[str] = None
    reason: str = ""


def extract_intent(query: str) -> SearchIntent:
    """Return structured intent using Gemini when available, else rules."""

    normalized_query = _normalize_text(query)
    if normalized_query in _INTENT_CACHE:
        cached = _clone_intent(_INTENT_CACHE[normalized_query])
        cached.notes.append("Using cached intent result for this query.")
        return cached

    llm_intent, failure_note = _extract_intent_with_gemini(query)
    if llm_intent is not None:
        _INTENT_CACHE[normalized_query] = _clone_intent(llm_intent)
        return llm_intent

    rule_intent = _extract_intent_with_rules(query)
    if failure_note:
        rule_intent.notes.append(failure_note)
    _INTENT_CACHE[normalized_query] = _clone_intent(rule_intent)
    return rule_intent


def get_llm_guardrail_status() -> Dict[str, object]:
    return {
        "max_calls_per_session": MAX_LLM_CALLS_PER_SESSION,
        "min_seconds_between_calls": MIN_SECONDS_BETWEEN_LLM_CALLS,
        "calls_made": _LLM_CALL_COUNT,
        "calls_remaining": max(0, MAX_LLM_CALLS_PER_SESSION - _LLM_CALL_COUNT),
        "cached_intent_queries": len(_INTENT_CACHE),
    }


def retrieve_candidates(
    songs: List[Dict], intent: SearchIntent, top_n: int = 10
) -> List[Dict]:
    scored_candidates: List[Tuple[Dict, float]] = []
    for song in songs:
        pre_score, _ = score_retrieval_candidate(song, intent)
        candidate = dict(song)
        candidate["_retrieval_score"] = pre_score
        scored_candidates.append((candidate, pre_score))

    ranked = sorted(scored_candidates, key=lambda item: (-item[1], item[0]["title"]))
    return [song for song, _ in ranked[:top_n]]


def rank_candidates(
    songs: List[Dict], intent: SearchIntent, k: int = 5
) -> List[Tuple[Dict, float, str]]:
    return recommend_songs(intent.to_user_prefs(), songs, k=k)


def build_state_from_intent(intent: SearchIntent) -> RecommendationState:
    return RecommendationState(
        query=intent.query,
        genre=intent.genre,
        mood=intent.mood,
        energy=intent.energy,
        acousticness=intent.acousticness,
        tempo_bpm=intent.tempo_bpm,
        danceability=intent.danceability,
        valence=intent.valence,
        avoid_intense=intent.avoid_intense,
        prefer_chill=intent.prefer_chill,
        prefer_acoustic=intent.prefer_acoustic,
        weights=dict(DEFAULT_FEATURE_WEIGHTS),
    )


def intent_from_state(state: RecommendationState) -> SearchIntent:
    intent = SearchIntent(query=state.query)
    intent.genre = state.genre
    intent.mood = state.mood
    intent.energy = state.energy
    intent.acousticness = state.acousticness
    intent.tempo_bpm = state.tempo_bpm
    intent.danceability = state.danceability
    intent.valence = state.valence
    intent.avoid_intense = state.avoid_intense
    intent.prefer_chill = state.prefer_chill
    intent.prefer_acoustic = state.prefer_acoustic
    intent.confidence = 1.0
    return intent


def recommend_with_state(
    songs: List[Dict], state: RecommendationState, k: int = 5
) -> List[Tuple[Dict, float, str]]:
    intent = intent_from_state(state)
    prefs = intent.to_user_prefs()
    prefs["weights"] = dict(state.weights)
    return recommend_songs(prefs, songs, k=k)


def propose_refinement(
    state: RecommendationState, user_feedback: str
) -> RefinementPlan:
    plan, failure_note = _propose_refinement_with_gemini(state, user_feedback)
    if plan is not None:
        return plan

    rule_plan = _propose_refinement_with_rules(state, user_feedback)
    if failure_note:
        rule_plan.reason = f"{rule_plan.reason} {failure_note}".strip()
    return rule_plan


def clarification_question_for_type(
    clarification_type: Optional[str],
    repeat_index: int = 0,
) -> str:
    normalized_type = _normalize(clarification_type or "general")
    if normalized_type not in CLARIFICATION_TYPE_QUESTIONS:
        normalized_type = "general"

    prompts = CLARIFICATION_TYPE_QUESTIONS[normalized_type]
    index = min(max(repeat_index, 0), len(prompts) - 1)
    return prompts[index]


def apply_refinement_plan(
    state: RecommendationState, plan: RefinementPlan
) -> Tuple[RecommendationState, List[str]]:
    updated_state = RecommendationState(**state.to_dict())
    change_log: List[str] = []

    for key, value in plan.suggested_updates.items():
        if key == "query" and isinstance(value, str) and value.strip():
            updated_state.query = value.strip()
            change_log.append(f"query -> {updated_state.query}")
            continue

        if key in {"genre", "mood"} and isinstance(value, str):
            normalized = _normalize(value)
            if normalized:
                setattr(updated_state, key, normalized)
                change_log.append(f"{key} -> {normalized}")
            continue

        if key in {"energy", "acousticness", "danceability", "valence"}:
            bounded = _bounded_float(value)
            if bounded is not None:
                setattr(updated_state, key, bounded)
                change_log.append(f"{key} -> {bounded:.2f}")
            continue

        if key == "tempo_bpm":
            tempo = _positive_float(value)
            if tempo is not None:
                updated_state.tempo_bpm = tempo
                change_log.append(f"tempo_bpm -> {tempo:.1f}")
            continue

        if key in {"avoid_intense", "prefer_chill", "prefer_acoustic"}:
            setattr(updated_state, key, bool(value))
            change_log.append(f"{key} -> {bool(value)}")

    if plan.weight_hints:
        for feature, level in plan.weight_hints.items():
            if feature not in updated_state.weights:
                continue
            multiplier = WEIGHT_LEVEL_VALUES.get(_normalize(level))
            if multiplier is None:
                continue
            updated_state.weights[feature] = updated_state.weights[feature] * multiplier
            change_log.append(f"weight hint: {feature} -> {level}")
        _normalize_weights(updated_state.weights)

    return updated_state, change_log


def score_retrieval_candidate(song: Dict, intent: SearchIntent) -> Tuple[float, str]:
    score = 0.0
    reasons: List[str] = []

    if intent.genre:
        genre_match = 1.0 if _normalize(song.get("genre")) == intent.genre else 0.0
        score += 0.35 * genre_match
        reasons.append(f"genre match={genre_match:.0f}")

    if intent.mood:
        mood_match = 1.0 if _normalize(song.get("mood")) == intent.mood else 0.0
        score += 0.25 * mood_match
        reasons.append(f"mood match={mood_match:.0f}")

    numeric_components: List[float] = []
    for feature, target in (
        ("energy", intent.energy),
        ("acousticness", intent.acousticness),
        ("tempo_bpm", intent.tempo_bpm),
        ("danceability", intent.danceability),
        ("valence", intent.valence),
    ):
        if target is None:
            continue
        song_value = _to_float(song.get(feature))
        if song_value is None:
            continue
        sigma = DEFAULT_FEATURE_SIGMAS.get(feature, 0.2)
        closeness = math.exp(-(((song_value - target) ** 2) / (2 * (sigma**2))))
        numeric_components.append(closeness)

    if numeric_components:
        numeric_closeness_avg = sum(numeric_components) / len(numeric_components)
        score += 0.40 * numeric_closeness_avg
        reasons.append(f"numeric closeness={numeric_closeness_avg:.2f}")

    penalty = 0.0
    if intent.avoid_intense:
        song_mood = _normalize(song.get("mood"))
        song_energy = _to_float(song.get("energy")) or 0.0
        if song_mood == "intense" or song_energy >= 0.85:
            penalty += 0.20

    if intent.prefer_chill:
        song_energy = _to_float(song.get("energy")) or 0.0
        if song_energy > 0.70:
            penalty += 0.10

    if intent.prefer_acoustic:
        song_acousticness = _to_float(song.get("acousticness")) or 0.0
        if song_acousticness < 0.55:
            penalty += 0.10

    final_score = max(0.0, score - penalty)
    reason_text = ", ".join(reasons) if reasons else "no strong match signals"
    if penalty > 0:
        reason_text = f"{reason_text}; penalty={penalty:.2f}"
    return final_score, reason_text


def _extract_intent_with_rules(query: str) -> SearchIntent:
    normalized = _normalize_text(query)
    intent = SearchIntent(query=query)

    intent.genre = _find_keyword(normalized, GENRE_KEYWORDS)
    intent.mood = _find_keyword(normalized, MOOD_KEYWORDS)

    intent.energy = _find_hint(normalized, ENERGY_HINTS)
    intent.acousticness = _find_hint(normalized, ACOUSTIC_HINTS)
    intent.tempo_bpm = _find_hint(normalized, TEMPO_HINTS)
    intent.danceability = _find_hint(normalized, DANCEABILITY_HINTS)
    intent.valence = _find_hint(normalized, VALENCE_HINTS)

    intent.avoid_intense = any(
        phrase in normalized
        for phrase in ("not intense", "avoid intense", "less intense")
    )
    intent.prefer_chill = any(
        phrase in normalized
        for phrase in ("chill", "calm", "relax", "relaxed", "laid back")
    )
    intent.prefer_acoustic = any(
        phrase in normalized for phrase in ("acoustic", "organic", "unplugged")
    )

    matched_signals = sum(
        1
        for value in (
            intent.genre,
            intent.mood,
            intent.energy,
            intent.acousticness,
            intent.tempo_bpm,
            intent.danceability,
            intent.valence,
        )
        if value is not None
    )
    intent.confidence = min(1.0, 0.15 + 0.12 * matched_signals)

    missing_focus = []
    if intent.genre is None:
        missing_focus.append("genre")
    if intent.mood is None:
        missing_focus.append("mood")
    if intent.energy is None:
        missing_focus.append("energy")
    if intent.acousticness is None:
        missing_focus.append("acousticness")

    if missing_focus:
        intent.clarification_type = _clarification_type_from_missing(missing_focus)
        intent.clarifying_question = clarification_question_for_type(
            intent.clarification_type
        )
    else:
        intent.clarification_type = None
        intent.clarifying_question = None

    return intent


def _extract_intent_with_gemini(
    query: str,
) -> Tuple[Optional[SearchIntent], Optional[str]]:
    guardrail_note = _guardrail_before_llm_call()
    if guardrail_note is not None:
        return None, guardrail_note

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None, "Gemini intent parsing unavailable: no API key configured."

    try:
        import urllib.error
        import urllib.request
    except ImportError:
        return None, "Gemini intent parsing unavailable: HTTP client support missing."

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    prompt = (
        "Extract music search intent from the user's request. Return strict JSON with keys "
        "genre, mood, energy, acousticness, tempo_bpm, danceability, valence, "
        "avoid_intense, prefer_chill, prefer_acoustic, confidence, clarification_type, clarifying_question. "
        "clarification_type must be one of genre_vs_mood, energy_vs_acoustic, genre, mood, energy, tempo, general. "
        "Use null for unknown numeric/categorical fields. Return only JSON. User request: "
        f"{query}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 512,
            "responseMimeType": "application/json",
        },
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return None, f"Gemini intent parsing failed: HTTP {exc.code}."
    except (urllib.error.URLError, TimeoutError, ValueError):
        return None, "Gemini intent parsing failed: network or decode error."

    text = _extract_response_text(data)
    if not text:
        return None, "Gemini intent parsing failed: empty model response."

    parsed = _parse_json_object(text)
    if not parsed:
        return None, "Gemini intent parsing failed: response was not valid JSON."

    return _intent_from_mapping(query, parsed), None


def _propose_refinement_with_gemini(
    state: RecommendationState, user_feedback: str
) -> Tuple[Optional[RefinementPlan], Optional[str]]:
    guardrail_note = _guardrail_before_llm_call()
    if guardrail_note is not None:
        return None, guardrail_note

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None, "Gemini refinement unavailable: no API key configured."

    try:
        import urllib.error
        import urllib.request
    except ImportError:
        return None, "Gemini refinement unavailable: HTTP client support missing."

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    prompt = (
        "You are helping refine a music recommendation state. "
        "Return strict JSON with keys: action, suggested_updates, weight_hints, "
        "clarification_type, confidence, reason. "
        "Rules: action must be one of clarify/refine/finalize. "
        "suggested_updates must include only changed fields. "
        "weight_hints values must be low/medium/high. "
        "clarification_type must be one of: genre_vs_mood, energy_vs_acoustic, genre, mood, energy, tempo, general. "
        "Do not invent question text; the application will choose the wording. "
        "Current state JSON: "
        f"{json.dumps(state.to_dict(), ensure_ascii=True)}. "
        "User feedback: "
        f"{user_feedback}."
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 512,
            "responseMimeType": "application/json",
        },
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return None, f"Gemini refinement failed: HTTP {exc.code}."
    except (urllib.error.URLError, TimeoutError, ValueError):
        return None, "Gemini refinement failed: network or decode error."

    text = _extract_response_text(data)
    if not text:
        return None, "Gemini refinement failed: empty model response."
    parsed = _parse_json_object(text)
    if not parsed:
        return None, "Gemini refinement failed: response was not valid JSON."
    return _refinement_plan_from_mapping(parsed), None


def _propose_refinement_with_rules(
    state: RecommendationState, user_feedback: str
) -> RefinementPlan:
    clarification_type = _pick_clarification_type(user_feedback)
    if _feedback_needs_clarification(user_feedback):
        return RefinementPlan(
            action="clarify",
            suggested_updates={},
            weight_hints={},
            clarification_type=clarification_type,
            confidence=0.35,
            clarifying_question=clarification_question_for_type(clarification_type),
            reason="Rule-based refinement needs a clarifying follow-up.",
        )

    feedback_intent = _extract_intent_with_rules(user_feedback)
    updates: Dict[str, object] = {}

    for feature in (
        "genre",
        "mood",
        "energy",
        "acousticness",
        "tempo_bpm",
        "danceability",
        "valence",
    ):
        value = getattr(feedback_intent, feature)
        if value is not None:
            updates[feature] = value

    if feedback_intent.avoid_intense != state.avoid_intense:
        updates["avoid_intense"] = feedback_intent.avoid_intense
    if feedback_intent.prefer_chill != state.prefer_chill:
        updates["prefer_chill"] = feedback_intent.prefer_chill
    if feedback_intent.prefer_acoustic != state.prefer_acoustic:
        updates["prefer_acoustic"] = feedback_intent.prefer_acoustic

    weight_hints = _extract_weight_hints_from_feedback(user_feedback)
    action = "refine" if updates or weight_hints else "clarify"
    clarification_type = _derive_clarification_type(
        state, feedback_intent, user_feedback
    )
    question = (
        clarification_question_for_type(clarification_type)
        if action == "clarify"
        else None
    )

    return RefinementPlan(
        action=action,
        suggested_updates=updates,
        weight_hints=weight_hints,
        clarification_type=clarification_type,
        confidence=min(1.0, 0.4 + 0.1 * len(updates) + 0.1 * len(weight_hints)),
        clarifying_question=question,
        reason="Rule-based refinement from user feedback.",
    )


def _refinement_plan_from_mapping(mapping: Dict[str, Any]) -> RefinementPlan:
    action = _normalize(mapping.get("action") or "refine")
    if action not in {"clarify", "refine", "finalize"}:
        action = "refine"

    raw_updates = mapping.get("suggested_updates")
    suggested_updates = raw_updates if isinstance(raw_updates, dict) else {}

    raw_hints = mapping.get("weight_hints")
    weight_hints: Dict[str, str] = {}
    if isinstance(raw_hints, dict):
        for feature, level in raw_hints.items():
            if feature in DEFAULT_FEATURE_WEIGHTS and isinstance(level, str):
                normalized_level = _normalize(level)
                if normalized_level in WEIGHT_LEVEL_VALUES:
                    weight_hints[feature] = normalized_level

    clarification_type = _normalize(mapping.get("clarification_type") or "")
    if clarification_type not in CLARIFICATION_TYPE_QUESTIONS:
        clarification_type = None

    confidence = _bounded_float(mapping.get("confidence")) or 0.5
    reason = mapping.get("reason")

    return RefinementPlan(
        action=action,
        suggested_updates=suggested_updates,
        weight_hints=weight_hints,
        clarification_type=clarification_type,
        confidence=confidence,
        clarifying_question=None,
        reason=reason.strip() if isinstance(reason, str) else "",
    )


def _extract_response_text(data: Dict) -> Optional[str]:
    candidates = data.get("candidates") or []
    for candidate in candidates:
        content = candidate.get("content") or {}
        parts = content.get("parts") or []
        for part in parts:
            text = part.get("text")
            if text:
                return text
    return None


def _parse_json_object(text: str) -> Optional[Dict]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def _intent_from_mapping(query: str, mapping: Dict) -> SearchIntent:
    intent = SearchIntent(query=query)
    intent.genre = _normalize(mapping.get("genre")) or None
    intent.mood = _normalize(mapping.get("mood")) or None
    intent.energy = _bounded_float(mapping.get("energy"))
    intent.acousticness = _bounded_float(mapping.get("acousticness"))
    intent.tempo_bpm = _positive_float(mapping.get("tempo_bpm"))
    intent.danceability = _bounded_float(mapping.get("danceability"))
    intent.valence = _bounded_float(mapping.get("valence"))
    intent.avoid_intense = bool(mapping.get("avoid_intense", False))
    intent.prefer_chill = bool(mapping.get("prefer_chill", False))
    intent.prefer_acoustic = bool(mapping.get("prefer_acoustic", False))
    intent.confidence = _bounded_float(mapping.get("confidence")) or 0.5
    clarification_type = _normalize(mapping.get("clarification_type") or "")
    if clarification_type not in CLARIFICATION_TYPE_QUESTIONS:
        clarification_type = None
    intent.clarification_type = clarification_type
    question = mapping.get("clarifying_question")
    if isinstance(question, str) and question.strip():
        intent.clarifying_question = question.strip()
    return intent


def _find_keyword(text: str, keywords: Sequence[str]) -> Optional[str]:
    for keyword in keywords:
        if keyword in text:
            return keyword
    return None


def _find_hint(text: str, hints: Dict[str, float]) -> Optional[float]:
    for phrase, value in hints.items():
        if phrase in text:
            return value
    return None


def _clarifying_question(missing_focus: List[str]) -> str:
    if "genre" in missing_focus and "mood" in missing_focus:
        return "Do you want to guide this by genre, mood, or both?"
    if "energy" in missing_focus and "acousticness" in missing_focus:
        return "Should I lean calmer and more acoustic, or more energetic and upbeat?"
    if "genre" in missing_focus:
        return "Is there a genre you want me to prioritize?"
    if "mood" in missing_focus:
        return "What mood should I prioritize?"
    if "energy" in missing_focus:
        return "Should this be low-energy, mid-energy, or high-energy?"
    return "Do you want to add any more details before I rank songs?"


def _clarification_type_from_missing(missing_focus: List[str]) -> str:
    if "genre" in missing_focus and "mood" in missing_focus:
        return "genre_vs_mood"
    if "energy" in missing_focus and "acousticness" in missing_focus:
        return "energy_vs_acoustic"
    if "tempo" in missing_focus:
        return "tempo"
    if "genre" in missing_focus:
        return "genre"
    if "mood" in missing_focus:
        return "mood"
    if "energy" in missing_focus:
        return "energy"
    return "general"


def _derive_clarification_type(
    state: RecommendationState, feedback_intent: SearchIntent, user_feedback: str
) -> str:
    text = _normalize_text(user_feedback)
    missing_genre = state.genre is None or feedback_intent.genre is None
    missing_mood = state.mood is None or feedback_intent.mood is None
    missing_energy = state.energy is None or feedback_intent.energy is None
    missing_acoustic = (
        state.acousticness is None or feedback_intent.acousticness is None
    )
    missing_tempo = state.tempo_bpm is None or feedback_intent.tempo_bpm is None

    if missing_genre and missing_mood:
        return "genre_vs_mood"
    if missing_energy and missing_acoustic:
        return "energy_vs_acoustic"
    if missing_tempo or "tempo" in text or "speed" in text:
        return "tempo"
    if missing_mood:
        return "mood"
    if missing_genre:
        return "genre"
    if missing_energy:
        return "energy"
    return "general"


def _clarification_question_from_plan(
    clarification_type: Optional[str], repeat_index: int = 0
) -> str:
    normalized_type = _normalize(clarification_type or "general")
    if normalized_type not in CLARIFICATION_TYPE_QUESTIONS:
        normalized_type = "general"

    prompts = CLARIFICATION_TYPE_QUESTIONS[normalized_type]
    index = min(max(repeat_index, 0), len(prompts) - 1)
    return prompts[index]


def _pick_clarification_type(feedback: str) -> str:
    text = _normalize_text(feedback)
    if any(token in text for token in ("genre or mood", "both", "style")):
        return "genre_vs_mood"
    if any(token in text for token in ("acoustic", "calm", "energy")):
        return "energy_vs_acoustic"
    if "tempo" in text or "speed" in text:
        return "tempo"
    if "mood" in text:
        return "mood"
    if "genre" in text:
        return "genre"
    if "acoustic" in text:
        return "acoustic_priority"
    if "energy" in text:
        return "energy"
    return "general"


def _feedback_needs_clarification(feedback: str) -> bool:
    text = _normalize_text(feedback)
    actionable_tokens = (
        "more ",
        "less ",
        "prioritize",
        "deprioritize",
        "balance",
        "calmer",
        "more acoustic",
        "more energetic",
        "slower",
        "faster",
    )
    if any(token in text for token in actionable_tokens):
        return False

    meta_tokens = (
        "not sure",
        "unsure",
        "guidance",
        "guide",
        "help",
        "both",
        "either",
        "what do you recommend",
        "genre or mood",
    )
    return any(token in text for token in meta_tokens)


def _guardrail_before_llm_call() -> Optional[str]:
    global _LLM_CALL_COUNT, _LAST_LLM_CALL_MONOTONIC

    now = time.monotonic()
    if _LLM_CALL_COUNT >= MAX_LLM_CALLS_PER_SESSION:
        return (
            "LLM guardrail: session call limit reached; using heuristic fallback. "
            "Increase VIBEFINDER_MAX_LLM_CALLS to allow more requests."
        )

    if _LAST_LLM_CALL_MONOTONIC > 0:
        elapsed = now - _LAST_LLM_CALL_MONOTONIC
        if elapsed < MIN_SECONDS_BETWEEN_LLM_CALLS:
            wait_seconds = MIN_SECONDS_BETWEEN_LLM_CALLS - elapsed
            return (
                "LLM guardrail: cooldown active; using heuristic fallback. "
                f"Try again in {wait_seconds:.2f}s."
            )

    _LLM_CALL_COUNT += 1
    _LAST_LLM_CALL_MONOTONIC = now
    return None


def _clone_intent(intent: SearchIntent) -> SearchIntent:
    return SearchIntent(
        query=intent.query,
        genre=intent.genre,
        mood=intent.mood,
        energy=intent.energy,
        acousticness=intent.acousticness,
        tempo_bpm=intent.tempo_bpm,
        danceability=intent.danceability,
        valence=intent.valence,
        avoid_intense=intent.avoid_intense,
        prefer_chill=intent.prefer_chill,
        prefer_acoustic=intent.prefer_acoustic,
        confidence=intent.confidence,
        clarification_type=intent.clarification_type,
        clarifying_question=intent.clarifying_question,
        notes=list(intent.notes),
    )


def _extract_weight_hints_from_feedback(feedback: str) -> Dict[str, str]:
    text = _normalize_text(feedback)
    hints: Dict[str, str] = {}
    feature_tokens = {
        "genre": "genre",
        "mood": "mood",
        "energy": "energy",
        "acousticness": "acoustic",
        "tempo_bpm": "tempo",
        "danceability": "dance",
        "valence": "valence",
    }

    for feature, token in feature_tokens.items():
        if f"more {token}" in text or f"prioritize {token}" in text:
            hints[feature] = "high"
        elif f"less {token}" in text or f"deprioritize {token}" in text:
            hints[feature] = "low"
        elif f"balance {token}" in text:
            hints[feature] = "medium"

    return hints


def _normalize_weights(weights: Dict[str, float]) -> None:
    total = sum(value for value in weights.values() if value > 0)
    if total <= 0:
        for key, value in DEFAULT_FEATURE_WEIGHTS.items():
            weights[key] = value
        return
    for key in list(weights.keys()):
        value = max(0.0, float(weights[key]))
        weights[key] = value / total


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _normalize(value: object) -> str:
    return str(value).strip().lower()


def _to_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bounded_float(value: object) -> Optional[float]:
    numeric_value = _to_float(value)
    if numeric_value is None:
        return None
    return max(0.0, min(1.0, numeric_value))


def _positive_float(value: object) -> Optional[float]:
    numeric_value = _to_float(value)
    if numeric_value is None or numeric_value <= 0:
        return None
    return numeric_value
