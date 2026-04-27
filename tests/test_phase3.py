import pytest

from src.agent_interface import run_guided_agent
from src.rag_interface import (
    apply_refinement_plan,
    build_state_from_intent,
    clarification_question_for_type,
    extract_intent,
    propose_refinement,
    retrieve_candidates,
)


def test_extract_intent_maps_basic_descriptors():
    intent = extract_intent("calm acoustic music for late-night studying")

    assert intent.acousticness is not None
    assert intent.prefer_acoustic is True
    assert intent.prefer_chill is True
    assert intent.clarification_type in {
        "genre_vs_mood",
        "genre",
        "mood",
        "general",
    }
    assert intent.confidence > 0


def test_retrieve_candidates_prefers_matching_tracks():
    songs = [
        {
            "id": 1,
            "title": "Acoustic Calm",
            "artist": "Artist A",
            "genre": "folk",
            "mood": "relaxed",
            "energy": 0.3,
            "tempo_bpm": 82,
            "valence": 0.55,
            "danceability": 0.45,
            "acousticness": 0.9,
        },
        {
            "id": 2,
            "title": "Loud Sprint",
            "artist": "Artist B",
            "genre": "metal",
            "mood": "intense",
            "energy": 0.95,
            "tempo_bpm": 160,
            "valence": 0.25,
            "danceability": 0.65,
            "acousticness": 0.05,
        },
    ]

    intent = extract_intent("calm acoustic music")
    candidates = retrieve_candidates(songs, intent, top_n=2)

    assert candidates[0]["title"] == "Acoustic Calm"


def test_guided_agent_can_run_one_refinement_round():
    songs = [
        {
            "id": 1,
            "title": "Study Breeze",
            "artist": "Artist A",
            "genre": "lofi",
            "mood": "chill",
            "energy": 0.35,
            "tempo_bpm": 78,
            "valence": 0.55,
            "danceability": 0.60,
            "acousticness": 0.82,
        },
        {
            "id": 2,
            "title": "Bright Rush",
            "artist": "Artist B",
            "genre": "pop",
            "mood": "happy",
            "energy": 0.82,
            "tempo_bpm": 122,
            "valence": 0.84,
            "danceability": 0.80,
            "acousticness": 0.18,
        },
    ]

    prompts = iter(
        [
            "calm study music",
            "lofi",
            "n",
            "n",
        ]
    )
    outputs = []

    def fake_input(prompt: str) -> str:
        outputs.append(prompt)
        return next(prompts)

    def fake_output(message: str) -> None:
        outputs.append(message)

    recommendations = run_guided_agent(
        songs,
        input_fn=fake_input,
        output_fn=fake_output,
        candidate_limit=2,
        recommendation_limit=1,
    )

    assert recommendations
    assert recommendations[0][0]["title"] == "Study Breeze"


def test_refinement_plan_from_feedback_contains_updates_and_weight_hint():
    base_intent = extract_intent("chill lofi music")
    state = build_state_from_intent(base_intent)

    plan = propose_refinement(state, "more acoustic and less energy")

    assert plan.action in {"refine", "clarify"}
    assert (
        "acousticness" in plan.suggested_updates or "acousticness" in plan.weight_hints
    )
    assert "energy" in plan.suggested_updates or "energy" in plan.weight_hints


def test_apply_refinement_plan_updates_state_and_normalizes_weights():
    base_intent = extract_intent("chill lofi music")
    state = build_state_from_intent(base_intent)
    original_weight_sum = sum(state.weights.values())

    plan = propose_refinement(
        state,
        "more acoustic, less energy, prioritize mood",
    )
    updated_state, change_log = apply_refinement_plan(state, plan)

    assert sum(updated_state.weights.values()) == pytest.approx(original_weight_sum)
    assert len(change_log) >= 1


def test_clarification_type_uses_fixed_templates_for_repeats():
    first_prompt = clarification_question_for_type("genre_vs_mood", repeat_index=0)
    second_prompt = clarification_question_for_type("genre_vs_mood", repeat_index=1)

    assert first_prompt != second_prompt
    assert "genre" in first_prompt.lower() or "mood" in first_prompt.lower()


def test_refinement_plan_uses_clarification_type(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    base_intent = extract_intent("chill lofi music")
    state = build_state_from_intent(base_intent)

    plan = propose_refinement(state, "not sure yet")

    assert plan.action == "clarify"
    assert plan.clarification_type in {"genre_vs_mood", "general"}
    assert plan.clarifying_question == clarification_question_for_type(
        plan.clarification_type or "general"
    )
