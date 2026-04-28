import pytest

from src.agent_interface import run_guided_agent, run_quick_recommendation
from src.rag_interface import (
    RefinementPlan,
    _intent_from_mapping,
    apply_refinement_plan,
    build_state_from_intent,
    clarification_question_for_type,
    extract_intent,
    get_llm_guardrail_status,
    propose_refinement,
    reset_llm_guardrails,
    retrieve_candidates,
)


@pytest.fixture(autouse=True)
def disable_llm_for_unit_tests(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("VIBEFINDER_SKIP_DOTENV", "true")
    reset_llm_guardrails(clear_cache=True)
    yield
    reset_llm_guardrails(clear_cache=True)


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


def test_quick_recommendation_prints_intent_diagnostics():
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
        }
    ]
    outputs = []

    run_quick_recommendation(
        songs,
        input_fn=lambda prompt: "calm lofi music",
        output_fn=outputs.append,
        candidate_limit=1,
        recommendation_limit=1,
    )

    joined_output = "\n".join(outputs)
    assert "Intent diagnostics" in joined_output
    assert "Source: HEURISTIC" in joined_output
    assert "Parsed scoring inputs:" in joined_output
    assert "Ranking criteria weights:" in joined_output


def test_forced_heuristic_mode_skips_llm_notes_and_budget():
    intent = extract_intent("poppy upbeat happy songs", mode="heuristic")
    status = get_llm_guardrail_status()

    assert intent.source == "heuristic"
    assert intent.notes == []
    assert intent.genre == "pop"
    assert status["calls_made"] == 0


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


def test_refinement_plan_uses_clarification_type():
    base_intent = extract_intent("chill lofi music")
    state = build_state_from_intent(base_intent)

    plan = propose_refinement(state, "not sure yet")

    assert plan.action == "clarify"
    assert plan.clarification_type in {"genre_vs_mood", "general"}
    assert plan.clarifying_question == clarification_question_for_type(
        plan.clarification_type or "general"
    )


def test_missing_api_key_does_not_consume_llm_call_budget():
    intent = extract_intent("calm acoustic music")
    status = get_llm_guardrail_status()

    assert intent.notes == ["Gemini intent parsing unavailable: no API key configured."]
    assert status["calls_made"] == 0
    assert status["calls_remaining"] == status["max_calls_per_session"]


def test_llm_mapping_rejects_unknown_categories_and_string_false_booleans():
    intent = _intent_from_mapping(
        "something weird",
        {
            "genre": "space polka",
            "mood": "focused",
            "avoid_intense": "false",
            "prefer_chill": "true",
            "prefer_acoustic": "no",
            "confidence": 0.9,
        },
    )

    assert intent.genre is None
    assert intent.mood == "focused"
    assert intent.avoid_intense is False
    assert intent.prefer_chill is True
    assert intent.prefer_acoustic is False
    assert intent.source == "llm"
    assert intent.llm_json is not None


def test_apply_refinement_plan_ignores_invalid_category_and_string_false_bool():
    state = build_state_from_intent(extract_intent("chill lofi music"))
    plan = RefinementPlan(
        suggested_updates={"genre": "space polka", "prefer_acoustic": "false"}
    )

    updated_state, change_log = apply_refinement_plan(state, plan)

    assert updated_state.genre == state.genre
    assert updated_state.prefer_acoustic is False
    assert "genre -> space polka" not in change_log
