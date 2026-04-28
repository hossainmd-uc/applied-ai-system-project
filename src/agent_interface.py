"""Terminal agent loop for guided recommendations."""

from __future__ import annotations

import json
from typing import Callable, Dict, List, Tuple

from .rag_interface import (
    RefinementPlan,
    apply_refinement_plan,
    clarification_question_for_type,
    build_state_from_intent,
    extract_intent,
    intent_from_state,
    propose_refinement,
    rank_candidates,
    recommend_with_state,
    retrieve_candidates,
    SearchIntent,
)


InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]


def run_quick_recommendation(
    songs: List[Dict],
    input_fn: InputFn = input,
    output_fn: OutputFn = print,
    candidate_limit: int = 10,
    recommendation_limit: int = 5,
    intent_mode: str = "llm",
) -> List[Tuple[Dict, float, str]]:
    query = input_fn("Describe what you want to hear: ").strip()
    if not query:
        output_fn("No request entered. Returning to the menu.")
        return []

    intent = extract_intent(query, mode=intent_mode)
    _announce_notes(intent.notes, output_fn)
    _display_intent_diagnostics(intent, output_fn)
    candidates = retrieve_candidates(songs, intent, top_n=candidate_limit)
    recommendations = rank_candidates(candidates, intent, k=recommendation_limit)
    _display_recommendations(recommendations, output_fn)
    return recommendations


def run_guided_agent(
    songs: List[Dict],
    input_fn: InputFn = input,
    output_fn: OutputFn = print,
    candidate_limit: int = 10,
    recommendation_limit: int = 5,
    intent_mode: str = "llm",
) -> List[Tuple[Dict, float, str]]:
    query = input_fn("What kind of music are you looking for? ").strip()
    if not query:
        output_fn("No request entered. Returning to the menu.")
        return []

    current_query = query
    last_recommendations: List[Tuple[Dict, float, str]] = []
    state = None
    clarification_counts: Dict[str, int] = {}

    while True:
        intent = extract_intent(current_query, mode=intent_mode)
        _announce_notes(intent.notes, output_fn)
        # Ask one clarification at a time and use the clarification type to pick
        # from fixed question templates so the same wording does not repeat.
        while intent.confidence < 0.75 and intent.clarification_type:
            clarification_type = intent.clarification_type or "general"
            repeat_index = clarification_counts.get(clarification_type, 0)
            clarification_prompt = clarification_question_for_type(
                clarification_type, repeat_index=repeat_index
            )
            clarification = input_fn(f"{clarification_prompt} ").strip()
            if clarification:
                current_query = f"{current_query} {clarification}"
                intent = extract_intent(current_query, mode=intent_mode)

            if not (intent.confidence < 0.75 and intent.clarification_type):
                clarification_counts.pop(clarification_type, None)
                break

            continue_clarifying = (
                input_fn(
                    "I can ask one more clarifying question before ranking. Continue? (y/n): "
                )
                .strip()
                .lower()
            )
            if continue_clarifying not in {"y", "yes"}:
                clarification_counts[clarification_type] = repeat_index + 1
                break

            clarification_counts[clarification_type] = repeat_index + 1

        # User chose to proceed or confidence improved: rank with best available signal.
        intent.clarifying_question = None
        _display_intent_diagnostics(intent, output_fn)
        state = build_state_from_intent(intent)

        candidates = retrieve_candidates(
            songs, intent_from_state(state), top_n=candidate_limit
        )
        last_recommendations = recommend_with_state(
            candidates, state, k=recommendation_limit
        )
        _display_recommendations(last_recommendations, output_fn)

        refine = (
            input_fn("Would you like to refine these results? (y/n): ").strip().lower()
        )
        if refine not in {"y", "yes"}:
            break

        adjustment = input_fn(
            "What should change? (for example: more acoustic, less intense): "
        ).strip()
        if not adjustment:
            break

        if state is None:
            continue

        plan = propose_refinement(state, adjustment, mode=intent_mode)
        _display_refinement_plan(plan, output_fn)

        if plan.action == "clarify":
            clarification_prompt = clarification_question_for_type(
                plan.clarification_type or "general"
            )
            clarification = input_fn(f"{clarification_prompt} ").strip()
            if clarification:
                plan = propose_refinement(state, clarification, mode=intent_mode)
                _display_refinement_plan(plan, output_fn)

        if plan.action == "finalize":
            output_fn("Keeping current recommendations as final.")
            break

        apply_changes = (
            input_fn("Apply these suggested changes? (y/n): ").strip().lower()
        )
        if apply_changes not in {"y", "yes"}:
            output_fn("Keeping the current state. You can refine again.")
            continue

        state, change_log = apply_refinement_plan(state, plan)
        if change_log:
            output_fn("Applied changes:")
            for change in change_log:
                output_fn(f"- {change}")
        else:
            output_fn("No valid changes were applied.")

        current_query = state.query

    return last_recommendations


def _display_recommendations(
    recommendations: List[Tuple[Dict, float, str]], output_fn: OutputFn
) -> None:
    if not recommendations:
        output_fn("No recommendations found.")
        return

    output_fn("\nTop recommendations\n")
    for index, (song, score, explanation) in enumerate(recommendations, start=1):
        reasons = [part.strip() for part in explanation.split("|") if part.strip()]
        output_fn(f"{index}. {song['title']} - {song['artist']}")
        output_fn(f"   Final Score: {score:.3f}")
        output_fn("   Reasons:")
        for reason in reasons:
            output_fn(f"   - {reason}")
        output_fn("-" * 60)


def _display_intent_diagnostics(intent: SearchIntent, output_fn: OutputFn) -> None:
    prefs = intent.to_user_prefs()

    output_fn("\nIntent diagnostics")
    output_fn(f"Source: {intent.source.upper()}")
    output_fn(f"Confidence: {intent.confidence:.2f}")
    if intent.clarification_type:
        output_fn(f"Clarification type: {intent.clarification_type}")
    if intent.llm_json:
        output_fn("LLM JSON:")
        output_fn(json.dumps(intent.llm_json, indent=2, sort_keys=True))

    output_fn("Parsed scoring inputs:")
    for key in (
        "genre",
        "mood",
        "energy",
        "acousticness",
        "tempo_bpm",
        "danceability",
        "valence",
        "likes_acoustic",
    ):
        value = prefs.get(key)
        if value not in ("", None):
            output_fn(f"- {key}: {value}")

    output_fn("Ranking criteria weights:")
    for key, value in prefs["weights"].items():
        output_fn(f"- {key}: {value:.2f}")


def _display_refinement_plan(plan: RefinementPlan, output_fn: OutputFn) -> None:
    output_fn("\nProposed refinement")
    output_fn(f"Action: {plan.action}")
    if plan.clarification_type:
        output_fn(f"Clarification type: {plan.clarification_type}")
    if plan.reason:
        output_fn(f"Reason: {plan.reason}")
    output_fn(f"Confidence: {plan.confidence:.2f}")
    if plan.suggested_updates:
        output_fn("Suggested updates:")
        for key, value in plan.suggested_updates.items():
            output_fn(f"- {key}: {value}")
    if plan.weight_hints:
        output_fn("Weight hints:")
        for key, value in plan.weight_hints.items():
            output_fn(f"- {key}: {value}")


def _announce_notes(notes: List[str], output_fn: OutputFn) -> None:
    if not notes:
        return

    output_fn("\nNote")
    for note in notes:
        output_fn(f"- {note}")
