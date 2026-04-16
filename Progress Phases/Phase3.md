# Phase 3: RAG + Agentic TUI Integration Plan

## Phase 3 Goal

Upgrade VibeFinder from a fixed preference simulation into an interactive AI-assisted recommender that runs in the terminal and supports:

- a quick one-turn recommendation flow,
- and a guided multi-turn refinement flow.

The key design choice is that Retrieval-Augmented Generation (RAG) and Agentic Workflow are not separate systems in this project. The agentic loop uses retrieval as a core tool.

## Core Decision: Unified Backend, Two User Modes

Instead of building two independent pipelines, Phase 3 will use one shared recommendation engine with two interaction styles:

1. Quick Recommend Mode (single-turn)
- User enters one natural language request.
- System extracts intent, retrieves candidates, scores songs, and returns top results.

2. Guided Agent Mode (multi-turn)
- User enters a request.
- System asks clarifying questions when needed.
- System retrieves, ranks, explains, and refines across multiple turns.

This keeps implementation cleaner, prevents duplicated logic, and makes testing easier.

## Unified Data Flow (Input -> Process -> Output)

### 1. Input Layer (TUI)

- User launches app in terminal.
- User selects mode:
  - Quick Recommend
  - Guided Agent
- User provides natural language prompt (example: "calm acoustic music for late-night studying").

### 2. Understanding Layer (LLM Intent Extraction)

- Convert user text into structured preference signals:
  - preferred genres/moods
  - target ranges for energy, tempo, acousticness, danceability, valence
  - optional constraints (exclude intense songs, prioritize chill, etc.)
- Output a normalized preference object compatible with current scoring inputs.

### 3. Retrieval Layer (RAG over CSV Catalog)

- Retrieve relevant songs from `data/songs.csv` based on extracted intent.
- Retrieval should use metadata filtering plus a lightweight pre-score to estimate how close each song is to the user intent before full ranking.
- In this document, "similarity heuristics" means simple, deterministic rules such as:
  - Categorical boosts: add points when genre or mood matches requested values.
  - Numeric closeness: add points when song values are close to target ranges (energy, tempo, acousticness, danceability, valence).
  - Constraint penalties: subtract points (or filter out songs) when a song violates explicit constraints like "not intense" or "low acousticness."
  - Optional artist diversity rule: avoid returning too many songs from the same artist in the candidate pool.
- Example retrieval pre-score (for candidate selection only):
  - `pre_score = 0.35*genre_match + 0.25*mood_match + 0.40*numeric_closeness_avg`
  - Keep top N candidates by `pre_score`, then pass those to the existing weighted recommender for final scoring.
- Return a candidate pool instead of scoring the full catalog every time.

### 4. Ranking Layer (Existing Recommender Logic)

- Score retrieved candidates with existing weighted + Gaussian scoring.
- Preserve explainability by keeping per-feature contribution reasons.
- Return top-k ranked recommendations.

### 5. Agentic Orchestration Layer (Guided Mode)

- Decide whether to ask clarification before final ranking.
- Run iterative refine loop:
  - propose songs,
  - receive user feedback,
  - adjust constraints/targets,
  - rerank.
- End when user accepts recommendations or exits.

### 6. Output Layer

- Terminal output includes:
  - ranked songs,
  - final score,
  - short explanation for each recommendation,
  - optional "why this changed" messages in guided refinements.

## Proposed Module Responsibilities

1. `src/recommender.py` (existing, shared core)
- Keep current scoring/ranking logic as source of truth.

2. `src/rag_interface.py` (new)
- Natural language intent extraction.
- Candidate retrieval from CSV catalog.
- Returns structured preferences + candidate set.

3. `src/agent_interface.py` (new)
- Multi-turn conversation loop for guided mode.
- Clarification logic and refinement state.
- Calls retrieval and recommender functions each turn.

4. `src/main.py` (update)
- TUI mode menu.
- Dispatch to quick mode or guided mode.

## Minimal Phase 3 Interaction Design (Terminal)

### On startup

- Show menu:
  - `1) Quick Recommend`
  - `2) Guided Agent`
  - `3) Exit`

### Quick Recommend

- Prompt once for user request.
- Return top results with concise reasons.

### Guided Agent

- Prompt for request.
- Ask 1-2 targeted clarifying questions only when confidence is low.
- Show recommendations.
- Ask whether user wants refinement.
- Repeat until user accepts or exits.

## Reliability and Evaluation Plan for Phase 3

To keep the new AI layer reliable and testable:

1. Deterministic fallback behavior
- If API key/model call fails, fallback to rule-based keyword mapping and existing scoring.

2. Structured output validation
- Require parsed intent JSON schema (or equivalent strict fields).
- Validate numeric bounds and allowed categorical values.

3. Test coverage additions
- Add tests for:
  - intent-to-profile mapping,
  - retrieval candidate filtering,
  - guided loop state transitions,
  - fallback path behavior.

4. Prompt and response logging (local)
- Save minimal debug logs for development (no sensitive data).

## Risks and Mitigations

- Risk: Hallucinated feature values from LLM.
  - Mitigation: strict parsing, range validation, and defaults.

- Risk: Over-complication for small CSV catalog.
  - Mitigation: keep retrieval simple and reuse existing scorer.

- Risk: Conversation loops become verbose.
  - Mitigation: cap clarifying questions and turns.

## Phase 3 Implementation Sequence

1. Add TUI mode selection in `main.py`.
2. Implement `rag_interface.py` for intent extraction + candidate retrieval.
3. Connect RAG output to existing recommender scoring.
4. Implement `agent_interface.py` guided refinement loop.
5. Add fallback rule-based parser for no-API or API-failure cases.
6. Add/extend tests for new interfaces and fallback behavior.
7. Validate end-to-end flows in both modes.

## Phase 3 Definition of Done

Phase 3 is complete when:

- Both TUI modes run successfully from terminal.
- RAG retrieval feeds the same shared ranking engine.
- Guided mode can perform at least one refinement turn.
- Fallback behavior works without external API.
- Tests cover core new logic and pass.

## Expected Outcome

By the end of Phase 3, VibeFinder will remain a transparent, explainable recommender while adding practical AI interaction capabilities:

- RAG improves natural language usability,
- Agentic flow improves iterative personalization,
- and one unified architecture keeps the system maintainable.
