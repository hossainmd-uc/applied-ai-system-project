# VibeFinder Demo Walkthrough

This document is a walkthrough script for explaining VibeFinder from start to finish. It is written for someone seeing the project for the first time.

## 1. What VibeFinder Is

VibeFinder is an AI-assisted music recommender that runs in the terminal. It started as a Modules 1-3 **Music Recommender Simulation**, where songs were ranked using a weighted content-based scoring system.

The final version keeps that original recommender, but adds an AI layer that lets users describe music in natural language. The system can use Gemini to interpret a request, or it can fall back to a simpler heuristic parser when the LLM is unavailable.

## 2. High-Level Architecture

```text
User
 |
 | natural language prompt
 v
Terminal menu
 |
 | Heuristic Quick, Heuristic Guided, LLM Quick, or LLM Guided
 v
Intent extraction
 |
 | LLM JSON or heuristic fallback
 v
Candidate retrieval
 |
 | smaller set of likely matching songs
 v
Weighted ranking algorithm
 |
 | final scores + explanations
 v
Top recommendations
```

The important design choice is that the LLM does **not** directly choose the songs. It translates the user's request into structured fields, and the existing recommender performs the final ranking.

## 3. Step-By-Step System Flow

### Step 1: User Chooses A Mode

When the app starts, the user sees:

```text
VibeFinder
1) Quick Recommend (Heuristic)
2) Guided Agent (Heuristic)
3) Quick Recommend (LLM)
4) Guided Agent (LLM)
5) Exit
```

`Quick Recommend` is a one-turn flow. The user enters one request and gets recommendations.

`Guided Agent` is iterative. The user can answer clarifying questions or ask the system to refine the results.

`Heuristic` mode forces the local keyword parser. `LLM` mode tries Gemini first, which is useful for demonstrating structured JSON parsing.

### Step 2: User Enters A Natural Language Request

Example:

```text
Looking for poppy upbeat and happy songs
```

This text is not directly useful to the scoring algorithm. The ranker needs structured values such as `genre`, `mood`, `energy`, and `valence`.

### Step 3: The Understanding Layer Parses Intent

The system first tries to use Gemini, if an API key is available.

The LLM receives:

- the user's natural language request,
- instructions to return only structured JSON,
- allowed categories for fields like genre and mood,
- numeric expectations for fields like energy and valence.

Expected LLM output looks like:

```json
{
  "genre": "pop",
  "mood": "happy",
  "energy": 0.8,
  "acousticness": null,
  "tempo_bpm": null,
  "danceability": null,
  "valence": 0.9,
  "avoid_intense": false,
  "prefer_chill": false,
  "prefer_acoustic": false,
  "confidence": 0.9,
  "clarification_type": null,
  "clarifying_question": null
}
```

The app validates and normalizes these fields before scoring. This matters because LLM output can be unexpected, incomplete, or malformed.

### Step 4: Fallback If The LLM Fails

If Gemini is unavailable, rate-limited, missing an API key, or returns invalid JSON, VibeFinder falls back to heuristic parsing.

The heuristic parser looks for known keywords. For example:

- `upbeat` maps to higher energy.
- `happy` maps to positive valence and/or happy mood.
- `acoustic` maps to higher acousticness.
- `calm` maps to lower energy and chill preference.

This fallback keeps the app usable even when the LLM route fails.

### Step 5: The Parsed Intent Becomes Scoring Preferences

The structured intent is converted into a preference object for ranking.

For the example prompt, the scoring inputs may become:

```text
genre: pop
mood: happy
energy: 0.8
valence: 0.9
likes_acoustic: false
```

If acousticness is not explicitly provided, the system uses `likes_acoustic` to infer an acousticness target:

- `likes_acoustic = true` means target acousticness is about `0.75`.
- `likes_acoustic = false` means target acousticness is about `0.20`.

## 4. Candidate Retrieval

Before final ranking, the app performs a lightweight retrieval step over `data/songs.csv`.

The retrieval step gives a rough pre-score to each song using:

- genre match,
- mood match,
- numeric closeness,
- penalties for constraints like `avoid_intense` or `prefer_chill`.

A simplified retrieval score is:

```text
pre_score = 0.35 * genre_match
          + 0.25 * mood_match
          + 0.40 * average_numeric_closeness
          - constraint_penalties
```

This step does not produce the final recommendation. It only selects likely candidates so the ranking step can focus on the most relevant songs.

## 5. Fields Used By The Ranking Algorithm

Each song in `data/songs.csv` has these fields:

- `genre`
- `mood`
- `energy`
- `tempo_bpm`
- `valence`
- `danceability`
- `acousticness`

The recommender compares these song fields against the parsed user preferences.

## 6. Ranking Weights

Each feature has a weight. A larger weight means that feature has more influence on the final score.

Current default weights:

```text
genre:       0.25
mood:        0.20
energy:      0.20
acousticness:0.15
tempo_bpm:   0.10
danceability:0.07
valence:     0.03
```

This means genre, mood, and energy matter most. Tempo, danceability, and valence still matter, but they fine-tune the ranking more than they dominate it.

In guided mode, user feedback can adjust the recommendation state or weight hints. For example, if the user says "more acoustic," the system can increase the importance of acousticness or update the acousticness target before ranking again.

## 7. Sigma And Numeric Closeness

For numeric fields, VibeFinder does not simply ask whether a value is exactly equal. Instead, it rewards songs that are close to the target.

The numeric score uses a Gaussian-style formula:

$$
s_f = e^{-\frac{(x_f - p_f)^2}{2\sigma_f^2}}
$$

Where:

- $s_f$ is the score for one feature.
- $x_f$ is the song's value.
- $p_f$ is the user's target value.
- $\sigma_f$ controls how strict the comparison is.

Sigma is important because it controls tolerance:

- Smaller sigma means the system is stricter.
- Larger sigma means the system is more forgiving.

Example:

If the target energy is `0.80` and a song has energy `0.82`, the score is very high because the values are close. If another song has energy `0.35`, it receives a much lower energy contribution.

Default sigma values:

```text
energy:       0.12
acousticness: 0.15
tempo_bpm:    18.0
danceability: 0.15
valence:      0.18
```

## 8. Categorical Matching

For categorical fields like `genre` and `mood`, the current system uses exact matching:

```text
match = 1.0
mismatch = 0.0
```

This is simple and explainable, but it is also a limitation. For example, `chill` and `relaxed` may feel similar to a human listener, but the current model treats them as different categories.

## 9. Final Score

Each feature score is multiplied by its weight. Then the weighted scores are added together and normalized.

Conceptually:

$$
R = \frac{\sum_i w_i s_i}{\sum_i w_i}
$$

Where:

- $R$ is the final relevance score.
- $w_i$ is the weight for feature $i$.
- $s_i$ is the feature score.

The final score is used to sort songs from best match to weakest match.

## 10. Example Ranking Explanation

For the prompt:

```text
Looking for poppy upbeat and happy songs
```

The system produced:

```text
1. Sunrise City - Neon Echo
   Final Score: 0.993
   Reasons:
   - Genre match: +0.25 points
   - Mood match: +0.20 points
   - Energy closeness (target 0.80, song 0.82): +0.20 points
   - Acousticness closeness (target 0.20, song 0.18): +0.15 points
   - Valence closeness (target 0.90, song 0.84): +0.03 points
```

This is a strong result because the song matches genre and mood, and its numeric values are close to the parsed targets.

## 11. How Guided Mode Refines Results

Guided mode uses the same pipeline, but repeats it across turns.

```text
Initial prompt
  -> parse intent
  -> retrieve candidates
  -> rank songs
  -> ask if user wants refinement
  -> parse feedback
  -> update state or weights
  -> rerank
```

Example feedback:

```text
more acoustic and less intense
```

The system may respond by:

- increasing acousticness,
- lowering energy,
- setting `avoid_intense = true`,
- changing feature weights,
- or asking a clarifying question if the feedback is too vague.

The key point is that guided mode does not use a separate recommender. It reuses the quick recommendation workflow, but loops through it with updated state.

## 12. Safety And Reliability Guardrails

VibeFinder treats LLM output as untrusted input. The system has safeguards so one bad LLM response does not break the app.

Important guardrails:

- structured JSON expectations,
- validation of numeric bounds,
- allowed category lists for genre and mood,
- fallback to heuristics when the LLM fails,
- local request cooldowns,
- per-session call limits,
- caching for repeated prompts,
- sanitized debug logging.

These guardrails helped during testing. At one point, Gemini returned truncated JSON. Instead of crashing, the app fell back to heuristics, and the issue became easier to debug once diagnostics were added.

## 13. What To Highlight During The Demo

During the walkthrough, the most important points to emphasize are:

1. The LLM understands language, but it does not directly rank songs.
2. The recommender is still transparent and score-based.
3. Diagnostics show the full chain from prompt to JSON to scoring inputs to ranked output.
4. Sigma controls how strict numeric matching is.
5. Weights control which features matter most.
6. Guided mode reuses the same workflow instead of duplicating the system.
7. Fallback behavior makes the app more reliable when the LLM fails.

## 14. Short Demo Script

1. Start the app:

```bash
python -m src.main
```

2. Choose `4) Guided Agent (LLM)`.

3. Enter:

```text
Looking for poppy upbeat and happy songs
```

4. Point out:

- `Source: LLM`
- the JSON fields Gemini returned,
- the parsed scoring inputs,
- the ranking weights,
- and why `Sunrise City - Neon Echo` scored highly.

5. Explain that if the LLM fails, the app switches to heuristic mode instead of crashing.

6. Optionally refine:

```text
more acoustic
```

Then explain how the system updates the recommendation state and reranks songs using the same core recommender.
