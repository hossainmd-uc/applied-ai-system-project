# Model Card: VibeFinder

## 1. Model Name

**VibeFinder: AI-Assisted Music Recommender**

VibeFinder builds on the original **Music Recommender Simulation** from Modules 1-3. The original project recommended songs from a small CSV catalog using a weighted content-based scoring system. The current version keeps that recommender as the core and adds an LLM-assisted terminal interface for natural language music requests.

---

## 2. Intended Use

VibeFinder is intended for classroom exploration of recommender systems and AI-assisted interfaces. It suggests songs from `data/songs.csv` based on a user's natural language request, such as:

- `Looking for poppy upbeat and happy songs`
- `I want a spacey calm vibey song`
- `calm acoustic music for studying`

The system supports two interaction styles and two parsing modes:

- `Quick Recommend`: one prompt, one ranked recommendation list.
- `Guided Agent`: a more interactive mode where the user can clarify or refine what they want.
- `Heuristic`: forces the local keyword parser.
- `LLM`: uses Gemini first, with fallback behavior if the model call fails.

This project is not intended for real production music recommendation. It uses a small hand-built dataset and a simple scoring model, so it should be treated as an educational prototype.

---

## 3. How The System Works

VibeFinder has two main parts:

1. **Understanding layer**
   - Uses Gemini when an API key is available.
   - Falls back to heuristic keyword parsing when the LLM is unavailable, rate-limited, or returns invalid output.
   - Converts natural language into structured intent fields like `genre`, `mood`, `energy`, `acousticness`, `tempo_bpm`, `danceability`, and `valence`.

2. **Recommendation layer**
   - Retrieves likely candidates from `data/songs.csv`.
   - Scores the candidates using the existing weighted recommender.
   - Returns ranked songs with explanation text showing why each song scored the way it did.

The LLM does **not** directly choose the songs. Its job is to translate language into structured scoring inputs. The final ranking is still handled by the transparent recommender in `src/recommender.py`.

The system also prints diagnostics during terminal use:

- whether the result came from `LLM`, `HEURISTIC`, or cache,
- the raw LLM JSON output when available,
- the parsed scoring inputs,
- the ranking criteria weights,
- and the reasons behind each recommendation.

---

## 4. Data

The catalog contains 20 songs in `data/songs.csv`.

Each song has these fields:

- `genre`
- `mood`
- `energy`
- `tempo_bpm`
- `valence`
- `danceability`
- `acousticness`

The catalog includes genres such as pop, lofi, rock, ambient, jazz, synthwave, hip hop, metal, house, folk, reggae, blues, cinematic, and soul.

The dataset is intentionally small, which makes it easier to inspect and explain. However, it also means the recommender cannot represent the full range of real-world music taste.

---

## 5. Strengths

VibeFinder works best when the user gives a request that can be mapped to the dataset's available features. For example, prompts involving genre, mood, energy, or valence usually work well.

The main strengths are:

- The recommendations are explainable.
- The LLM output is structured and visible.
- The system can still work without the LLM by using heuristic fallback.
- The guided mode reuses the same ranking pipeline instead of creating a separate recommender.
- The user can see the chain from prompt, to parsed intent, to scoring weights, to final recommendations.

One successful test was:

```text
Looking for poppy upbeat and happy songs
```

The LLM mapped this to `genre = pop`, `mood = happy`, `energy = 0.8`, and `valence = 0.9`. The top recommendation was `Sunrise City - Neon Echo`, which made sense because it matched the requested genre and mood and was close to the target energy and valence.

---

## 6. Limitations And Bias

VibeFinder is limited by the simplicity of its ranking system. It uses fixed feature weights and Gaussian closeness for numeric fields, so recommendation quality depends heavily on the chosen weights and sigma thresholds. If a sigma value is too strict, songs that are close but not exact may be punished too heavily. If it is too loose, songs that are only loosely related may score too highly.

Categorical scoring is also strict. For example, `chill` and `relaxed` are similar moods, but the current model treats them as different categories unless they exactly match. This can create unfair or unintuitive rankings for prompts that use related but non-identical words.

The dataset also creates bias. Since the catalog is small, genres or moods that appear more often have more chances to show up in the top results. The system cannot recommend songs outside the CSV file, even if they would be a better fit for the user's request.

The LLM layer adds another limitation. Gemini can misunderstand vague prompts, return malformed output, hit rate limits, or fail because of API quota. The system has guardrails and fallback behavior, but those do not make the LLM perfect. Guided mode is also partly limited by hardcoded clarification categories, so it cannot ask every possible follow-up question a human music expert might ask.

Finally, the system mainly optimizes for similarity to the user's stated request. It does not deeply model novelty, long-term taste, social context, listening history, or diversity across artists.

---

## 7. Misuse And Risk Prevention

The main misuse risk is API abuse or irrelevant input. A user could enter repeated prompts, very long prompts, or prompts unrelated to music. I cannot fully control user input, so the app assumes that input may be noisy or irrelevant.

To reduce risk, I added guardrails:

- per-session LLM call limits,
- cooldowns between LLM calls,
- query-level intent caching,
- provider backoff after `429` rate-limit errors,
- fallback to heuristic parsing when the LLM fails,
- sanitized debug logs that do not expose the API key.

Since this is a classroom recommender, the risk is not high-stakes harm. The bigger risks are misleading recommendations, over-trusting the LLM, exposing secrets, or accidentally overusing a limited API key.

---

## 8. Evaluation

I evaluated VibeFinder with unit tests and manual terminal tests.

Automated tests cover:

- core recommendation scoring,
- Phase 3 intent parsing,
- candidate retrieval,
- fallback behavior,
- guided-agent state changes,
- LLM guardrail behavior,
- and diagnostic output.

The full test suite currently passes:

```text
13 passed
```

Manual tests included:

1. **Successful LLM path**
   - Prompt: `Looking for poppy upbeat and happy songs`
   - Result: Gemini produced valid JSON, the app printed `Source: LLM`, and the top recommendation was `Sunrise City - Neon Echo`.

2. **Malformed LLM output**
   - Prompt: `I want a spacey calm vibey song`
   - Result: Gemini returned malformed/truncated JSON during testing, so the app fell back to heuristic parsing and still returned recommendations.

3. **No API key fallback**
   - Result: the app printed a note that Gemini was unavailable and used heuristic parsing instead.

What surprised me most was that the recommender could be working correctly while the overall system still looked confusing. Before diagnostics were added, I could not easily tell whether an answer came from the LLM or heuristic fallback. Once the app printed the source, JSON, parsed scoring inputs, weights, and reasons, debugging became much clearer.

---

## 9. Ethical Considerations

This project shows why explainability matters in AI systems. If the app only returned a ranked list, it would be hard to know whether the model understood the user or just got lucky. Showing the intermediate steps makes the system more transparent.

The project also shows that LLM output should be treated as untrusted input. The LLM can return unexpected, malformed, or incomplete output. Because of that, the system validates structured fields, uses fallback behavior, and avoids relying on the LLM as the final decision-maker.

The system should not be used to make important decisions about users. It is a learning project for understanding recommendation logic, LLM integration, and debugging.

---

## 10. Future Work

Future improvements could include:

- adding graded similarity for related moods such as `chill` and `relaxed`,
- improving heuristic parsing for vague words like `spacey`, `vibey`, and `dreamy`,
- expanding the song catalog,
- adding artist diversity rules,
- adding a user option to hide or show detailed diagnostics,
- supporting exploration mode versus safe recommendation mode,
- and saving manual test results in a more formal evaluation log.

---

## 11. Personal Reflection

This project taught me that adding AI to an existing system is not just about calling an LLM. The harder part is deciding what job the LLM should do. In VibeFinder, the LLM is useful for translating natural language into structured preferences, but the transparent recommender still handles final ranking.

I also learned that structure is one of the most important parts of reliability. At first, LLM output was too open-ended, which made it hard to map into the existing scoring functions. Once I added JSON schemas, validation, fallback behavior, and diagnostics, the system became much easier to understand and improve.

The biggest lesson was that AI systems need visible intermediate steps. Seeing the raw JSON, parsed fields, scoring weights, and recommendation reasons helped me debug problems that would otherwise have seemed mysterious. It also helped me understand that guardrails and explanations are just as important as the model itself.
