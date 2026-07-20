# Research Prompt Log 

## 1. Streaming recommender systems

**Prompt**

Research and summarize how major streaming platforms like Spotify or YouTube predict what users will love next.

What is the difference between collaborative filtering and content-based filtering?

How are these used by these major platforms respectively?

**Follow-up prompt**

Can you give a shorter explanation focused on the practical difference between the two approaches?

**Answer used**

Collaborative filtering learns from behavior patterns across users. If many people who liked one item also liked another, the system uses that pattern to recommend the second item. Content-based filtering learns from item attributes. If a user liked a song with certain features, it recommends songs with similar features.

**Takeaway**

Streaming systems are usually hybrid. Spotify uses strong content-based signals such as audio features and seed tracks, but also benefits from behavioral signals like plays and skips. YouTube leans heavily on collaborative and behavior-driven ranking, especially watch and click history.

---

## 2. Feature selection for a content-based recommender

**Prompt**

Analyze the available data and suggest which features would be most effective for a simple content-based recommender.

**Follow-up prompt**

What does valence entail in this context and how relevant is it?

What is acousticness?

What is energy tied to?

**Answer used**

The strongest features in the catalog are genre, mood, energy, acousticness, tempo, and danceability. Valence is useful for fine-tuning mood, but it is less decisive than energy or acousticness. Acousticness describes how acoustic or organic a track sounds. Energy reflects perceived intensity and activity.

**Takeaway**

For a small dataset, the best content-based features are the ones that clearly separate song clusters: genre and mood for broad categories, then energy, acousticness, and tempo for closer matching.

---

## 3. Scoring and ranking logic

**Prompt**

Let's create an algorithmic blueprint for the specific variables you mentioned and assign importance values.

Specifically, let's make a math-based scoring rule for the recommender. How can I calculate a score for a numerical feature like energy that rewards songs that are closer to the user's preference rather than just having higher or lower values?

**Follow-up prompt**

Is this a scoring rule or a ranking rule? Would I need both? What does either entail?

I am trying to conceptualize the scoring you mentioned. You mentioned the weights which make sense because it's the backbone of the system and how it considers how much importance to give to a field.

But when scoring, if a song is relevant, how does that come into play exactly, is there data that the user has regarding their preference initially, and that is used against the song to calculate the differences which are summed? What is the exact process?

**Answer used**

Numerical features can be scored by closeness to a target value using a Gaussian-style rule. A weighted sum then combines all feature scores into one overall relevance score. The user profile stores preference values, and each song is compared against those preferences field by field.

**Takeaway**

The recommender has two stages: scoring calculates how well a song matches the user profile, and ranking sorts all songs by that score.

---

## 4. README explanation

**Prompt**

I want you to fill in the readme with this information:

My system will use a weighted system considering all the available fields. The higher fields such as genre mood and energy will be given greater importance and subsequent ones will be given lower importance based on my relevance preference.

[...]

I want you to include this information in the read me under the appropriate headings. Use equations where needed.

**Answer used**

The README should explain the song features, the user profile fields, the weight ordering, the Gaussian scoring rule, and the overall relevance equation. It should also describe the recommendation flow from feature comparison to ranking.

**Takeaway**

Use plain language for the explanation, but include the math in the system design section so the scoring logic is explicit and easy to justify.

---

## 5. Phase 2 data flow and implementation planning

**Prompt**

Create a quick mental or written map of the data flow: Input (User Prefs) -> Process (loop through songs and score each one) -> Output (Top-K ranking).

Store this as first documentation in Phase2.md.

**Follow-up prompt**

Based on my outline and next steps, outline the methods and variables needed in `UserProfile` to store user taste profile information.

**Answer used**

The data flow was documented as Input -> Process -> Output, and the `UserProfile` was expanded with optional preference fields, feature weights, feature sigmas, validation, and helper methods.

**Takeaway**

Phase 2 started with a clear architecture map, then translated that design into concrete profile fields and methods.

---

## 6. Scoring + ranking implementation prompts

**Prompt**

Implement `load_songs` to organize relevant fields from the CSV. Is dictionary output ideal?

**Follow-up prompt**

Implement `score_song` based on the algorithm recipe, then implement `recommend_songs` to sort by score and return top K in the most Pythonic way.

**Answer used**

`load_songs` was implemented with typed parsing into dictionaries. `score_song` was implemented with weighted categorical + Gaussian numeric scoring and normalized final score. `recommend_songs` now scores all songs, sorts descending, and returns top K.

**Takeaway**

The core simulation loop was fully implemented and aligned to the design formula from Phase 1 and README.

---

## 7. Explainability and output formatting prompts

**Prompt**

Add a "reasons" component so users understand why each score is what it is, using plain language like "Genre match +0.15 points".

**Follow-up prompt**

Format recommendation output in a clean terminal layout with song title, final score, and specific reasons.

**Answer used**

Per-feature reason lines were added and terminal output was reformatted into a readable list with rank, score, and bullet-point reasons.

**Takeaway**

The model became much easier to interpret because each recommendation now shows transparent feature-level contributions.

---

## 8. Debugging weight inconsistency prompts

**Prompt**

I changed feature weights (for example genre down, energy up) but output seems inconsistent. Why?

**Follow-up prompt**

Fix the inconsistency so scoring defaults and profile defaults use the same source of truth.

**Answer used**

The issue came from separate default weight definitions in different parts of the file. The fix was to use shared module-level constants (`DEFAULT_FEATURE_WEIGHTS`, `DEFAULT_FEATURE_SIGMAS`) referenced by both `UserProfile` and `score_song`.

**Takeaway**

Centralized defaults removed confusing behavior and made score explanations consistent with active scoring logic.

---

## 9. Evaluation/model card refinement prompts

**Prompt**

Fill the model card evaluation section using the actual Phase 2 tests and changes, and make comparisons explicit.

**Follow-up prompt**

Keep original prompt text in section headers, and rewrite explanations in plain language for non-programmers.

**Answer used**

The evaluation section now includes A/B profile comparisons, concrete output changes, why those changes make sense, bug-fix impact, and plain-language interpretation.

**Takeaway**

Phase 2 documentation now reflects real evidence from implementation and testing, not just general statements.
