import csv
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


DEFAULT_FEATURE_WEIGHTS: Dict[str, float] = {
    "genre": 0.25,
    "mood": 0.20,
    "energy": 0.20,
    "acousticness": 0.15,
    "tempo_bpm": 0.10,
    "danceability": 0.07,
    "valence": 0.03,
}

DEFAULT_FEATURE_SIGMAS: Dict[str, float] = {
    "energy": 0.12,
    "acousticness": 0.15,
    "tempo_bpm": 18.0,
    "danceability": 0.15,
    "valence": 0.18,
}


@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """

    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float


@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """

    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

    # Optional fine-tuning preferences (Phase 2).
    target_acousticness: Optional[float] = None
    target_tempo_bpm: Optional[float] = None
    target_danceability: Optional[float] = None
    target_valence: Optional[float] = None

    # Weights control relative importance in overall scoring.
    feature_weights: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_FEATURE_WEIGHTS)
    )

    # Sigma values define tolerance for Gaussian closeness scoring.
    feature_sigmas: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_FEATURE_SIGMAS)
    )

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate profile ranges and keep configuration sane."""
        bounded = [
            ("target_energy", self.target_energy),
            ("target_acousticness", self.target_acousticness),
            ("target_danceability", self.target_danceability),
            ("target_valence", self.target_valence),
        ]
        for name, value in bounded:
            if value is None:
                continue
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")

        if self.target_tempo_bpm is not None and self.target_tempo_bpm <= 0:
            raise ValueError("target_tempo_bpm must be > 0")

        if not self.favorite_genre.strip():
            raise ValueError("favorite_genre cannot be empty")
        if not self.favorite_mood.strip():
            raise ValueError("favorite_mood cannot be empty")

    def get_target_acousticness(self) -> float:
        """Return explicit acousticness target or infer from likes_acoustic."""
        if self.target_acousticness is not None:
            return self.target_acousticness
        return 0.75 if self.likes_acoustic else 0.20

    def get_numeric_targets(self) -> Dict[str, float]:
        """Return numeric targets used by the scoring layer."""
        targets: Dict[str, float] = {
            "energy": self.target_energy,
            "acousticness": self.get_target_acousticness(),
        }
        if self.target_tempo_bpm is not None:
            targets["tempo_bpm"] = self.target_tempo_bpm
        if self.target_danceability is not None:
            targets["danceability"] = self.target_danceability
        if self.target_valence is not None:
            targets["valence"] = self.target_valence
        return targets

    def get_categorical_targets(self) -> Dict[str, str]:
        """Return categorical preferences used by the scoring layer."""
        return {
            "genre": self.favorite_genre,
            "mood": self.favorite_mood,
        }

    def set_optional_preferences(
        self,
        tempo_bpm: Optional[float] = None,
        danceability: Optional[float] = None,
        valence: Optional[float] = None,
        acousticness: Optional[float] = None,
    ) -> None:
        """Update optional targets and re-validate."""
        self.target_tempo_bpm = tempo_bpm
        self.target_danceability = danceability
        self.target_valence = valence
        self.target_acousticness = acousticness
        self.validate()

    def set_feature_weight(self, feature: str, weight: float) -> None:
        """Update one feature weight and re-normalize all weights."""
        if weight < 0:
            raise ValueError("weight must be non-negative")
        self.feature_weights[feature] = weight
        total = sum(self.feature_weights.values())
        if total <= 0:
            raise ValueError("sum of feature weights must be > 0")
        self.feature_weights = {
            key: value / total for key, value in self.feature_weights.items()
        }

    def to_dict(self) -> Dict:
        """Serialize profile so it can be saved and reloaded later."""
        return {
            "favorite_genre": self.favorite_genre,
            "favorite_mood": self.favorite_mood,
            "target_energy": self.target_energy,
            "likes_acoustic": self.likes_acoustic,
            "target_acousticness": self.target_acousticness,
            "target_tempo_bpm": self.target_tempo_bpm,
            "target_danceability": self.target_danceability,
            "target_valence": self.target_valence,
            "feature_weights": dict(self.feature_weights),
            "feature_sigmas": dict(self.feature_sigmas),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "UserProfile":
        """Hydrate a profile from persisted data."""
        profile = cls(
            favorite_genre=data["favorite_genre"],
            favorite_mood=data["favorite_mood"],
            target_energy=float(data["target_energy"]),
            likes_acoustic=bool(data["likes_acoustic"]),
            target_acousticness=data.get("target_acousticness"),
            target_tempo_bpm=data.get("target_tempo_bpm"),
            target_danceability=data.get("target_danceability"),
            target_valence=data.get("target_valence"),
        )
        if "feature_weights" in data and isinstance(data["feature_weights"], dict):
            profile.feature_weights.update(data["feature_weights"])
            profile.set_feature_weight(
                "genre", profile.feature_weights.get("genre", 0.25)
            )
        if "feature_sigmas" in data and isinstance(data["feature_sigmas"], dict):
            profile.feature_sigmas.update(data["feature_sigmas"])
        profile.validate()
        return profile


class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """

    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        if k <= 0:
            return []

        user_prefs = _user_profile_to_prefs(user)
        ranked_songs = sorted(
            (
                (song, score_song(user_prefs, _song_to_dict(song))[0])
                for song in self.songs
            ),
            key=lambda item: (-item[1], item[0].id),
        )
        return [song for song, _ in ranked_songs[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        user_prefs = _user_profile_to_prefs(user)
        return score_song(user_prefs, _song_to_dict(song))[1]


def _song_to_dict(song: Song) -> Dict[str, object]:
    return {
        "id": song.id,
        "title": song.title,
        "artist": song.artist,
        "genre": song.genre,
        "mood": song.mood,
        "energy": song.energy,
        "tempo_bpm": song.tempo_bpm,
        "valence": song.valence,
        "danceability": song.danceability,
        "acousticness": song.acousticness,
    }


def _user_profile_to_prefs(user: UserProfile) -> Dict[str, object]:
    prefs: Dict[str, object] = {
        "genre": user.favorite_genre,
        "mood": user.favorite_mood,
        "energy": user.target_energy,
        "likes_acoustic": user.likes_acoustic,
        "weights": dict(user.feature_weights),
        "sigmas": dict(user.feature_sigmas),
    }
    if user.target_acousticness is not None:
        prefs["acousticness"] = user.target_acousticness
    if user.target_tempo_bpm is not None:
        prefs["tempo_bpm"] = user.target_tempo_bpm
    if user.target_danceability is not None:
        prefs["danceability"] = user.target_danceability
    if user.target_valence is not None:
        prefs["valence"] = user.target_valence
    return prefs


def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    songs: List[Dict] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            songs.append(
                {
                    "id": int(row["id"]),
                    "title": row["title"].strip(),
                    "artist": row["artist"].strip(),
                    "genre": row["genre"].strip(),
                    "mood": row["mood"].strip(),
                    "energy": float(row["energy"]),
                    "tempo_bpm": float(row["tempo_bpm"]),
                    "valence": float(row["valence"]),
                    "danceability": float(row["danceability"]),
                    "acousticness": float(row["acousticness"]),
                }
            )
    return songs


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, str]:
    """
    Compute a relevance score in [0, 1] for one song using a weighted recipe.

    Categorical fields use exact-match scoring, while numeric fields use
    Gaussian closeness scoring.
    """
    weights = dict(DEFAULT_FEATURE_WEIGHTS)
    if isinstance(user_prefs.get("weights"), dict):
        for feature, value in user_prefs["weights"].items():
            if feature in weights and isinstance(value, (int, float)) and value >= 0:
                weights[feature] = float(value)

    sigmas = dict(DEFAULT_FEATURE_SIGMAS)
    if isinstance(user_prefs.get("sigmas"), dict):
        for feature, value in user_prefs["sigmas"].items():
            if feature in sigmas and isinstance(value, (int, float)) and value > 0:
                sigmas[feature] = float(value)

    numeric_targets: Dict[str, float] = {}
    if "energy" in user_prefs:
        numeric_targets["energy"] = float(user_prefs["energy"])
    if "acousticness" in user_prefs:
        numeric_targets["acousticness"] = float(user_prefs["acousticness"])
    elif "likes_acoustic" in user_prefs:
        numeric_targets["acousticness"] = 0.75 if user_prefs["likes_acoustic"] else 0.20
    if "tempo_bpm" in user_prefs:
        numeric_targets["tempo_bpm"] = float(user_prefs["tempo_bpm"])
    if "danceability" in user_prefs:
        numeric_targets["danceability"] = float(user_prefs["danceability"])
    if "valence" in user_prefs:
        numeric_targets["valence"] = float(user_prefs["valence"])

    weighted_sum = 0.0
    active_weight_sum = 0.0
    reason_lines: List[str] = []

    for feature in ("genre", "mood"):
        if feature not in user_prefs:
            continue
        feature_weight = weights[feature]
        song_value = str(song.get(feature, "")).strip().lower()
        user_value = str(user_prefs[feature]).strip().lower()
        feature_score = 1.0 if song_value == user_value else 0.0
        contribution = feature_weight * feature_score
        weighted_sum += contribution
        active_weight_sum += feature_weight
        if feature_score == 1.0:
            reason_lines.append(f"{feature.title()} match: +{contribution:.2f} points")
        else:
            reason_lines.append(
                f"{feature.title()} mismatch: +{contribution:.2f} points"
            )

    for feature, target in numeric_targets.items():
        if feature not in song:
            continue
        feature_weight = weights[feature]
        sigma = sigmas[feature]
        song_value = float(song[feature])
        diff = song_value - target
        feature_score = math.exp(-((diff**2) / (2 * (sigma**2))))
        contribution = feature_weight * feature_score
        weighted_sum += contribution
        active_weight_sum += feature_weight
        reason_lines.append(
            (
                f"{feature.replace('_', ' ').title()} closeness "
                f"(target {target:.2f}, song {song_value:.2f}): +{contribution:.2f} points"
            )
        )

    if active_weight_sum <= 0:
        return 0.0, "No usable preferences were provided."

    score = weighted_sum / active_weight_sum
    explanation = " | ".join(reason_lines)
    return score, explanation


def recommend_songs(
    user_prefs: Dict, songs: List[Dict], k: int = 5
) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py
    """
    if k <= 0:
        return []

    scored: List[Tuple[Dict, float, str]] = [
        (song, *score_song(user_prefs, song)) for song in songs
    ]
    return sorted(scored, key=lambda item: item[1], reverse=True)[:k]
