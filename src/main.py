"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

Phase 3 adds a terminal menu for quick recommendations and a guided agent loop.
"""

try:
    from .agent_interface import run_guided_agent, run_quick_recommendation
    from .recommender import load_songs
except ImportError:
    # Fallback for direct execution: python src/main.py
    from agent_interface import run_guided_agent, run_quick_recommendation
    from recommender import load_songs


def main() -> None:
    songs = load_songs("data/songs.csv")

    while True:
        print("\nVibeFinder")
        print("1) Quick Recommend")
        print("2) Guided Agent")
        print("3) Exit")

        choice = input("Choose an option: ").strip()
        if choice == "1":
            run_quick_recommendation(songs)
        elif choice == "2":
            run_guided_agent(songs)
        elif choice == "3":
            break
        else:
            print("Please choose 1, 2, or 3.")


if __name__ == "__main__":
    main()
