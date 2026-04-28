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
        print("1) Quick Recommend (Heuristic)")
        print("2) Guided Agent (Heuristic)")
        print("3) Quick Recommend (LLM)")
        print("4) Guided Agent (LLM)")
        print("5) Exit")

        choice = input("Choose an option: ").strip()
        if choice == "1":
            run_quick_recommendation(songs, intent_mode="heuristic")
        elif choice == "2":
            run_guided_agent(songs, intent_mode="heuristic")
        elif choice == "3":
            run_quick_recommendation(songs, intent_mode="llm")
        elif choice == "4":
            run_guided_agent(songs, intent_mode="llm")
        elif choice == "5":
            break
        else:
            print("Please choose 1, 2, 3, 4, or 5.")


if __name__ == "__main__":
    main()
