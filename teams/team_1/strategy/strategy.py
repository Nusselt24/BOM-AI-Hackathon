def select_strategy(intent, mode_override=None):
    """
    mode_override: "understanding" | "evidence" | "population"
    """

    if mode_override:
        purpose = mode_override
    else:
        purpose = intent["purpose"]

    if purpose == "understanding":
        return {
            "weights": {
                "kanker.nl": 0.6,
                "reports": 0.3,
                "scientific_publications": 0.1
            },
            "depth": "low",
            "uncertainty": "low"
        }

    if purpose == "evidence":
        return {
            "weights": {
                "scientific_publications": 0.5,
                "reports": 0.4,
                "kanker.nl": 0.1
            },
            "depth": "high",
            "uncertainty": "high"
        }

    if purpose == "population":
        return {
            "weights": {
                "reports": 0.6,
                "scientific_publications": 0.3,
                "kanker.nl": 0.1
            },
            "depth": "medium",
            "uncertainty": "medium"
        }
    

if __name__ == "__main__":
    test_intents = [
        {"scope": "population", "purpose": "understanding"},
        {"scope": "individual", "purpose": "evidence"},
        {"scope": "population", "purpose": "evidence"},
        {"scope": "individual", "purpose": "understanding"}
    ]

    for intent in test_intents:
        strategy = select_strategy(intent)
        print(f"Intent: {intent}")
        print(f"Selected Strategy: {strategy}\n")