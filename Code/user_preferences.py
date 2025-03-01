class UserPreferenceCollector:
    """Handles user input collection and parsing into structured preferences."""

    def __init__(self, interactive=True):
        """
        Initializes the preference collector.
        
        Args:
            interactive (bool): If True, prompts user for input interactively.
        """
        self.interactive = interactive
        self.questions = [
            "How big do you want your house to be?",
            "What are the 3 most important things for you in choosing this property?",
            "Which amenities would you like?",
            "Which transportation options are important to you?",
            "How urban do you want your neighborhood to be?"
        ]
        
        self.default_answers = [
            "A comfortable three-bedroom house with a spacious kitchen and a cozy living room.",
            "A quiet neighborhood, good local schools, and convenient shopping options.",
            "A backyard for gardening, a two-car garage, and a modern, energy-efficient heating system.",
            "Easy access to a reliable bus line, proximity to a major highway, and bike-friendly roads.",
            "A balance between suburban tranquility and access to urban amenities like restaurants and theaters."
        ]
        
    def collect_preferences(self):
        """Collects buyer preferences interactively or using default responses."""
        if self.interactive:
            print("\n🏡 Home Preference Questionnaire 🏡\n")
            answers = [input(f"{q} ") for q in self.questions]
        else:
            answers = self.default_answers
        
        return self.parse_preferences(answers)
    
    def parse_preferences(self, answers):
        """Converts user responses into a structured dictionary."""
        structured_prefs = {
            "house_size": answers[0],
            "key_factors": answers[1].split(", "),  # Convert comma-separated values to a list
            "amenities": answers[2].split(", "),
            "transportation": answers[3].split(", "),
            "urban_preference": answers[4]
        }
        
        return structured_prefs

# Example Usage
if __name__ == "__main__":
    collector = UserPreferenceCollector(interactive=True)  # Change to False for default answers
    user_prefs = collector.collect_preferences()
    print("\n🎯 Structured Preferences:", user_prefs)