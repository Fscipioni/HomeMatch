import openai
import re

class UserPreferenceCollector:
    """Handles user input collection and parsing into structured preferences."""

    def __init__(self, interactive=True, use_llm=True):
        """
        Initializes the preference collector.
        
        Args:
            interactive (bool): If True, prompts user for input interactively.
            use_llm (bool): If True, uses an LLM to refine user input.
        """
        self.interactive = interactive
        self.use_llm = use_llm
        self.questions = [
            "In which State, or States, would you like to look for a property?",
            "Which city or cities?",
            "How big do you want your house to be?",
            "What are the 3 most important things for you in choosing this property?",
            "Which amenities would you like?",
            "Which transportation options are important to you?",
            "How urban do you want your neighborhood to be?"
        ]
        
        self.default_answers = [
            "Caifornia and Colorado",
            "Mountain View, Cupertino, or Denver",
            "A comfortable three-bedroom house with a spacious kitchen and a cozy living room.",
            "A quiet neighborhood, good local schools, and convenient shopping options.",
            "A backyard for gardening, a two-car garage, and a modern, energy-efficient heating system.",
            "Easy access to a reliable bus line, proximity to a major highway, and bike-friendly roads.",
            "A balance between suburban tranquility and access to urban amenities like restaurants and theaters."
        ]
    
    def call_llm(self, text):
        """
        Uses an LLM to correct typos, improve wording, and remove unnecessary words.
        
        Args:
            text (str): Raw user input to be refined.
        
        Returns:
            str: Improved response.
        """
        prompt = f"""
        Please improve the following user input by:
        1. Fixing typos and improving clarity.
        2. Replace words like 'and', 'or' with commas ',' when separating items.
        3. Keeping it concise and easy to parse.

        User Input: "{text}"
        Improved Output:
        """
        
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "You are an AI that refines user input for structured data."},
                          {"role": "user", "content": prompt}],
                temperature=0
            )
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"❌ LLM Error: {e}")
            return text  # Return original if LLM fails

    def collect_preferences(self):
        """Collects buyer preferences interactively or using default responses."""
        if self.interactive:
            print("\n🏡 Home Preference Questionnaire 🏡\n")
            print("Please answer the following questions to get a personalized home recommendation.")
            print("Your responses will help us find the best listings that match your preferences.\n")
            answers = [input(f"{q} ") for q in self.questions]
        else:
            answers = self.default_answers

        # Process responses with LLM if enabled
        if self.use_llm:
            answers = [self.call_llm(answer) for answer in answers]

        return self.parse_preferences(answers)
    
    def parse_preferences(self, answers):
        """Converts user responses into a structured dictionary."""
        structured_prefs = {
            "state": re.split(r",\s*", answers[0]),  # Splitting by comma & spaces
            "city": re.split(r",\s*", answers[1]),
            "house_size": answers[2],
            "key_factors": re.split(r",\s*", answers[3]),  
            "amenities": re.split(r",\s*", answers[4]),
            "transportation": re.split(r",\s*", answers[5]),
            "urban_preference": answers[6]
        }
        
        return structured_prefs

# Example Usage
if __name__ == "__main__":
    collector = UserPreferenceCollector(interactive=False, use_llm=True)  # Change to False for default answers
    user_prefs = collector.collect_preferences()
    print("\n🎯 Structured Preferences:", user_prefs)