"""
Module: user_preferences
Description: Handles user input collection and converts responses into structured preferences.
"""

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
            "In which state(s) would you like to look for a property?",
            "Which city or cities?",
            "How big do you want your house to be?",
            "What are the three most important factors for you in choosing this property?",
            "Which amenities would you like?",
            "Which transportation options are important to you?",
            "How urban do you want your neighborhood to be?"
        ]
        
        self.default_answers = [
            "California, Colorado",
            "Mountain View, Cupertino, Denver",
            "A comfortable three-bedroom house with a spacious kitchen and a cozy living room.",
            "A quiet neighborhood, good local schools, convenient shopping options.",
            "Backyard for gardening, two-car garage, energy-efficient heating system.",
            "Reliable bus line, proximity to a major highway, bike-friendly roads.",
            "A balance between suburban tranquility and urban amenities like restaurants and theaters."
        ]

    def call_llm(self, text):
        """
        Uses an LLM to refine user input by fixing typos, improving clarity, and ensuring conciseness.

        Args:
            text (str): Raw user input to be refined.

        Returns:
            str: Improved response.
        """
        prompt = f"""
        Please improve the following user input by:
        1. Fixing typos and improving clarity.
        2. Replacing 'and', 'or' with commas ',' when separating items.
        3. Keeping it concise and easy to parse.

        User Input: "{text}"
        Improved Output:
        """

        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an AI that refines user input for structured data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"❌ LLM Error: {e}")
            return text  # Return original input if LLM fails

    def collect_preferences(self):
        """
        Collects buyer preferences interactively or using default responses.

        Returns:
            dict: Structured dictionary of user preferences.
        """
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
        """
        Converts user responses into a structured dictionary.

        Args:
            answers (list): List of processed user responses.

        Returns:
            dict: Dictionary containing structured user preferences.
        """
        try:
            structured_prefs = {
                "state": re.split(r",\s*", answers[0].strip()),  # Splitting by comma & spaces
                "city": re.split(r",\s*", answers[1].strip()),
                "house_size": answers[2].strip(),
                "key_factors": re.split(r",\s*", answers[3].strip()),
                "amenities": re.split(r",\s*", answers[4].strip()),
                "transportation": re.split(r",\s*", answers[5].strip()),
                "urban_preference": answers[6].strip()
            }
            return structured_prefs
        except IndexError as e:
            print(f"❌ Error parsing preferences: {e}")
            return {}

