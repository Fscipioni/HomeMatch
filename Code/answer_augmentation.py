"""
Module: answer_augmentation
Description: Uses an LLM to enhance real estate listing descriptions for better personalization.
"""

import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from config_loader import load_config_value  # Ensure this is correctly implemented

class LlmAugmentation:
    """Handles LLM-based augmentation of real estate listings."""

    def __init__(self):
        """
        Initializes the LLM model using LangChain.
        """
        # Load API credentials securely (avoid hardcoding API keys)
        os.environ["OPENAI_API_KEY"] = load_config_value("VOCAREUM_OPENAI_API_KEY")
        os.environ["OPENAI_API_BASE"] = "https://openai.vocareum.com/v1"

        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7
        )

        # Define the LLM prompt template
        self.llm_prompt = PromptTemplate.from_template(
            """
            Your role is to enhance real estate listing descriptions.
            You will receive {n_answers} real estate listings retrieved using similarity search. 
            Each listing matches the buyer’s preferences in terms of location, property features, amenities, and neighborhood.

            Your task:
            - **Enhance the property descriptions**: Subtly emphasize features that align with the buyer's specific preferences.
            - **Maintain factual integrity**: Do not add fictional information. Keep the facts unchanged while improving appeal.

            Here are the listings:
            {listings}

            Now, rewrite the descriptions to highlight the aspects most relevant to the buyer.
            """
        )

    def generate_augmented_descriptions(self, listings):
        """
        Uses the LLM to augment property descriptions for better personalization.

        Args:
            listings (list): A list of retrieved real estate listings (dictionaries).

        Returns:
            list: A list of augmented listing descriptions.
        """
        try:
            if not listings:
                raise ValueError("Listings data is empty. Cannot generate augmented descriptions.")

            # Format listings into readable JSON
            listings_text = json.dumps(listings, indent=2)

            # Generate augmented descriptions using LangChain's LLM
            response = self.llm.invoke(
                self.llm_prompt.format(n_answers=len(listings), listings=listings_text)
            )

            # Ensure response is valid
            if not response or not hasattr(response, "content"):
                raise ValueError("Received invalid response from LLM.")

            return response.content.strip()

        except Exception as e:
            print(f"❌ Error generating augmented descriptions: {e}")
            return []

