import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from config_loader import load_config_value

class LlmAugmentation:
    """Handles LLM-based augmentation of real estate listings."""

    def __init__(self):
        """Initializes the LLM model using LangChain."""
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            openai_api_key=load_config_value("OPENAI_API_KEY")
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
            user_prompt (str): The user's home preference details.
            listings (list): A list of retrieved real estate listings (dictionaries).

        Returns:
            list: A list of augmented listing descriptions.
        """
        try:
            # Format listings into readable JSON
            listings_text = json.dumps(listings, indent=2)

            # Generate augmented descriptions using LangChain's LLM
            response = self.llm.invoke(
                self.llm_prompt.format(n_answers=len(listings), listings=listings_text)
            )

            # Extract LLM response
            return response.content.strip()
        
        except Exception as e:
            print(f"❌ Error generating augmented descriptions: {e}")
            return []

# Example Usage
if __name__ == "__main__":
    # Initialize the augmentation class
    llm_augmentor = LlmAugmentation()

    # Example user preferences (can be passed from `UserPreferenceCollector`)
    user_prompt = "Looking for a spacious suburban home with a large backyard, modern kitchen, and easy access to public transit."

    # Example listings (retrieved from the vector database)
    listings = [
        {
            "Neighborhood": "Sunnyvale",
            "City": "San Francisco",
            "State": "CA",
            "Price": 1500000,
            "Bedrooms": 3,
            "Bathrooms": 2,
            "House Size": "1800 sqft",
            "Description": "A beautiful 3-bedroom home with an open-plan kitchen.",
            "Neighborhood Description": "Family-friendly neighborhood with great parks."
        },
        {
            "Neighborhood": "Downtown Denver",
            "City": "Denver",
            "State": "CO",
            "Price": 900000,
            "Bedrooms": 2,
            "Bathrooms": 2,
            "House Size": "1300 sqft",
            "Description": "Modern condo with floor-to-ceiling windows and city views.",
            "Neighborhood Description": "Vibrant area with excellent dining and nightlife."
        }
    ]

    # Generate personalized descriptions
    augmented_listings = llm_augmentor.generate_augmented_descriptions(listings)
    
    # Display results
    print("\n📌 Augmented Listings Descriptions:\n", augmented_listings)