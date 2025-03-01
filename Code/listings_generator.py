import json
import openai
import time
import re

class ListingsGenerator:
    """Handles the generation of real estate listings using OpenAI's API."""
    
    def __init__(self, total_listings=100, batch_size=20, output_file="../Data/listings.json"):
        """
        Initializes the listing generator with OpenAI API settings.
        
        Args:
            total_listings (int): Total number of listings to generate.
            batch_size (int): Number of listings per API request.
            output_file (str): Path to store generated listings.
        """
        self.total_listings = total_listings
        self.batch_size = batch_size
        self.output_file = output_file
        
        # OpenAI API setup
        openai.api_base = "https://openai.vocareum.com/v1"
        openai.api_key = "voc-615907097126677342454766bbd54dcda1a5.27571968"
    
    @staticmethod
    def clean_json_output(response_text):
        """Removes Markdown JSON formatting from OpenAI output."""
        return re.sub(r"```json\n(.*)\n```", r"\1", response_text, flags=re.DOTALL)
    
    def generate_batch(self):
        """Generates a batch of real estate listings using OpenAI."""
        listings_prompt = """
        You are an experienced real estate agent with extensive knowledge of property listings across all 50 states in the U.S.,
        spanning diverse neighborhoods from luxury estates to budget-friendly homes.

        Generate real estate listings following this schema:

        Neighborhood: A real neighborhood in a randomly selected city
        City: The city where the property is located
        State: The state where the property is located
        Price: The property price, ranging from $100,000 to $5,000,000
        Bedrooms: Number of bedrooms, ranging from 1 to 15
        Bathrooms: Number of bathrooms, ranging from 1 to 5
        House Size: Property size, ranging from 500 sqft to 50,000 sqft

        Description: A 40-word description of the house.
        Neighborhood Description: A brief description of the neighborhood.

        The more expensive the property, the higher the number of bedrooms and bathrooms, the larger the size, 
        and the more detailed the descriptions of the property and neighborhood.

        Return exactly {} listings in a structured JSON format.
        """.format(self.batch_size)
        
        try:
            response = openai.chat.completions.create( #openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an experienced real estate agent."},
                    {"role": "user", "content": listings_prompt}
                ],
                temperature=0.7
            )
            raw_output = response["choices"][0]["message"]["content"]
            clean_output = self.clean_json_output(raw_output)  # Remove ` ```json ` wrapping
            return json.loads(clean_output)["listings"]  # Extract listings from JSON
        except Exception as e:
            print(f"❌ Error generating batch: {e}")
            return []

    def generate_listings(self):
        """Generates all listings in batches and saves them to a file."""
        all_listings = []
        
        for _ in range(self.total_listings // self.batch_size):
            batch = self.generate_batch()
            if batch:
                all_listings.extend(batch)
                with open(self.output_file, "w") as f:
                    json.dump(all_listings, f, indent=4)
            time.sleep(1)  # Avoid rate limits
        
        print(f"✅ Successfully generated {len(all_listings)} listings and saved to {self.output_file}")

# Usage Example
if __name__ == "__main__":
    generator = ListingsGenerator(total_listings=100, batch_size=20)
    generator.generate_listings()