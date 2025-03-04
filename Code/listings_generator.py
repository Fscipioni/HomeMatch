"""
Module: listings_generator
Description: Generates structured real estate listings using OpenAI's LLM and realistic images using Stable Diffusion.
"""

import json
import time
import re
import uuid
import os
import torch
import random
from openai import OpenAI
from PIL import Image
from diffusers import AutoPipelineForText2Image
from config_loader import load_config_value

class ListingsGenerator:
    """Generates real estate listings using OpenAI's LLM and realistic images using Stable Diffusion."""

    def __init__(self, total_listings=10, batch_size=5, output_file="../Data/listings.json"):
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

    @staticmethod
    def clean_json_output(response_text):
        """
        Cleans OpenAI's JSON output by removing Markdown formatting.

        Args:
            response_text (str): Raw response from OpenAI.

        Returns:
            str: Cleaned JSON string.
        """
        return re.sub(r"```json\n(.*)\n```", r"\1", response_text, flags=re.DOTALL)

    def generate_batch(self):
        """
        Generates a batch of real estate listings using OpenAI's LLM.

        Returns:
            list: A list of generated property listings (dictionaries).
        """
        listings_prompt = f"""
        You are an experienced real estate agent with extensive knowledge of property listings across all 50 U.S. states,
        covering a variety of neighborhoods from luxury estates to budget-friendly homes.

        Generate real estate listings using the following schema:

        - Neighborhood: A real neighborhood in a randomly selected city
        - City: The city where the property is located
        - State: The state where the property is located
        - Price: The property price, ranging from $100,000 to $5,000,000
        - Bedrooms: Number of bedrooms, ranging from 1 to 15
        - Bathrooms: Number of bathrooms, ranging from 1 to 5
        - House Size: Property size, ranging from 500 sqft to 50,000 sqft
        - Description: A 40-word description of the house.
        - Neighborhood Description: A brief description of the neighborhood.

        Higher-priced properties should have more bedrooms, bathrooms, and larger sizes.

        📌 Provide the response **strictly** as a JSON object with the key "listings", 
        where "listings" is a list of dictionary objects.
        """

        client = OpenAI(
            base_url="https://openai.vocareum.com/v1",
            api_key=load_config_value("VOCAREUM_OPENAI_API_KEY")
        )

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an experienced real estate agent."},
                    {"role": "user", "content": listings_prompt}
                ],
                temperature=0.8
            )

            raw_output = response.choices[0].message.content
            clean_output = self.clean_json_output(raw_output)  # Remove Markdown JSON wrapping
            
            # print("🔍 Raw OpenAI Output:\n", clean_output)  # Debugging

            # Try parsing JSON safely
            try:
                listings_json = json.loads(clean_output)
                # print("✅ Parsed JSON Structure:", listings_json)  # Debugging
                
                if "listings" not in listings_json or not isinstance(listings_json["listings"], list):
                    raise ValueError("❌ 'listings' key missing or not a list in OpenAI response.")

                listings = listings_json["listings"]

            except json.JSONDecodeError as e:
                print(f"❌ JSON Decoding Error: {e}")
                return []
            except Exception as e:
                print(f"❌ Unexpected Error in JSON parsing: {e}")
                return []

            # 🔹 Assign a unique ID and generate an image for each listing
            for listing in listings:
                listing["id"] = str(uuid.uuid4())
                listing["image_path"] = self.generate_image(listing)  # Generate & store image path

            return listings

        except Exception as e:
            print(f"❌ Error generating batch: {e}")
            return []
    
    def generate_listings(self):
        """
        Generates all real estate listings in batches and saves them to a file.
        """
        all_listings = []

        for _ in range(self.total_listings // self.batch_size):
            batch = self.generate_batch()
            if batch:
                all_listings.extend(batch)

                # Save listings to file
                try:
                    with open(self.output_file, "w") as f:
                        json.dump(all_listings, f, indent=4)
                except IOError as e:
                    print(f"❌ Error saving listings to file: {e}")

            time.sleep(1)  # Prevent rate limit issues

        print(f"✅ Successfully generated {len(all_listings)} listings and saved to {self.output_file}")

    def generate_image(self, listing):
        """
        Generates an image for a given real estate listing using Stable Diffusion.

        Args:
            listing (dict): Dictionary containing details about the house.

        Returns:
            str: Path to the saved generated image, or None if an error occurs.
        """
        try:
            # Select device: prioritize CUDA if available, otherwise use MPS (Mac) or CPU
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

            # Load the Stable Diffusion pipeline
            pipe = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/sdxl-turbo",
                torch_dtype=torch.float16,
                variant="fp16"
            ).to(device)

            # Construct a safe prompt from the listing details
            prompt = (
                f"A {random.choice(['modern', 'classic', 'Victorian', 'Mediterranean', 'ranch-style'])} house in "
                f"{listing.get('Neighborhood', 'a neighborhood')}, {listing.get('City', 'a city')}, {listing.get('State', 'a state')}. "
                # f"It has {listing.get('Bedrooms', 'an unknown number of')} bedrooms, "
                # f"{listing.get('Bathrooms', 'an unknown number of')} bathrooms, and is "
                f"{listing.get('House Size', 'unknown')} sqft. "
                f"The house features a {random.choice(['red brick', 'white stucco', 'blue wooden', 'gray stone', 'tan adobe'])} exterior, "
                f"a {random.choice(['spacious front yard', 'lush garden', 'swimming pool', 'rooftop terrace', 'wraparound porch'])}, and "
                f"{listing.get('Description', 'a beautiful architectural design')}."
                )

            # Use a unique random seed for each image.
            # This ensures each house image is unique but still reproducible if needed.
            random_seed = random.randint(1, 1_000_000)  
            torch.manual_seed(random_seed)  # Different seed for each listing

            # Generate the image
            print(f"🖼️ Generating image for listing in {listing.get('City', 'Unknown City')}...")
            image = pipe(
                prompt=prompt,
                num_inference_steps=1,
                guidance_scale=1.0,
                negative_prompt=["overexposed", "underexposed", "low quality", "unrealistic", "artifacts", "distortion"]
            ).images[0]

            # Ensure output directory exists
            image_dir = "../Data/Images"
            os.makedirs(image_dir, exist_ok=True)

            # Save the image with the listing's unique ID
            image_path = os.path.join(image_dir, f"{listing.get('id', 'unknown')}.png")
            image.save(image_path)

            print(f"✅ Image saved: {image_path}")
            return image_path

        except Exception as e:
            print(f"❌ Error generating image: {e}")
            return None

