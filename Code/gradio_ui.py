import gradio as gr
import openai
from answer_augmentation import LlmAugmentation

def create_gradio_interface(vector_db):
    """Creates and returns the Gradio UI."""

    def call_llm(text):
        """Refines user input for clarity and consistency using LLM."""
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

    def search_houses(location, house_size, max_price, num_bedrooms, num_bathrooms, amenities, description, num_listings):
        """Retrieves property listings and enhances descriptions using LLM."""
        
        description = call_llm(description)  # Refine input description
        city, state = location.split(", ")

        user_prefs = {
            "state": state,
            "city": city,
            "house_size": house_size,
            "max_price": max_price,
            "num_bedrooms": num_bedrooms,
            "num_bathrooms": num_bathrooms,
            "amenities": ", ".join(amenities) if amenities else "None",
            "description": description
        }

        search_results = vector_db.search(user_prefs, num_listings)

        if not search_results:
            return [], []  # Empty lists to avoid breaking Gradio

        llm_augm = LlmAugmentation()
        augmented_results = [llm_augm.generate_augmented_descriptions(result) for result in search_results]

        # **Formatted Data for Text Table**
        formatted_text = [
            [
                f"{result['bedrooms']} Bed | {result['bathrooms']} Bath | {result['house_size']} sq ft",  # Title
                f"{result['state']}, {result['city']}, {result.get('neighborhood', 'N/A')}",  # Location
                f"${result['price']}",  # Price
                augmented_results[i] if i < len(augmented_results) else "N/A"  # Description
            ]
            for i, result in enumerate(search_results)
        ]

        # **List of Image Paths for Gallery**
        image_paths = [result["image_path"] for result in search_results]

        return formatted_text, image_paths

    with gr.Blocks() as demo:
        gr.Markdown("## Welcome to HomeMatch 🏡\nFill in your preferences and press 'Search' to find a home!")

        city_list = [
            "Tucson, Arizona", "Los Angeles, California", "San Francisco, California",
            "Santa Barbara, California", "Denver, Colorado", "Atlanta, Georgia",
            "Honolulu, Hawaii", "Chicago, Illinois", "Boston, Massachusetts",
            "Baltimore, Maryland", "Portland, Maine", "Las Vegas, Nevada",
            "Newark, New Jersey", "New York City, New York", "Cincinnati, Ohio",
            "Pittsburgh, Pennsylvania", "Nashville, Tennessee", "Houston, Texas",
            "Salt Lake City, Utah", "New Orleans, Louisiana"
        ]

        with gr.Row():
            location = gr.Dropdown(city_list, label="Location", info='Where would you like to search for a property?')
            house_size = gr.Textbox(label="House Size (sq ft)", placeholder="e.g., 2000")
            max_price = gr.Textbox(label="Maximum Price", placeholder="e.g., 500000")

        with gr.Row():
            amenities = gr.CheckboxGroup(["Pool", "Garage", "Garden", "Gym", "Fireplace", "Balcony", "Basement"], label="Select Amenities")
            description = gr.Textbox(label="Description", placeholder="Additional details about the house...")
            with gr.Column(scale=1):
                num_bedrooms = gr.Slider(1, 10, step=1, label="Number of Bedrooms", value=3)
                num_bathrooms = gr.Slider(1, 10, step=1, label="Number of Bathrooms", value=3)
            
            with gr.Column(scale=1):
                num_listings = gr.Slider(1, 10, step=1, label="How many listings do you wish to see?", value=3)
                search_button = gr.Button("Search")

        # **Results Display**
        with gr.Row():
            results_table = gr.Dataframe(
                headers=["Title", "Location", "Price", "Description"],
                datatype=["str", "str", "str", "str"],
                interactive=False,
                wrap=True
            )

        image_gallery = gr.Gallery(label="Listing Images", columns=3, height=300)

        # **Button Action**
        search_button.click(
            search_houses,
            inputs=[location, house_size, max_price, num_bedrooms, num_bathrooms, amenities, description, num_listings],
            outputs=[results_table, image_gallery]  # ✅ Now outputs images separately
        )

    return demo