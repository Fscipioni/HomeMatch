import gradio as gr
import openai
from vector_database import VectorDatabase 
from answer_augmentation import LlmAugmentation

# Initialize VectorDatabase
# vector_db = VectorDatabase()

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
        
    # **Search Function**
    def search_houses(state, city, house_size, max_price, num_bedrooms, num_bathrooms, amenities, description, num_listings):
        """Retrieves property listings and enhances descriptions using LLM."""
        
        description = call_llm(description)  # Refine input description

        # **Format User Preferences**
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

        # **Retrieve Listings from Vector Database**
        search_results = vector_db.search(user_prefs, num_listings)

        if not search_results:
            return [["❌ No matching listings found.", "", "", "", ""]], []  # Return empty image list

        # **Augment Descriptions**
        llm_augm = LlmAugmentation()
        augmented_results = [llm_augm.generate_augmented_descriptions(result) for result in search_results]

        # print('\n🔍 Augmented Results:', augmented_results, '\n')

        # **Format Results for DataFrame Output**
        formatted_results = [
            [
                f"{result['bedrooms']} Bed | {result['bathrooms']} Bath | {result['house_size']} sq ft", # Title
                f"${result['price']}",
                f"{result['city']}, {result['state']}",
                augmented_results[i] if i < len(augmented_results) else "N/A"
            ]
            for i, result in enumerate(search_results)
        ]
        # print('search_results ', search_results, '\n')
        # **Extract Images for Gradio Gallery**
        image_paths = [result["image_path"] for result in search_results]

        # print("🔍 Final Formatted Results:", formatted_results)
        
        return formatted_results, image_paths  # ✅ Return listings + images


    # **Gradio Interface**
    with gr.Blocks() as demo:
        gr.Markdown("## Welcome to HomeMatch 🏡\nFill in your preferences and press 'Search' to find a home!")

        with gr.Row():
            state = gr.Textbox(label="State", placeholder="Enter the state (e.g., California)")
            city = gr.Textbox(label="City", placeholder="Enter the city (e.g., San Francisco)")

        with gr.Row():
            house_size = gr.Textbox(label="House Size (sq ft)", placeholder="e.g., 2000")
            max_price = gr.Textbox(label="Maximum Price", placeholder="e.g., 500000")

        with gr.Row():
            num_bedrooms = gr.Slider(1, 10, step=1, label="Number of Bedrooms", value=3)
            num_bathrooms = gr.Slider(1, 10, step=1, label="Number of Bathrooms", value=3)

        amenities = gr.CheckboxGroup(["Pool", "Garage", "Garden", "Gym", "Fireplace", "Balcony", "Basement"], label="Select Amenities")

        description = gr.Textbox(label="Description", placeholder="Additional details about the house...")

        num_listings = gr.Slider(1, 10, step=1, label="How many listings do you wish to see?", value=3)

        search_button = gr.Button("Search")

        # **Results Display**
        with gr.Row():
            results_output = gr.Dataframe(
                headers=["Title", "Price", "Location", "Description"],
                datatype=["str", "str", "str", "str"],
                interactive=False,
                wrap=True
            )

        image_output = gr.Gallery(label="Listing Images")

        print('results_output ', results_output)
        print('image_output', image_output)

        # **Button Action**
        search_button.click(
            search_houses,
            inputs=[state, city, house_size, max_price, num_bedrooms, num_bathrooms, amenities, description, num_listings],
            outputs=[results_output, image_output]  # ✅ Now outputs images!
        )

    return demo

# # **Run the Gradio App**
# if __name__ == "__main__":
#     demo.queue().launch(share=True, allowed_paths=["/Users/francescascipioni/Library/Mobile Documents/com~apple~CloudDocs/Work/Online courses/Nanodegrees/Generative AI Nanodegree/05 - Final Project/HomeMatch/Data/Images"])

# formatted_results = search_houses('California', 'Los Angeles', '1500', '500000', '3', '2', ['garden', 'garge'], 
# 'Stunning mansion in the prestigious neighborhood of Bel Air. This luxurious property features elegant architecture, expansive 
# living spaces, and breathtaking views of the city. Perfect for those seeking the ultimate in luxury living.', 3)