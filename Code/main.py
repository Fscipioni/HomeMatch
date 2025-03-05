import os
import gradio as gr
from listings_generator import ListingsGenerator
from vector_database import VectorDatabase
from gradio_ui import create_gradio_interface



"""Generates listings, populates the vector DB, and simulates user input."""
    
listings_path = "../Data/listings.json"
vector_db = VectorDatabase()
    
# Generate Listings if Not Present and Upload to Vector Database
if not os.path.exists(listings_path):
    print("📝 Listings file not found. Generating real estate listings...")
    generator = ListingsGenerator(total_listings=1000, batch_size=10)
    generator.generate_listings()

    print("\n📥 Loading listings into ChromaDB...")
    real_estate_db = VectorDatabase()
    real_estate_db.store_listings()
else:
    print("✅ Listings file found. Skipping generation and storing.")



if __name__ == "__main__":
    demo = create_gradio_interface(vector_db)
    if demo is None:
        raise ValueError("❌ create_gradio_interface() did not return a valid Gradio Blocks object.")
    demo.queue().launch(share=True, allowed_paths=["/Users/francescascipioni/Library/Mobile Documents/com~apple~CloudDocs/Work/Online courses/Nanodegrees/Generative AI Nanodegree/05 - Final Project/HomeMatch/Data/Images"])