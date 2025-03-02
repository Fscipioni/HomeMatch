import os
import json
from listings_generator import ListingsGenerator
from vector_database import RealEstateVectorStore
from user_preferences import UserPreferenceCollector
from answer_augmentation import LlmAugmentation

def main():
    """Generates listings, populates the vector DB, and simulates user input."""
    
    listings_path = "../Data/listings.json"
    db_path = "../Data/chroma_langchain_db"
    
# 1️⃣ Generate Listings if Not Present and Upload to Vector Database
    if not os.path.exists(listings_path):
        print("📝 Listings file not found. Generating real estate listings...")
        generator = ListingsGenerator(total_listings=10, batch_size=2)
        generator.generate_listings()

        print("\n📥 Loading listings into ChromaDB...")
        real_estate_db = RealEstateVectorStore()
        real_estate_db.store_listings()
    else:
        print("✅ Listings file found. Skipping generation and storing.")
    
    # 2️⃣ Simulate User Input (or collect real input)
    print("\n🗣️ Collecting user preferences...")
    collector = UserPreferenceCollector(interactive=False)  # Change to True for live input
    user_prefs = collector.collect_preferences()

    # 3️⃣ Retrieve Most Relevant Listings
    num_listings = int(input("How many listings do you wish to see? "))
    num_listings = min(num_listings, 10)  # Ensure max is 10

    retriever = RealEstateVectorStore()
    retrieved_documents = retriever.search(user_prefs, num_listings)

    if not retrieved_documents:
        print("⚠️ No listings found. Try adjusting your preferences.")
        return

    # 🔹 Convert Document objects to JSON-serializable dictionaries
    listings = [
        {
            "description": doc.page_content,  # Property description
            **doc.metadata  # Unpack all metadata
        }
        for doc in retrieved_documents
    ]

    # 4️⃣ LLM-Based Augmentation to Improve Listings Descriptions
    llm_augm = LlmAugmentation()
    
    # ✅ Pass structured listings (as a list of dicts) to the augmentation function
    response = llm_augm.generate_augmented_descriptions(listings)
    
    print("\n📌 Augmented Listings:\n", response)


if __name__ == "__main__":
    main()