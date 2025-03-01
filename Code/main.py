import os
import json
from listings_generator import ListingsGenerator
from vector_database import RealEstateVectorStore
from user_preferences import UserPreferenceCollector

def main():
    """Main script to test the system: Generates listings, populates the vector DB, and simulates user input."""
    
    listings_path = "./Data/listings.json"
    db_path = "./Data/Data/chroma_langchain_db"
    
    # 1️⃣ Generate Listings if Not Present and upload to Vector Database
    if not os.path.exists(listings_path):
        print("📝 Listings file not found. Generating real estate listings...")
        generator = ListingsGenerator(total_listings=10, batch_size=2)
        generator.generate_listings()

        print("\n📥 Loading listings into ChromaDB...")
        real_estate_db = RealEstateVectorStore()
        real_estate_db.store_listings()
    else:
        print("✅ Listings file found. Skipping generation and storing.")
    
    # # 2️⃣ Populate Vector Database
    # if not os.path.exists(db_path):
    #     print("\n📥 Loading listings into ChromaDB...")
    #     real_estate_db = RealEstateVectorStore()
    #     real_estate_db.store_listings()
    # else:
    #     print("✅ Listings file found. Skipping generation.")
    
    # 3️⃣ Simulate User Input (or collect real input)
    print("\n🗣️ Collecting user preferences...")
    collector = UserPreferenceCollector(interactive=False)  # Change to True for live input
    user_prefs = collector.collect_preferences()
    # print(user_prefs)

    
    # print("\n🎯 User Preferences (Structured):")
    # print(json.dumps(user_prefs, indent=4))

    # 4 Retrieve most relevant listings

    num_listings = int(input("How many listings do you wish to see?"))

    if num_listings > 10: 
        num_listings = 10

    retriever = RealEstateVectorStore()
    answers = retriever.search(user_prefs, num_listings)
    print(answers)

    # LLM answers augmentation to improve answers description



if __name__ == "__main__":
    main()