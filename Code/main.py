import os
import json
from listings_generator import ListingsGenerator
from vector_database import RealEstateVectorStore
from user_preferences import UserPreferenceCollector

def main():
    """Main script to test the system: Generates listings, populates the vector DB, and simulates user input."""
    
    listings_path = "listings.json"
    
    # 1️⃣ Generate Listings if Not Present
    if not os.path.exists(listings_path):
        print("📝 Listings file not found. Generating real estate listings...")
        generator = ListingsGenerator(total_listings=10, batch_size=2)
        generator.generate_listings()
    else:
        print("✅ Listings file found. Skipping generation.")
    
    # # 2️⃣ Populate Vector Database
    # print("\n📥 Loading listings into ChromaDB...")
    # real_estate_db = RealEstateVectorStore()
    # real_estate_db.store_listings()
    
    # 3️⃣ Simulate User Input (or collect real input)
    print("\n🗣️ Collecting user preferences...")
    collector = UserPreferenceCollector(interactive=False)  # Change to True for live input
    user_prefs = collector.collect_preferences()
    
    print("\n🎯 User Preferences (Structured):")
    print(json.dumps(user_prefs, indent=4))

if __name__ == "__main__":
    main()