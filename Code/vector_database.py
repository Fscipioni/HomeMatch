import json
from uuid import uuid4
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.documents import Document

class RealEstateVectorStore:
    def __init__(self, listings_path="./Data/listings.json", db_path="./Data/chroma_langchain_db"):
        """
        Initializes the vector store by loading listings and setting up ChromaDB.
        """
        self.listings_path = listings_path
        self.db_path = db_path
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
        
        # Load Listings
        self.listings = self._load_listings()
        self.documents = self._prepare_documents()
        
        # Initialize ChromaDB
        self.vector_store = Chroma(
            collection_name="real_estate_listings",
            embedding_function=self.embedding_model,
            persist_directory=self.db_path
        )

    def _load_listings(self):
        """Loads the real estate listings from a JSON file."""
        with open(self.listings_path, "r") as f:
            return json.load(f)

    def _prepare_documents(self):
        """Converts listings into Document objects with metadata."""
        return [
            Document(
                page_content=listing["Description"],  # Store main listing text
                metadata={
                    "id": str(uuid4()),  # Unique ID
                    "neighborhood": listing["Neighborhood"],
                    "city": listing["City"],
                    "state": listing["State"],
                    "price": listing["Price"],
                    "bedrooms": listing["Bedrooms"],
                    "bathrooms": listing["Bathrooms"],
                    "house_size": listing["House Size"],
                    "neighborhood_description": listing["Neighborhood Description"],
                }
            )
            for listing in self.listings
        ]

    def store_listings(self):
        """Stores embeddings and metadata in ChromaDB."""
        self.vector_store.add_documents(self.documents)
        print("✅ Listings successfully stored in ChromaDB!")

    # def search(self, query, k=5):
    #     """Performs a similarity search for a given query."""
    #     results = self.vector_store.similarity_search(query, k=k)
    #     return results

# Usage Example
if __name__ == "__main__":
    real_estate_db = RealEstateVectorStore()
    real_estate_db.store_listings()
    
    # # Example Query
    # search_results = real_estate_db.search("luxury house with ocean view", k=3)
    # for result in search_results:
    #     print(result.metadata)