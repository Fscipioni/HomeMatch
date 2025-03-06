"""
Module: vector_database
Description: Handles storing and searching real estate listings using ChromaDB and OpenAI embeddings.
"""

import json
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

class VectorDatabase:
    """Handles vector-based storage and retrieval of real estate listings using ChromaDB."""

    def __init__(self, listings_path="../Data/listings.json", db_path="../Data/chroma_langchain_db"):
        """
        Initializes the vector store by loading real estate listings and setting up ChromaDB.

        Args:
            listings_path (str): Path to the JSON file containing real estate listings.
            db_path (str): Path to the directory where ChromaDB stores embeddings.
        """
        self.listings_path = listings_path
        self.db_path = db_path
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

        # Load and process listings
        self.listings = self._load_listings()
        self.documents = self._prepare_documents()

        # Initialize ChromaDB for storage
        self.vector_store = Chroma(
            collection_name="real_estate_listings",
            embedding_function=self.embedding_model,
            persist_directory=self.db_path
        )

    def _load_listings(self):
        """
        Loads real estate listings from a JSON file.

        Returns:
            list: A list of real estate listings.
        """
        try:
            with open(self.listings_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"❌ Error loading listings file: {e}")
            return []

    def _prepare_documents(self):
        """
        Converts listings into Document objects with metadata.

        Returns:
            list: A list of Document objects with structured metadata.
        """
        return [
            Document(
                page_content=listing["Description"],  # Store property description
                metadata={
                    "id": listing["id"],  
                    "property_type": listing["Property Type"],
                    "neighborhood": listing["Neighborhood"],
                    "city": listing["City"],
                    "state": listing["State"],
                    "price": listing["Price"],
                    "house_size": listing["House Size"],
                    "bedrooms": listing["Bedrooms"],
                    "bathrooms": listing["Bathrooms"],
                    "neighborhood_description": listing["Neighborhood Description"],
                    "image_path": listing["image_path"]
                }
            )
            for listing in self.listings
        ]

    def store_listings(self):
        """
        Stores real estate listings in ChromaDB.
        """
        try:
            self.vector_store.add_documents(self.documents)
            print("✅ Listings successfully stored in ChromaDB!")
        except Exception as e:
            print(f"❌ Error storing listings: {e}")


    def format_user_prefs(self, user_prefs):
        """
        Converts structured user preferences into a readable search query.

        Args:
            user_prefs (dict): Dictionary containing user preferences.

        Returns:
            str: A natural language query string for embedding.
        """
        try:
            return (
                f"Looking for a property in {', '.join(user_prefs.get('city', []))}, {', '.join(user_prefs.get('state', []))}. "
                f"House size preference: {user_prefs.get('house_size', 'any size')}. "
                f"Maximum price: {user_prefs.get('max_price', '100000')}. "
                f"Number of Bedrooms: {user_prefs.get('num_bedrooms', 3)}. "
                f"Number of Bathrooms: {user_prefs.get('num_bathrooms', 3)}. "
                f"Amenities: {', '.join(user_prefs.get('amenities', []))}. "
                f"Property description: {user_prefs.get('description', 'no preference')}."
            )
        except Exception as e:
            print(f"❌ Error formatting user preferences: {e}")
            return ""

    def search(self, user_prefs, k=5):
        """
        Performs a similarity search based on user preferences and retrieves matching listings with images.

        Args:
            user_prefs (dict): Dictionary containing user search preferences.
            k (int): Number of top matches to return.

        Returns:
            list: A list of dictionaries containing listing details and image paths.
        """
        try:
            # Convert user preferences into a natural language query
            query = self.format_user_prefs(user_prefs)

            # Generate embeddings for the query
            query_embedding = self.embedding_model.embed_query(query)

            if not isinstance(query_embedding, list):
                raise ValueError("❌ Embedding function did not return a valid vector list.")

            # Perform similarity search using the embedding
            results = self.vector_store.similarity_search_by_vector(query_embedding, k=k)

            # Extract relevant metadata, including image paths
            listings_with_images = [
                {
                    "description": doc.page_content,
                    "id": doc.metadata.get("id"),
                    "city": doc.metadata.get("city", "Unknown"),
                    "state": doc.metadata.get("state", "Unknown"),
                    "price": doc.metadata.get("price", "N/A"),
                    "bedrooms": doc.metadata.get("bedrooms", "N/A"),
                    "bathrooms": doc.metadata.get("bathrooms", "N/A"),
                    "house_size": doc.metadata.get("house_size", "N/A"),
                    "neighborhood": doc.metadata.get("neighborhood", "Unknown"),
                    "neighborhood_description": doc.metadata.get("neighborhood_description", ""),
                    "image_path": doc.metadata.get("image_path", "❌ No image available")  # Ensure image path is included
                }
                for doc in results
            ]

            return listings_with_images

        except Exception as e:
            print(f"❌ Error during search: {e}")
            return []

