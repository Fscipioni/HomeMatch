# HomeMatch - AI-Powered Real Estate Search

## Table of Contents
1. [Description](#description)
2. [Directory Structure](#directory-structure)
3. [Installation](#installation)
4. [Project Structure](#project-structure)
5. [Usage](#usage)
6. [Contributing](#contributing)
7. [Badges](#badges)

## Description
HomeMatch is an AI-powered real estate search application that enables users to find properties based on their preferences. It leverages LangChain and ChromaDB for efficient vector storage and retrieval, OpenAI’s GPT model for listing descriptions, and Diffusers for image generation. The project features a Gradio-based UI for seamless user interaction.

This project is the final submission for the [Udacicy](https://www.udacity.com/dashboard) Nanodegree Program “Generative AI.”

## Directory Structure
<pre>
HomeMatch/
│── 📂 Data/                     	# Stores JSON files, embeddings, etc.
│   ├── chroma_langchain_db      		# Stores embeddings and metadata of generated listings
│   ├── images/                  		# Stores AI-generated property images using Diffusers
│   ├── listings.json            		# Generated real estate listings
│   ├── city_choices.txt         		# State-City pairs to add variety in the generated listings
│── 📂 Dode/                     	# Python scripts for each step
│   ├── main.py                  		# Generates listings, uploads embeddings to the vector DB, and runs the chatbot
│   ├── listings_generator.py     		# Generates AI-powered property listings
│   ├── vector_database.py        		# Manages storage and search using ChromaDB
│   ├── answer_augmentation.py    		# Enhances descriptions using LLMs
│   ├── gradio_ui.py              		# Builds the Gradio frontend interface
│   ├── config_loader.py          		# Loads OpenAI API key  
│── requirements.txt              	# Python dependencies
│── README.md                     	# Project documentation
</pre>

## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/HomeMatch.git
   cd HomeMatch
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate Listings, Upload to Vector DB, and Start the Gradio App:**
   ```bash
   python main.py
   ```
This command:
- Generates AI-powered property listings.
- Stores them in ChromaDB.
- Launches the web-based interface for searching properties based on user preferences.


## Project Structure
### `main.py`
- The **initialization script** that runs `listings_generator.py` to create new listings if needed.
- Stores the generated listings in **ChromaDB** by calling `vector_database.py`.
- Ensures that the vector database is set up for searches.
- Runs the **Gradio-based UI** for property searches.

### `gradio_ui.py`
- Provides a structured layout with input fields for filtering listings and displaying results.
- Calls the `search` function from `vector_database.py` to retrieve relevant listings.

### `listings_generator.py`
- Uses **OpenAI’s GPT** to generate **diverse property descriptions**.
- Uses **Diffusers** to create **realistic AI-generated property images**.
- Loads city choices from `city_choices.txt` to ensure **varied locations**.

### `vector_database.py`
- Implements the `VectorDatabase` class using **LangChain and ChromaDB**.
- Handles **vector storage and similarity search** for property listings.
- Loads, stores, and retrieves listing embeddings efficiently.

### `answer_augmentation.py`
- Defines `LlmAugmentation` for **enhancing property details**.
- Uses **LangChain and OpenAI API** to **add contextual information** to listings.

### `config_loader.py`
- Loads configuration settings such as **database paths, model parameters, and API keys**.

## Usage
1. **OpenAI API key**
- In your home directory, create a file named ~/config.json.
- Store your OpenAI API key as follows:
```
{
    "OPENAI_API_KEY": "YOUR_PERSONAL_OPENAI_API_KEY",
    "VOCAREUM_OPENAI_API_KEY": "VOCAREUML_OPENAI_API_KEY"
}
```

2. **Run the project**
Launch the application from a terminal:
```bash
python main.py
```
Once the app is running:
1. Open the **public URL provided** by Gradio in your browser.
2. Enter your search criteria and navigate through the results using the interactive UI.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch: `git checkout -b feature-name`.
3. Make your changes.
4. Push your branch: `git push origin feature-name`.
5. Create a pull request.

## Badges
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange)
![LangChain](https://img.shields.io/badge/LangChain-VectorDB-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Storage-yellow)

