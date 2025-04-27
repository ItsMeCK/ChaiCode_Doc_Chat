import os
import logging
from urllib.parse import urlparse, urljoin, unquote
import re # Import regex for filtering
import sys
import requests # Import requests library
import json # To pretty-print the output dictionary

from bs4 import BeautifulSoup as Soup # Import BeautifulSoup
# *** Re-enable Langchain/Qdrant imports ***
from langchain_community.document_loaders import WebBaseLoader # Use WebBaseLoader for single pages
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams

# Import configuration loading
# Add backend directory to sys.path to import config
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
from core.config import settings # Re-enable settings import

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
BASE_DOCS_URL = "https://docs.chaicode.com"
# Start discovery from the 'Getting Started' page
MAIN_YOUTUBE_PAGE = urljoin(BASE_DOCS_URL, "/youtube/getting-started/")
# Recursion depth (not used in this version as we load subsections directly)
# RECURSION_DEPTH = 2
REQUESTS_TIMEOUT = 30 # Timeout for fetching discovery page

# --- Helper Functions ---

def sanitize_name(name_part: str) -> str | None:
    """Sanitizes a path segment for use as a dictionary key or collection name."""
    if not name_part:
        return None
    try:
        # Sanitize: replace hyphens/spaces with underscores, make lowercase
        name = name_part.replace('-', '_').replace(' ', '_').lower()
        # Remove potential query params or anchors just in case
        name = name.split('?')[0].split('#')[0]
        # Basic validation for Qdrant collection names (good enough for dict keys too)
        if re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_]*$", name):
             # Handle cases where the name might become empty after sanitization
            if not name:
                return None # Or assign a default if needed
            return name
        else:
            # Try removing invalid characters - more aggressive sanitization
            name = re.sub(r'[^a-zA-Z0-9_]', '', name)
            # Ensure it starts with a valid character
            if re.match(r"^[a-zA-Z0-9]", name):
                 if not name: return None
                 logging.warning(f"Aggressively sanitized '{name_part}' to '{name}'")
                 return name
            else:
                 logging.warning(f"Could not derive a valid sanitized name from segment: '{name_part}'")
                 return None
    except Exception as e:
        logging.error(f"Error sanitizing name part '{name_part}': {e}")
        return None


def discover_sections_and_subsections(main_page_url: str) -> dict:
    """
    Scrapes the provided starting page using `requests`, parses the HTML,
    finds the `ul.top-level` list, and extracts a nested structure of
    sections and their subsections with links.
    Returns: { "section_name": { "base_url": "...", "subsections": { "subsection_name": "url", ... } } }
    """
    logging.info(f"Discovering sections and subsections starting from: {main_page_url}")
    # Structure: { "section_name": { "base_url": "...", "subsections": { "subsection_name": "url", ... } } }
    sections_hierarchy = {}
    headers = { # Add headers to mimic a browser request
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        # Use requests to fetch raw HTML
        response = requests.get(main_page_url, headers=headers, timeout=REQUESTS_TIMEOUT)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        # Parse the fetched HTML content
        soup = Soup(response.text, "lxml")

        # --- Strategy: Directly target the first ul.top-level list ---
        logging.info("Attempting discovery using the first 'ul' with class 'top-level'...")
        top_level_ul = soup.find('ul', class_='top-level')

        if not top_level_ul:
            logging.error(f"Could not find any 'ul' with class 'top-level' on page {main_page_url}. Cannot discover structure.")
            return {}

        # Find all direct links (<a>) within list items (<li>) of this specific ul
        potential_links = top_level_ul.select('li > a[href]')
        logging.info(f"Found {len(potential_links)} potential links within the first ul.top-level on {main_page_url}.")

        for link in potential_links:
            href = link.get('href')
            try:
                # Normalize href before parsing
                if not href: continue
                decoded_href = unquote(href)
                parsed_href = urlparse(decoded_href)

                # Construct absolute URL for domain check
                absolute_href = urljoin(BASE_DOCS_URL, decoded_href)
                parsed_absolute_href = urlparse(absolute_href)

                # Check if it's within the target domain and starts with the expected path
                if parsed_absolute_href.netloc == urlparse(BASE_DOCS_URL).netloc and \
                   parsed_absolute_href.path.startswith('/youtube/'):

                    path_parts = [part for part in parsed_absolute_href.path.strip('/').split('/') if part]

                    if len(path_parts) >= 1 and path_parts[0] == 'youtube':
                        section_slug = None
                        subsection_slug = None
                        collection_name = None
                        subsection_name = None

                        if len(path_parts) == 1: # Base /youtube/ page -> getting_started
                            section_slug = "getting_started"
                            collection_name = "getting_started"
                        elif len(path_parts) >= 2:
                            section_slug = path_parts[1]
                            collection_name = sanitize_name(section_slug)
                            if len(path_parts) >= 3:
                                subsection_slug = path_parts[2]
                                subsection_name = sanitize_name(subsection_slug)

                        if not collection_name: # Skip if section name couldn't be sanitized
                             logging.warning(f"Skipping link '{href}' - could not derive valid section name from '{section_slug}'")
                             continue

                        # Ensure the main section entry exists
                        if collection_name not in sections_hierarchy:
                            section_base_path = f"/youtube/{section_slug}/" if section_slug != "getting_started" else "/youtube/getting-started/"
                            section_base_url = urljoin(BASE_DOCS_URL, section_base_path)
                            sections_hierarchy[collection_name] = {
                                "base_url": section_base_url, # Store base URL for reference if needed
                                "subsections": {}
                            }
                            logging.info(f"  Created section entry: '{collection_name}' -> {section_base_url}")

                        # Add subsection if applicable and valid
                        if subsection_name:
                             subsection_url = parsed_absolute_href._replace(query="", fragment="").geturl()
                             # Store the original subsection slug as key if sanitization fails but name exists
                             subsection_key = subsection_name if subsection_name else subsection_slug
                             if subsection_key and subsection_key not in sections_hierarchy[collection_name]["subsections"]:
                                 sections_hierarchy[collection_name]["subsections"][subsection_key] = subsection_url
                                 logging.debug(f"    Added subsection: '{subsection_key}' -> {subsection_url}")

            except Exception as e:
                logging.warning(f"Could not process link href '{href}': {e}")
                continue # Skip problematic links

        logging.info(f"Discovery complete. Found {len(sections_hierarchy)} main sections.")

    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP Request failed for {main_page_url}: {e}")
        return {}
    except Exception as e:
        logging.error(f"Error during section discovery from {main_page_url}: {e}", exc_info=True)
        return {}
    return sections_hierarchy


# --- Function for embedding and storing chunks for a section ---
def embed_and_store_chunks(chunks: list, collection_name: str):
    """
    Embeds a list of text chunks and stores them in the specified Qdrant collection.
    """
    if not chunks:
        logging.warning(f"No chunks provided for embedding in collection '{collection_name}'.")
        return False

    try:
        # 1. Embed and Store
        logging.info(f"Initializing OpenAI embeddings (Model: {settings.EMBEDDING_MODEL_NAME})...")
        embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL_NAME,
            openai_api_key=settings.OPENAI_API_KEY
        )

        logging.info(f"Connecting to Qdrant at {settings.QDRANT_URL}...")
        client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            prefer_grpc=False
        )

        # Get embedding dimension dynamically
        try:
            dummy_embedding = embeddings.embed_query("test")
            vector_size = len(dummy_embedding)
            logging.info(f"Detected embedding vector size: {vector_size}")
        except Exception as e:
            logging.error(f"Failed to determine embedding vector size: {e}")
            vector_size = 1536 # Fallback
            logging.warning(f"Falling back to default vector size: {vector_size}. Ensure this matches your embedding model!")

        # Ensure the collection exists
        try:
            collection_info = client.get_collection(collection_name=collection_name)
            logging.info(f"Collection '{collection_name}' already exists.")
            # Determine existing vector size (handle different config types)
            existing_vector_size = None
            if isinstance(collection_info.vectors_config, models.VectorParams):
                existing_vector_size = collection_info.vectors_config.size
            elif isinstance(collection_info.vectors_config, models.NamedVectorParams) and 'vector' in collection_info.vectors_config.map:
                 existing_vector_size = collection_info.vectors_config.map['vector'].size

            if existing_vector_size is not None and existing_vector_size != vector_size:
                 logging.error(f"Vector size mismatch for collection '{collection_name}'! Expected {vector_size}, found {existing_vector_size}. Embedding aborted for this section.")
                 return False # Indicate failure
            elif existing_vector_size is None:
                 logging.warning(f"Could not verify vector size for collection '{collection_name}' due to unexpected config type: {type(collection_info.vectors_config)}")

        except Exception as e:
             logging.info(f"Collection '{collection_name}' not found or error checking: {e}. Creating...")
             try:
                 client.create_collection(
                     collection_name=collection_name,
                     vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
                 )
                 logging.info(f"Collection '{collection_name}' created.")
             except Exception as create_e:
                 logging.error(f"Failed to create collection '{collection_name}': {create_e}", exc_info=True)
                 return False # Indicate failure

        logging.info(f"Embedding {len(chunks)} chunks and loading into Qdrant collection: '{collection_name}'...")

        qdrant = Qdrant.from_documents(
            chunks, # Use the provided list of chunks
            embeddings,
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            prefer_grpc=False,
            collection_name=collection_name,
            force_recreate=False, # Set to True if you want to overwrite existing collection content on each run
        )
        logging.info(f"Successfully loaded {len(chunks)} chunks into collection: '{collection_name}'")
        return True # Indicate success

    except ImportError as e:
        logging.error(f"Import error during embedding/storage for {collection_name}: {e}.")
        return False # Indicate failure
    except Exception as e:
        logging.error(f"Failed to embed/store chunks for section {collection_name}: {e}", exc_info=True)
        return False # Indicate failure


# --- Main Execution Logic ---
if __name__ == "__main__":
    logging.info("Starting Chai Code documentation ingestion process (Embed Subsections Only)...")

    # --- Pre-run Checks ---
    # Ensure necessary libraries are installed and config is present
    try: import lxml
    except ImportError: logging.error("lxml parser required: pip install lxml"); sys.exit(1)
    try: import requests
    except ImportError: logging.error("requests library required: pip install requests"); sys.exit(1)
    if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY == "your_openai_api_key_here":
        logging.error("OpenAI API Key not configured in .env file. Aborting.")
        sys.exit(1)
    if not settings.QDRANT_URL:
        logging.error("Qdrant URL not configured in .env file. Aborting.")
        sys.exit(1)
    # --- End Pre-run Checks ---

    # 1. Discover Sections and Subsections Dynamically
    discovered_structure = discover_sections_and_subsections(MAIN_YOUTUBE_PAGE)

    if not discovered_structure:
        logging.error("No structure was discovered. Exiting.")
        sys.exit(1)

    # 2. Print the discovered structure (Optional, keep for verification)
    logging.info(f"\n--- Discovered Structure ({len(discovered_structure)} sections) ---")
    # print(json.dumps(discovered_structure, indent=2)) # Keep if you want to see the structure
    logging.info("--- End of Discovered Structure ---")


    # --- Section processing: Scrape and embed only subsections ---
    logging.info(f"\nProceeding to scrape and embed subsections for {len(discovered_structure)} discovered sections...")
    total_sections = len(discovered_structure)
    processed_sections_count = 0
    failed_sections = []
    total_chunks_embedded = 0

    # Initialize text splitter once
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )

    # Iterate through the discovered main sections
    for collection_name, section_data in discovered_structure.items():
        subsections = section_data.get("subsections", {})
        base_url = section_data.get("base_url") # Keep for logging context
        logging.info(f"\n--- Processing Section: {collection_name} (Base URL: {base_url}) ---")

        if not subsections:
            logging.warning(f"No subsections found for '{collection_name}'. Skipping embedding for this section.")
            continue # Skip to the next main section

        all_splits_for_section = []
        subsection_load_errors = 0

        # Iterate through the subsections of the current main section
        for subsection_name, subsection_url in subsections.items():
            logging.info(f"  Loading subsection: '{subsection_name}' from {subsection_url}")
            try:
                # Load the single subsection page
                loader = WebBaseLoader([subsection_url])
                docs = loader.load()

                if not docs:
                    logging.warning(f"  No document loaded from subsection URL: {subsection_url}")
                    subsection_load_errors += 1
                    continue

                # Add metadata (source is the specific subsection URL)
                for doc in docs:
                    doc.metadata["source"] = subsection_url
                    doc.metadata["namespace"] = collection_name # Main section name is the namespace
                    doc.metadata["subsection"] = subsection_name # Add subsection name

                # Split the loaded document(s)
                splits = text_splitter.split_documents(docs)
                logging.info(f"  Split subsection '{subsection_name}' into {len(splits)} chunks.")
                all_splits_for_section.extend(splits)

            except Exception as e:
                logging.error(f"  Failed to load or split subsection '{subsection_name}' ({subsection_url}): {e}", exc_info=False) # Less detail for loading errors
                subsection_load_errors += 1
                continue # Skip to next subsection

        # After processing all subsections for the current section, embed and store the collected chunks
        if all_splits_for_section:
            logging.info(f"Collected {len(all_splits_for_section)} chunks from {len(subsections) - subsection_load_errors} successfully loaded subsections for '{collection_name}'.")
            # Call function to embed and store these chunks in the main section's collection
            success = embed_and_store_chunks(all_splits_for_section, collection_name)
            if success:
                processed_sections_count += 1
                total_chunks_embedded += len(all_splits_for_section)
            else:
                failed_sections.append(collection_name)
        elif subsections: # Only log if subsections existed but failed/yielded no splits
             logging.warning(f"No content chunks generated from any subsections for '{collection_name}'.")
             # Optionally count this as a failure if subsections existed
             if subsection_load_errors == len(subsections): # All subsections failed to load
                  failed_sections.append(collection_name)


    logging.info("\n--- Ingestion Summary ---")
    logging.info(f"Total main sections discovered: {total_sections}")
    logging.info(f"Sections successfully processed (subsections embedded): {processed_sections_count}")
    logging.info(f"Sections failed (no subsections embedded or error): {len(failed_sections)}")
    logging.info(f"Total chunks embedded across all sections: {total_chunks_embedded}")
    if failed_sections:
        logging.warning(f"Failed sections: {', '.join(failed_sections)}")
    # --- End of section processing ---

    logging.info("\nIngestion process (embedding subsections only) finished.")

