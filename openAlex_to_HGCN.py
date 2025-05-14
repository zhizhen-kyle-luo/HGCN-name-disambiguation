import os
import json
import argparse
import xml.etree.ElementTree as ET
from nameparser import HumanName
import requests
from collections import defaultdict
import sys

def ensure_directory(path):
    """Ensure that a directory exists."""
    os.makedirs(path, exist_ok=True)

def fetch_author_data(author_name, max_results=200):
    """
    Fetch author data from OpenAlex API
    Args:
        author_name: Name of the author to disambiguate
        max_results: Maximum number of results to return
    
    Returns:
        dict with author IDs as keys and author data as values
    """
    print(f"Fetching author data for {author_name}...")
    cursor = "*"
    authors_data = {}
    result_count = 0
    
    while True and result_count < max_results:
        query_url = f'https://api.openalex.org/authors?search={author_name}&per_page=100&cursor={cursor}'
        
        try:
            response = requests.get(query_url)
            if response.status_code != 200:
                print(f"Error fetching data: {response.status_code}")
                break
                
            data = response.json()
            authors = data["results"]
            
            if not authors:
                break
                
            # Process authors
            for author in authors:
                name = HumanName(author.get("display_name", ""))
                
                # Skip if the name doesn't match
                first_name = name.first.lower()
                last_name = name.last.lower()
                target_name_parts = author_name.lower().split()
                
                # Stricter name matching
                # The query's first name part must match the candidate's first name
                # The query's last name part must match the candidate's last name
                # This handles cases where query is "First Last" and candidate is "First Middle Last"
                
                query_first = ""
                query_last = "" # Default to empty string

                if len(target_name_parts) > 0:
                    query_first = target_name_parts[0]
                # Use the last part of the query as the last name component
                if len(target_name_parts) > 1: 
                    query_last = target_name_parts[-1]
                elif len(target_name_parts) == 1: # If only one name part in query, assume it could be first or last
                    # If we only have one query part, we can't enforce first AND last match
                    # For now, let's stick to the logic that if query_last is not set, it's not checked strictly
                    # This means a query like "Fry" would rely on query_last matching candidate_last_normalized
                    pass


                candidate_first_normalized = name.first.lower()
                candidate_last_normalized = name.last.lower()
                
                match = False
                if query_first and query_last: # Query like "Terry Fry"
                    if candidate_first_normalized == query_first and candidate_last_normalized == query_last:
                        match = True
                elif query_first: # Query like "Terry" (and query_last is empty)
                    if candidate_first_normalized == query_first:
                        # This could be "Terry Smith" for a query "Terry".
                        # To be stricter for single name queries, one might want to check if candidate_last_normalized is empty or also matches.
                        # For now, this allows matching on first name if query is just one word.
                        match = True 
                elif query_last: # Query like "Fry" (and query_first is empty, implies single word query "Fry")
                    if candidate_last_normalized == query_last:
                        match = True
                
                if not match:
                    continue
                
                # Extract needed data
                author_id = author["id"].replace("https://openalex.org/", "")
                authors_data[author_id] = {
                    "id": author_id,
                    "name": author.get("display_name", ""),
                    "name_first": name.first,
                    "name_middle": name.middle,
                    "name_last": name.last,
                    "works_count": author.get("works_count", 0),
                    "works": []
                }
                
                result_count += 1
                if result_count >= max_results:
                    break
            
            # Update cursor for next page
            cursor = data["meta"].get("next_cursor")
            if not cursor:
                break
                
        except Exception as e:
            print(f"Error fetching author data: {e}")
            break
    
    print(f"Found {len(authors_data)} authors matching {author_name}")
    return authors_data

def fetch_works_for_author(author_id, max_works=100):
    """
    Fetch works (publications) for a specific author ID from OpenAlex API
    """
    print(f"Fetching works for author ID {author_id}...")
    cursor = "*"
    works = []
    fetched_count = 0
    
    while True and fetched_count < max_works:
        query_url = f'https://api.openalex.org/works?filter=author.id:{author_id}&per_page=100&cursor={cursor}'
        
        try:
            response = requests.get(query_url)
            if response.status_code != 200:
                print(f"Error fetching works: {response.status_code}")
                break
                
            data = response.json()
            batch_works = data["results"]
            
            if not batch_works:
                break
                
            for work in batch_works:
                # Extract needed fields
                work_id = work["id"].replace("https://openalex.org/", "")
                
                # Get authors
                authors = []
                for authorship in work.get("authorships", []):
                    if "author" in authorship:
                        author_name = authorship["author"].get("display_name", "")
                        author_id = authorship["author"]["id"].replace("https://openalex.org/", "")
                        authors.append({"name": author_name, "id": author_id})
                
                # Get venue
                venue_name = ""
                if "primary_location" in work and work["primary_location"] and "source" in work["primary_location"]:
                    venue_name = work["primary_location"]["source"].get("display_name", "")
                
                # Create work entry
                work_entry = {
                    "id": work_id,
                    "title": work.get("title", ""),  # OpenAlex may return None for title
                    "year": work.get("publication_year", 0),
                    "authors": authors,
                    "venue": venue_name
                }
                
                # Ensure title is never None
                if not work_entry["title"]:
                    work_entry["title"] = "Untitled publication"
                
                works.append(work_entry)
                fetched_count += 1
                
                if fetched_count >= max_works:
                    break
            
            # Update cursor for next page
            cursor = data["meta"].get("next_cursor")
            if not cursor:
                break
                
        except Exception as e:
            print(f"Error fetching works: {e}")
            break
    
    print(f"Found {len(works)} works for author ID {author_id}")
    return works

def create_xml_file(author_name, author_data, works_data, author_id_to_label=None):
    """
    Create XML file in the format expected by HGCN name disambiguation
    """
    print(f"Creating XML file for {author_name}...")
    
    # Create mapping from author ID to label if not provided
    if author_id_to_label is None:
        author_id_to_label = {}
        for i, author_id in enumerate(author_data.keys()):
            author_id_to_label[author_id] = str(i)
    
    # Helper function to escape XML special characters
    def escape_xml(text):
        if text is None:
            return ""
        # Replace XML special characters
        text = str(text)
        text = text.replace("&", "&amp;")
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")
        text = text.replace("\"", "&quot;")
        text = text.replace("'", "&apos;")
        # Remove any control characters that would break XML
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\t\n\r')
        return text
    
    # Format the XML with proper indentation to match existing files
    # Create the XML content as a string with manual formatting
    xml_str = '<?xml version="1.0" encoding="utf-8"?>\n'
    xml_str += '<person>\n'
    
    # Use the first author ID as the personID
    first_author_id = next(iter(author_data.keys()))
    xml_str += f'\t<personID>{escape_xml(first_author_id)}</personID>\n'
    
    xml_str += f'\t<FullName>{escape_xml(author_name)}</FullName>\n'
    xml_str += f'\t<FirstName>{escape_xml(author_name.split()[0])}</FirstName>\n'
    xml_str += f'\t<LastName>{escape_xml(author_name.split()[-1])}</LastName>\n'
    
    # Track unique works to avoid duplicates
    unique_works = {}
    
    # Add publications
    for author_id, author in author_data.items():
        for work_id in author["works"]:
            if work_id in works_data and work_id not in unique_works:
                work = works_data[work_id]
                unique_works[work_id] = work
                
                # Ensure title is never None or empty
                title_text = work["title"] if work["title"] else "Untitled publication"
                
                # Add publication element
                xml_str += '\t<publication>\n'
                xml_str += f'\t\t<title>{escape_xml(title_text)}</title>\n'
                xml_str += f'\t\t<year>{escape_xml(work["year"])}</year>\n'
                
                # Join authors
                authors_text = ", ".join([a["name"] for a in work["authors"]])
                xml_str += f'\t\t<authors>{escape_xml(authors_text)}</authors>\n'
                
                # Add venue
                venue_text = work["venue"] if work["venue"] else "Unknown"
                xml_str += f'\t\t<jconf>{escape_xml(venue_text)}</jconf>\n'
                
                # Add ID
                xml_str += f'\t\t<id>{escape_xml(work_id)}</id>\n'
                
                # Add label (which author ID this publication belongs to)
                # We'll map each unique author ID to a unique integer for the label
                xml_str += f'\t\t<label>{escape_xml(author_id_to_label.get(author_id, "0"))}</label>\n'
                
                # Add organization (use OpenAlex author institution if available, otherwise "null")
                xml_str += f'\t\t<organization>{escape_xml("null")}</organization>\n'
                
                xml_str += '\t</publication>\n'
    
    xml_str += '</person>'
    
    # Create directory if it doesn't exist
    ensure_directory("raw-data-temp")
    
    # Write XML file
    file_path = os.path.join("raw-data-temp", f"{author_name}.xml")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(xml_str)
    
    print(f"XML file created: {file_path}")
    return unique_works

def create_author_pair_file(author_name, works_data):
    """
    Create author pair file in the format expected by HGCN name disambiguation
    """
    print(f"Creating author pair file for {author_name}...")
    
    # Map from publication ID to index
    pub_to_idx = {}
    for i, (pub_id, _) in enumerate(works_data.items()):
        pub_to_idx[pub_id] = i
    
    # Track co-author relationships
    co_author_pairs = []
    
    # For each publication
    for pub_id, pub in works_data.items():
        pub_idx = pub_to_idx[pub_id]
        
        # For each author in the publication
        for i in range(len(pub["authors"])):
            for j in range(i+1, len(pub["authors"])):
                author_i = pub["authors"][i]
                author_j = pub["authors"][j]
                
                co_author_pairs.append((pub_idx, pub_idx, author_i["name"], author_j["name"]))
    
    # Create directory if it doesn't exist
    ensure_directory(os.path.join("experimental-results", "authors"))
    
    # Write author pair file
    file_path = os.path.join("experimental-results", "authors", f"{author_name}_authorlist.txt")
    with open(file_path, 'w', encoding='utf-8') as f:
        for pair in co_author_pairs:
            f.write(f"{pair[0]}\t{pair[1]}\t{pair[2]}\t{pair[3]}\n")
    
    print(f"Author pair file created: {file_path}")

def create_venue_pair_file(author_name, works_data):
    """
    Create venue pair file in the format expected by HGCN name disambiguation
    """
    print(f"Creating venue pair file for {author_name}...")
    
    # Map from publication ID to index
    pub_to_idx = {}
    for i, (pub_id, _) in enumerate(works_data.items()):
        pub_to_idx[pub_id] = i
    
    # Group publications by venue
    venues = defaultdict(list)
    for pub_id, pub in works_data.items():
        venue = pub["venue"]
        venues[venue].append(pub_id)
    
    # Create venue pairs
    venue_pairs = []
    for venue, pubs in venues.items():
        for i in range(len(pubs)):
            for j in range(i+1, len(pubs)):
                pub_i = pubs[i]
                pub_j = pubs[j]
                idx_i = pub_to_idx[pub_i]
                idx_j = pub_to_idx[pub_j]
                venue_pairs.append((idx_i, idx_j, venue, venue))
    
    # Create directory if it doesn't exist
    ensure_directory("experimental-results")
    
    # Write venue pair file
    file_path = os.path.join("experimental-results", f"{author_name}_jconfpair.txt")
    with open(file_path, 'w', encoding='utf-8') as f:
        for pair in venue_pairs:
            f.write(f"{pair[0]}\t{pair[1]}\t{pair[2]}\t{pair[3]}\n")
    
    print(f"Venue pair file created: {file_path}")

def save_data_to_json(author_name, author_data, works_data, author_id_to_label):
    """Save the fetched data to JSON for future use"""
    print(f"Saving data for {author_name} to JSON...")
    
    data = {
        "author_name": author_name,
        "author_data": author_data,
        "works_data": works_data,
        "author_id_to_label": author_id_to_label
    }
    
    # Create directory if it doesn't exist
    ensure_directory("cache")
    
    # Write JSON file
    file_path = os.path.join("cache", f"{author_name}_data.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"Data saved to: {file_path}")

def load_data_from_json(author_name):
    """Load data from JSON if available"""
    file_path = os.path.join("cache", f"{author_name}_data.json")
    
    if os.path.exists(file_path):
        print(f"Loading cached data for {author_name}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data["author_data"], data["works_data"], data["author_id_to_label"]
    
    return None, None, None

def fetch_works_only(author_id, author_name, max_works=100):
    """Function to fetch works only for a specific author ID"""
    works = fetch_works_for_author(author_id, max_works)
    
    # Save to cache
    cache_dir = os.path.join("cache", "works")
    ensure_directory(cache_dir)
    
    file_path = os.path.join(cache_dir, f"{author_name}_{author_id}_works.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(works, f, indent=2)
    
    print(f"Works for author {author_id} saved to: {file_path}")
    return works

def create_files_from_cache(author_name):
    """Create XML and pair files from cached data"""
    # Load author data
    author_data = {}
    works_data = {}
    author_id_to_label = {}
    
    # Check if we have the main cache file
    main_cache = os.path.join("cache", f"{author_name}_data.json")
    if os.path.exists(main_cache):
        with open(main_cache, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        author_data = data["author_data"]
        works_data = data.get("works_data", {})
        author_id_to_label = data["author_id_to_label"]
    
    # Check for individual works cache files
    works_cache_dir = os.path.join("cache", "works")
    if os.path.exists(works_cache_dir):
        for file in os.listdir(works_cache_dir):
            if file.startswith(f"{author_name}_") and file.endswith("_works.json"):
                with open(os.path.join(works_cache_dir, file), 'r', encoding='utf-8') as f:
                    author_works = json.load(f)
                
                # Add works to the works_data dictionary
                for work in author_works:
                    works_data[work["id"]] = work
    
    if not author_data or not works_data:
        print(f"No cached data found for {author_name}")
        return False
    
    # Create files
    unique_works = create_xml_file(author_name, author_data, works_data, author_id_to_label)
    create_author_pair_file(author_name, unique_works)
    create_venue_pair_file(author_name, unique_works)
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract OpenAlex data for HGCN name disambiguation')
    parser.add_argument('--name', type=str, help='Name to disambiguate (e.g., "John Smith")')
    parser.add_argument('--max_authors', type=int, default=30, help='Maximum number of authors to fetch')
    parser.add_argument('--max_works', type=int, default=100, help='Maximum number of works per author')
    parser.add_argument('--use_cache', action='store_true', help='Use cached data if available')
    
    # New arguments for batch processing
    parser.add_argument('--fetch_works_only', action='store_true', help='Only fetch works for a specific author ID')
    parser.add_argument('--author_id', type=str, help='Author ID to fetch works for (use with --fetch_works_only)')
    parser.add_argument('--create_files_only', action='store_true', help='Create XML and pair files from cached data')
    
    args = parser.parse_args()
    
    # Check for required arguments
    if args.fetch_works_only:
        if not args.author_id or not args.name:
            print("Error: --author_id and --name are required with --fetch_works_only")
            sys.exit(1)
        fetch_works_only(args.author_id, args.name, args.max_works)
        sys.exit(0)
    
    if args.create_files_only:
        if not args.name:
            print("Error: --name is required with --create_files_only")
            sys.exit(1)
        success = create_files_from_cache(args.name)
        sys.exit(0 if success else 1)
    
    if not args.name:
        print("Error: --name is required")
        sys.exit(1)
    
    # Check for cached data
    if args.use_cache:
        author_data, works_data, author_id_to_label = load_data_from_json(args.name)
        if author_data and works_data and author_id_to_label:
            # Create files from cached data
            unique_works = create_xml_file(args.name, author_data, works_data, author_id_to_label)
            create_author_pair_file(args.name, unique_works)
            create_venue_pair_file(args.name, unique_works)
            print(f"Data extraction and formatting complete for {args.name} (using cached data)")
            print(f"Found {len(author_data)} authors and {len(unique_works)} unique publications")
            print(f"Run name_disambiguation.py to perform disambiguation")
            sys.exit(0)
    
    # 1. Fetch author data from OpenAlex
    author_data = fetch_author_data(args.name, args.max_authors)
    
    # Create mapping from author ID to label (integer)
    author_id_to_label = {}
    for i, author_id in enumerate(author_data.keys()):
        author_id_to_label[author_id] = str(i)
    
    # 2. Fetch works for each author
    works_data = {}
    for author_id, author in author_data.items():
        author_works = fetch_works_for_author(author_id, args.max_works)
        author["works"] = [w["id"] for w in author_works]
        
        # Add works to global works data
        for work in author_works:
            works_data[work["id"]] = work
    
    # Save data to JSON for future use
    save_data_to_json(args.name, author_data, works_data, author_id_to_label)
    
    # 3. Create XML file
    unique_works = create_xml_file(args.name, author_data, works_data, author_id_to_label)
    
    # 4. Create author pair file
    create_author_pair_file(args.name, unique_works)
    
    # 5. Create venue pair file
    create_venue_pair_file(args.name, unique_works)
    
    print(f"Data extraction and formatting complete for {args.name}")
    print(f"Found {len(author_data)} authors and {len(unique_works)} unique publications")
    print(f"Run name_disambiguation.py to perform disambiguation") 