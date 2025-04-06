#!/usr/bin/env python3
import os
import json
import argparse
import subprocess
import sys
from tqdm import tqdm
import time
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_disambiguation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('batch_disambiguation')

def run_command(cmd, verbose=False):
    """Run a shell command and return the output."""
    if verbose:
        logger.info(f"Running command: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Command failed with error: {result.stderr}")
        return False, result.stderr
    
    if verbose:
        logger.info(f"Command output: {result.stdout}")
    
    return True, result.stdout

def process_single_name(name, max_authors=30, max_works=100, verbose=False):
    """Process a single name through the entire disambiguation workflow."""
    logger.info(f"Processing name: {name}")
    
    # Step 1: Fetch author data from OpenAlex and create HGCN input files
    cmd = f"python openAlex_to_HGCN.py --name \"{name}\" --max_authors {max_authors} --max_works {max_works}"
    success, output = run_command(cmd, verbose)
    
    if not success:
        logger.error(f"Failed to fetch data for {name}")
        return False
    
    # Step 2: Run the name disambiguation
    cmd = f"python run_disambiguation.py --name \"{name}\""
    success, output = run_command(cmd, verbose)
    
    if not success:
        logger.error(f"Failed to run disambiguation for {name}")
        return False
    
    logger.info(f"Successfully processed {name}")
    return True

def process_from_file(filename, max_authors=30, max_works=100, verbose=False):
    """Process names from a text file."""
    if not os.path.exists(filename):
        logger.error(f"File not found: {filename}")
        return False
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            names = [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Error reading file {filename}: {e}")
        return False
    
    logger.info(f"Processing {len(names)} names from {filename}")
    
    success_count = 0
    with tqdm(total=len(names), desc="Processing names") as pbar:
        for name in names:
            result = process_single_name(name, max_authors, max_works, verbose)
            if result:
                success_count += 1
            pbar.update(1)
            # Add a small delay to avoid overwhelming the API
            time.sleep(1)
    
    logger.info(f"Completed processing {len(names)} names. {success_count} successful, {len(names) - success_count} failed.")
    return success_count > 0

def process_from_json(json_file, name_field, max_authors=30, max_works=100, verbose=False):
    """Process names from a JSON file."""
    if not os.path.exists(json_file):
        logger.error(f"JSON file not found: {json_file}")
        return False
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error reading JSON file {json_file}: {e}")
        return False
    
    # Extract names from JSON data
    names = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and name_field in item:
                names.append(item[name_field])
            elif isinstance(item, str):
                names.append(item)
    elif isinstance(data, dict):
        if name_field in data:
            if isinstance(data[name_field], list):
                names = data[name_field]
            else:
                names = [data[name_field]]
    
    if not names:
        logger.error(f"No names found in JSON file using field '{name_field}'")
        return False
    
    logger.info(f"Processing {len(names)} names from JSON file {json_file}")
    
    success_count = 0
    with tqdm(total=len(names), desc="Processing names") as pbar:
        for name in names:
            result = process_single_name(name, max_authors, max_works, verbose)
            if result:
                success_count += 1
            pbar.update(1)
            # Add a small delay to avoid overwhelming the API
            time.sleep(1)
    
    logger.info(f"Completed processing {len(names)} names. {success_count} successful, {len(names) - success_count} failed.")
    return success_count > 0

def process_existing_openAlex_data(directory='openAlex_scraper', max_authors=30, max_works=100, verbose=False):
    """Process existing data from the OpenAlex scraper directory."""
    if not os.path.exists(directory):
        logger.error(f"Directory not found: {directory}")
        return False
    
    # Try to find existing data sources
    sources = []
    
    # Check for the JSON data file
    json_file = os.path.join(directory, 'asci_aap_dataJSON.json')
    if os.path.exists(json_file):
        sources.append(('json', json_file))
    
    # Check for OpenAlex IDs
    ids_file = os.path.join(directory, 'openAlex_ids.json')
    if os.path.exists(ids_file):
        sources.append(('ids', ids_file))
    
    if not sources:
        logger.error(f"No usable data sources found in {directory}")
        return False
    
    # Process each source
    for source_type, source_path in sources:
        logger.info(f"Processing source: {source_path}")
        
        if source_type == 'json':
            # Try to identify the name field by examining the file
            try:
                with open(source_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Check if it's a list of items
                if isinstance(data, list) and len(data) > 0:
                    sample = data[0]
                    # Look for common name fields
                    for field in ['name', 'author_name', 'display_name', 'full_name']:
                        if isinstance(sample, dict) and field in sample:
                            logger.info(f"Found name field: {field}")
                            process_from_json(source_path, field, max_authors, max_works, verbose)
                            break
                    else:
                        logger.warning(f"Could not identify name field in {source_path}")
            except Exception as e:
                logger.error(f"Error processing JSON file {source_path}: {e}")
        
        elif source_type == 'ids':
            try:
                with open(source_path, 'r', encoding='utf-8') as f:
                    ids_data = json.load(f)
                
                # Check if this is a dictionary mapping names to IDs
                if isinstance(ids_data, dict):
                    names = list(ids_data.keys())
                    logger.info(f"Found {len(names)} names in {source_path}")
                    
                    success_count = 0
                    with tqdm(total=len(names), desc="Processing names") as pbar:
                        for name in names:
                            result = process_single_name(name, max_authors, max_works, verbose)
                            if result:
                                success_count += 1
                            pbar.update(1)
                            # Add a small delay to avoid overwhelming the API
                            time.sleep(1)
                    
                    logger.info(f"Completed processing {len(names)} names. {success_count} successful, {len(names) - success_count} failed.")
            except Exception as e:
                logger.error(f"Error processing IDs file {source_path}: {e}")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch process names for disambiguation')
    
    # Input sources
    input_group = parser.add_argument_group('Input Sources')
    input_group.add_argument('--names', nargs='+', help='List of names to process')
    input_group.add_argument('--file', type=str, help='Text file containing names (one per line)')
    input_group.add_argument('--json', type=str, help='JSON file containing names')
    input_group.add_argument('--json_field', type=str, default='name', help='Field name containing the name in the JSON file')
    input_group.add_argument('--use_existing', action='store_true', help='Use existing data from the OpenAlex scraper directory')
    
    # Processing options
    options_group = parser.add_argument_group('Processing Options')
    options_group.add_argument('--max_authors', type=int, default=30, help='Maximum number of authors to fetch per name')
    options_group.add_argument('--max_works', type=int, default=100, help='Maximum number of works per author')
    options_group.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Check if at least one input source is provided
    if not (args.names or args.file or args.json or args.use_existing):
        parser.print_help()
        print("\nError: At least one input source must be provided")
        sys.exit(1)
    
    # Process names from command line
    if args.names:
        logger.info(f"Processing {len(args.names)} names from command line")
        for name in args.names:
            process_single_name(name, args.max_authors, args.max_works, args.verbose)
    
    # Process names from text file
    if args.file:
        process_from_file(args.file, args.max_authors, args.max_works, args.verbose)
    
    # Process names from JSON file
    if args.json:
        process_from_json(args.json, args.json_field, args.max_authors, args.max_works, args.verbose)
    
    # Process existing OpenAlex data
    if args.use_existing:
        process_existing_openAlex_data('openAlex_scraper', args.max_authors, args.max_works, args.verbose)
    
    logger.info("Batch processing complete") 