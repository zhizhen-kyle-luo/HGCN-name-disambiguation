#!/usr/bin/env python3
import os
import argparse
import subprocess
import sys
import json
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('run_disambiguation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('run_disambiguation')

def ensure_directory(path):
    """Ensure that a directory exists."""
    os.makedirs(path, exist_ok=True)

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

def run_disambiguation(author_name, verbose=False):
    """Run the HGCN name disambiguation for a specific author."""
    logger.info(f"Running disambiguation for {author_name}")
    
    # Check if the required input files exist
    xml_path = os.path.join("raw-data-temp", f"{author_name}.xml")
    if not os.path.exists(xml_path):
        logger.error(f"XML file not found: {xml_path}")
        return False
    
    # Run name_disambiguation.py directly
    cmd = "python name_disambiguation.py"
    success, output = run_command(cmd, verbose)
    
    if not success:
        logger.error("Failed to run name disambiguation")
        return False
    
    # Check if results were generated
    results_dir = "result"
    if not os.path.exists(results_dir):
        logger.error("Results directory not found")
        return False
    
    # Process and organize results
    ensure_directory(os.path.join("results", "author_clusters"))
    
    # Find the clustering results
    cluster_files = [f for f in os.listdir(results_dir) if f.endswith("_clusters.json")]
    
    if not cluster_files:
        logger.error("No cluster results found")
        return False
    
    # Copy and rename the results file
    for cluster_file in cluster_files:
        source_path = os.path.join(results_dir, cluster_file)
        dest_path = os.path.join("results", "author_clusters", f"{author_name}_clusters.json")
        
        # Read the source file
        with open(source_path, 'r', encoding='utf-8') as f:
            clusters_data = json.load(f)
        
        # Write to the destination file
        with open(dest_path, 'w', encoding='utf-8') as f:
            json.dump(clusters_data, f, indent=2)
        
        logger.info(f"Results saved to: {dest_path}")
    
    return True

def summarize_results(author_name):
    """Summarize the disambiguation results."""
    results_path = os.path.join("results", "author_clusters", f"{author_name}_clusters.json")
    
    if not os.path.exists(results_path):
        logger.error(f"Results file not found: {results_path}")
        return False
    
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            clusters = json.load(f)
        
        # Print summary
        logger.info(f"\nDisambiguation results for {author_name}:")
        logger.info(f"Found {len(clusters[author_name])} distinct people with this name")
        
        for person_id, openAlex_ids in clusters[author_name].items():
            logger.info(f"Person {person_id}: has {len(openAlex_ids)} OpenAlex IDs: {', '.join(openAlex_ids)}")
        
        return True
    except Exception as e:
        logger.error(f"Error summarizing results: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run HGCN name disambiguation')
    parser.add_argument('--name', type=str, required=True, help='Name to disambiguate (e.g., "John Smith")')
    parser.add_argument('--verbose', action='store_true', help='Show detailed output')
    
    args = parser.parse_args()
    
    # Run the disambiguation
    success = run_disambiguation(args.name, args.verbose)
    
    if success:
        # Summarize the results
        summarize_results(args.name)
        logger.info("Disambiguation completed successfully")
        sys.exit(0)
    else:
        logger.error("Disambiguation failed")
        sys.exit(1) 