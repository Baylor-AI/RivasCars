import csv
import re
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

def clean_and_hash(text):
    """Remove punctuation from text and return its MD5 hash."""
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    cleaned_text = cleaned_text.lower()
    hashed_text = hashlib.md5(cleaned_text.encode('utf-8')).hexdigest()
    return hashed_text

def process_chunk(chunk, index_title, index_body):
    """Process a chunk of CSV rows, identifying unique and duplicate entries."""
    seen_hashes_chunk = {}  # Local seen hashes for this chunk
    duplicate_groups_chunk = {}  # Duplicate groups found in this chunk

    for row in chunk:
        text_to_hash = row[index_body].strip() if row[index_body].strip() else row[index_title].strip()
        post_hash = clean_and_hash(text_to_hash)

        if post_hash in seen_hashes_chunk:
            # Append row to the existing duplicate group for this hash
            duplicate_groups_chunk[post_hash].append(row)
        else:
            # Add the row as unique and keep a reference to it in case of future duplicates
            seen_hashes_chunk[post_hash] = row
            duplicate_groups_chunk[post_hash] = [row]  # Start a group with the original row

    return duplicate_groups_chunk

def write_output_files(duplicate_groups, dedup_file_path, duplicates_file_path):
    """Write the deduplicated entries and the duplicate groups to their respective files."""
    with open(dedup_file_path, 'w', newline='', encoding='utf-8') as dedup_file, \
         open(duplicates_file_path, 'w', newline='', encoding='utf-8') as duplicates_file:
        dedup_writer = csv.writer(dedup_file)
        duplicates_writer = csv.writer(duplicates_file)

        for post_hash, group in duplicate_groups.items():
            if len(group) > 1:  # If there's more than one entry, it's a duplicate group
                for row in group:
                    duplicates_writer.writerow(row)
                duplicates_writer.writerow([])  # Blank row after a group of duplicates
            else:  # This is a unique entry
                dedup_writer.writerow(group[0])

def process_csv(file_path, dedup_file_path, duplicates_file_path, index_title, index_body):
    """Read the CSV, process it in chunks using threading, and write the results."""
    chunks = []  # This will store the chunks of CSV rows
    CHUNK_SIZE = 1000  # Define how many rows each chunk should have

    # Read the CSV and divide it into chunks
    with open(file_path, 'r', encoding='utf-8', newline='') as csvfile:
        reader = csv.reader(csvfile)
        chunk = []
        for row in reader:
            chunk.append(row)
            if len(chunk) == CHUNK_SIZE:
                chunks.append(chunk)
                chunk = []
        if chunk:  # Add any remaining rows as a final chunk
            chunks.append(chunk)

    # Process each chunk in parallel
    all_duplicate_groups = {}  # Initialize an empty dict to collect all duplicate groups across chunks
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_chunk, chunk, index_title, index_body) for chunk in chunks]
        for future in as_completed(futures):
            duplicate_groups_chunk = future.result()
            for post_hash, group in duplicate_groups_chunk.items():
                if post_hash in all_duplicate_groups:
                    # Extend the main list if this hash was already encountered in another chunk
                    all_duplicate_groups[post_hash].extend(group[1:])  # Skip the original since it's already there
                else:
                    all_duplicate_groups[post_hash] = group

    # Write the deduplicated entries and the duplicate groups to their respective files
    write_output_files(all_duplicate_groups, dedup_file_path, duplicates_file_path)

def main():
    """Main function to set file paths and initiate processing."""
    input_file_path = 'processed_data_craigslist_v2.csv'
    dedup_file_path = 'dedup.csv'
    duplicates_file_path = 'duplicates.csv'
    index_title = 4  # Column index for the post title
    index_body = 6  # Column index for the post body

    process_csv(input_file_path, dedup_file_path, duplicates_file_path, index_title, index_body)
    print(f"Finished processing. Deduplicated version stored in '{dedup_file_path}'.")
    print(f"Duplicates grouped and stored in '{duplicates_file_path}'.")

if __name__ == "__main__":
    main()
