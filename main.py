import json
import os 
from scrape import scraper, extract_body_content, clean_body_content, wikilinks
from parse import process_content
import time
from tqdm import tqdm
from huggingface_hub import HfApi

BASE_URL = "https://minecraft.wiki/" #you guys can enter the url of the game wiki you want to scrape.
OUTPUT_FILE = "wiki_pages.txt" #you can change this 2 doesnt matter what name
output_json_name = 'dataset.json' #you can change this too
HF_DATASET = False  #set to False to disable upload, or (True, "your_token", "your_dataset_name") to enable
SNAPSHOTS = False  #set to False to disable snapshots, or (True, interval) to enable (e.g., (True, 100))
#TODO:
#test this further for other game wikis
#add better error handling
#add option to scrape multiple wikis at once

def read_links_from_file(filename):
    with open(filename, 'r') as file:
        return [line.strip() for line in file if line.strip()]

def process_link(url):
    try:
        html_content = scraper(url)  
        
        if not html_content:
            return None

        body_content = extract_body_content(html_content)
        cleaned_content = clean_body_content(body_content)
        
        question, answer = process_content([cleaned_content])

        return {
            "url": url,
            "question": question,
            "answer": answer
        }
    
    except Exception as e:
        print(f"Error processing {url}: {str(e)}")
        return None

def save_snapshot(results, snapshot_number):
    if isinstance(SNAPSHOTS, tuple) and SNAPSHOTS[0]:
        snapshot_file = f'snapshot_{snapshot_number}.json'
        with open(snapshot_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Snapshot saved: {snapshot_file}")

def upload_to_huggingface(file_path):
    if isinstance(HF_DATASET, tuple) and HF_DATASET[0]:
        try:
            api = HfApi()
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=output_json_name,
                repo_id=HF_DATASET[2],
                repo_type="dataset",
                token=HF_DATASET[1]
            )
            print(f"Dataset uploaded successfully to Hugging Face Hub: {HF_DATASET[2]}")
        except Exception as e:
            print(f"Error uploading to Hugging Face Hub: {str(e)}")

def main():
    if not os.path.exists(OUTPUT_FILE):
        wikilinks(BASE_URL, OUTPUT_FILE)
    else:
        print(f"Using existing file {OUTPUT_FILE}.")

    links = read_links_from_file(OUTPUT_FILE)
    results = []

    print(f"Processing {len(links)} links...")

    for i, link in enumerate(tqdm(links), 1):
        result = process_link(link)
        if result:
            results.append(result)

        if isinstance(SNAPSHOTS, tuple) and SNAPSHOTS[0] and i % SNAPSHOTS[1] == 0:
            save_snapshot(results, i // SNAPSHOTS[1])

        time.sleep(1)

    with open(output_json_name, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Processed {len(results)} webpages successfully. Results saved to {output_json_name}")

    if isinstance(HF_DATASET, tuple) and HF_DATASET[0]:
        upload_to_huggingface(output_json_name)

if __name__ == "__main__":
    main()
