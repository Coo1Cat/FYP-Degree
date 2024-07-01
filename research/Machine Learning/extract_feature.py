import pandas as pd
import os
from urllib.parse import urlparse, urljoin
import re
import aiohttp
import asyncio
from bs4 import BeautifulSoup
import math
from collections import Counter
from string import ascii_letters
import itertools
import Levenshtein
import random
import pygame

# just to make noise when finished running
def play_mp3(file_path):
    pygame.mixer.init()
    
    # Load and play the MP3 file
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
        
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def load_data(filepath):
    """Load data from a CSV file."""
    data = pd.read_csv(filepath)
    print("Data loaded successfully. Columns in the dataset:", data.columns)
    print(data.head())
    return data

def load_reference_urls(filepath, sample_size=100, random_seed=42):
    """Load reference URLs from a CSV file and sample them."""
    reference_data = pd.read_csv(filepath)
    reference_urls = reference_data['URL'].tolist()
    if len(reference_urls) > sample_size:
        random.seed(random_seed)
        reference_urls = random.sample(reference_urls, sample_size)
    print(f"Reference URLs loaded successfully. Total URLs: {len(reference_urls)}")
    return reference_urls

def is_absolute(url):
    """Check if a URL is absolute."""
    return bool(urlparse(url).netloc)

def add_scheme(url):
    """Add http or https scheme to the URL if missing."""
    parsed_url = urlparse(url)
    if not parsed_url.scheme:
        return 'https://' + url if 'https' in url else 'http://' + url
    return url

def calculate_entropy(s):
    """Calculate entropy of a string."""
    p, lns = Counter(s), float(len(s))
    return -sum(count/lns * math.log(count/lns, 2) for count in p.values())

def lexical_density(s):
    """Calculate lexical density."""
    letters = sum(c.isalpha() for c in s)
    return letters / len(s) if len(s) > 0 else 0

def calculate_similarity_index(url, reference_urls):
    """Calculate similarity index of a URL against a set of reference URLs using Levenshtein distance."""
    distances = [Levenshtein.distance(url, ref_url) for ref_url in reference_urls]
    min_distance = min(distances) if distances else 0
    max_distance = max(len(url), max(len(ref_url) for ref_url in reference_urls))
    similarity_percentage = (1 - min_distance / max_distance) * 100
    return round(similarity_percentage, 2)

def extract_features(data, reference_urls):
    """Extract features from URLs."""
    if 'URL' not in data.columns:
        raise KeyError("The 'URL' column is missing from the dataset.")

    data = data.copy() 
    data['original_url_length'] = data['URL'].apply(len)  # Store original URL length
    data['URL'] = data['URL'].apply(add_scheme)  # Ensure URLs have schemes

    def safe_urlparse(url, attribute):
        try:
            return getattr(urlparse(url), attribute)
        except ValueError:
            return ""

    data['pathDomainRatio'] = data['URL'].apply(lambda x: len(safe_urlparse(x, 'path')) / len(safe_urlparse(x, 'netloc')) if len(safe_urlparse(x, 'netloc')) > 0 else -1)
    data['domain_token_count'] = data['URL'].apply(lambda x: len(safe_urlparse(x, 'netloc').split('.')))
    data['tld'] = data['URL'].apply(lambda x: len(safe_urlparse(x, 'netloc').split('.')[-1]))
    data['domainlength'] = data['URL'].apply(lambda x: len(safe_urlparse(x, 'netloc')))
    data['host_letter_count'] = data['URL'].apply(lambda x: sum(c.isalpha() for c in safe_urlparse(x, 'netloc')))
    data['domainUrlRatio'] = data['URL'].apply(lambda x: len(safe_urlparse(x, 'netloc')) / len(x) if len(x) > 0 else -1)
    data['SymbolCount_Domain'] = data['URL'].apply(lambda x: sum(not c.isalnum() for c in safe_urlparse(x, 'netloc')))
    data['NumberofDotsinURL'] = data['URL'].apply(lambda x: x.count('.'))
    data['LongestPathTokenLength'] = data['URL'].apply(lambda x: max(len(token) for token in safe_urlparse(x, 'path').split('/')) if safe_urlparse(x, 'path') else -1)
    data['urlLen'] = data['URL'].apply(len)
    data['pathLength'] = data['URL'].apply(lambda x: len(safe_urlparse(x, 'path')))
    data['argDomanRatio'] = data['URL'].apply(lambda x: len(safe_urlparse(x, 'query')) / len(safe_urlparse(x, 'netloc')) if len(safe_urlparse(x, 'query')) > 0 and len(safe_urlparse(x, 'netloc')) > 0 else -1)
    data['pathurlRatio'] = data['URL'].apply(lambda x: len(safe_urlparse(x, 'path')) / len(x) if len(x) > 0 else -1)
    data['Directory_LetterCount'] = data['URL'].apply(lambda x: sum(c.isalpha() for c in safe_urlparse(x, 'path')))
    data['subDirLen'] = data['URL'].apply(lambda x: len(safe_urlparse(x, 'path').split('/')[1]) if len(safe_urlparse(x, 'path').split('/')) > 1 else -1)
    data['delimeter_path'] = data['URL'].apply(lambda x: len(safe_urlparse(x, 'path').split('/')))
    data['path_token_count'] = data['URL'].apply(lambda x: len(safe_urlparse(x, 'path').split('/')))
    data['CharacterContinuityRate'] = data['URL'].apply(lambda x: max(len(list(g)) for k, g in itertools.groupby(x)))
    data['Entropy_Domain'] = data['URL'].apply(lambda x: calculate_entropy(safe_urlparse(x, 'netloc')))
    data['charcompvowels'] = data['URL'].apply(lambda x: sum(c in 'aeiou' for c in x.lower()))
    data['ldl_url'] = data['URL'].apply(lexical_density)
    data['SymbolCount_Directoryname'] = data['URL'].apply(lambda x: sum(not c.isalnum() for c in safe_urlparse(x, 'path')))
    data['longdomaintokenlen'] = data['URL'].apply(lambda x: max(len(token) for token in safe_urlparse(x, 'netloc').split('.')))
    data['SymbolCount_URL'] = data['URL'].apply(lambda x: sum(not c.isalnum() for c in x))
    data['avgdomaintokenlen'] = data['URL'].apply(lambda x: sum(len(token) for token in safe_urlparse(x, 'netloc').split('.')) / len(safe_urlparse(x, 'netloc').split('.')) if len(safe_urlparse(x, 'netloc').split('.')) > 0 else -1)
    data['avgpathtokenlen'] = data['URL'].apply(lambda x: sum(len(token) for token in safe_urlparse(x, 'path').split('/')) / len(safe_urlparse(x, 'path').split('/')) if len(safe_urlparse(x, 'path').split('/')) > 0 else -1)
    data['ldl_path'] = data['URL'].apply(lambda x: lexical_density(safe_urlparse(x, 'path')))
    data['Entropy_Filename'] = data['URL'].apply(lambda x: calculate_entropy(os.path.basename(safe_urlparse(x, 'path'))))
    data['Extension_LetterCount'] = data['URL'].apply(lambda x: sum(c.isalpha() for c in os.path.splitext(safe_urlparse(x, 'path'))[1]))
    data['sub-Directory_LongestWordLength'] = data['URL'].apply(lambda x: max(len(word) for word in safe_urlparse(x, 'path').split('/')[1].split('.')) if len(safe_urlparse(x, 'path').split('/')) > 1 else -1)
    
    # Calculate URLSimilarityIndex as a percentage
    data['URLSimilarityIndex'] = data['URL'].apply(lambda x: calculate_similarity_index(x, reference_urls))
    
    # Features that require fetching and parsing the HTML content
    async def fetch(session, url, retries=1, retry_delay=1):
        for attempt in range(retries):
            try:
                async with session.get(url, timeout=3) as response:
                    if response.status == 200:
                        try:
                            return await response.text()
                        except UnicodeDecodeError:
                            return await response.text(encoding='latin-1')
                    else:
                        print(f"Failed to fetch {url}: HTTP {response.status}")
                        return None
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                print(f"Attempt {attempt+1} to fetch {url} failed with error: {e}")
                await asyncio.sleep(retry_delay)
        print(f"All attempts to fetch {url} failed.")
        return None

    async def scrape_url(session, i, url):
        print(f"Extracting features for URL {i+1}: {url}")
        try:
            html = await fetch(session, url)
            if html:
                soup = BeautifulSoup(html, 'html.parser')
                LineOfCode = html.count('\n')
                NoOfExternalRef = sum(1 for link in soup.find_all('a', href=True) if urlparse(link['href']).netloc and urlparse(link['href']).netloc != urlparse(url).netloc)
                NoOfSelfRef = sum(1 for link in soup.find_all('a', href=True) if urlparse(link['href']).netloc and urlparse(link['href']).netloc == urlparse(url).netloc)
                NoOfImage = len(soup.find_all('img'))
                NoOfJS = len(soup.find_all('script', src=True))
                NoOfCSS = len(soup.find_all('link', rel="stylesheet"))
                HasDescription = int(bool(soup.find('meta', attrs={"name": "description"})))
                HasSocialNet = int(any(social in str(soup) for social in ['facebook.com', 'twitter.com']))
                NoOfOtherSpecialCharsInURL = sum(not c.isalnum() for c in url)
                IsHTTPS = int(url.startswith('https://'))
                HasCopyrightInfo = int('©' in html or 'copyright' in html.lower())
                DegitRatioInURL = sum(c.isdigit() for c in url) / len(url) if len(url) > 0 else 0
                HasSubmitButton = int(bool(soup.find('input', attrs={"type": "submit"}) or soup.find('button', attrs={"type": "submit"})))
                
                return {
                    'URL': url,
                    'LineOfCode': LineOfCode,
                    'NoOfExternalRef': NoOfExternalRef,
                    'NoOfSelfRef': NoOfSelfRef,
                    'NoOfImage': NoOfImage,
                    'NoOfJS': NoOfJS,
                    'NoOfCSS': NoOfCSS,
                    'HasDescription': HasDescription,
                    'HasSocialNet': HasSocialNet,
                    'NoOfOtherSpecialCharsInURL': NoOfOtherSpecialCharsInURL,
                    'IsHTTPS': IsHTTPS,
                    'HasCopyrightInfo': HasCopyrightInfo,
                    'DegitRatioInURL': DegitRatioInURL,
                    'HasSubmitButton': HasSubmitButton
                }
            else:
                return None
        except Exception as e:
            print(f"Error processing URL {url}: {e}")
            return None

    async def extract_all_features(data):
        async with aiohttp.ClientSession() as session:
            tasks = [scrape_url(session, i, url) for i, url in enumerate(data['URL'])]
            results = await asyncio.gather(*tasks)
            extracted_data = [result for result in results if result is not None and 'LineOfCode' in result]
            return pd.DataFrame(extracted_data)

    loop = asyncio.get_event_loop()
    extracted_data = loop.run_until_complete(extract_all_features(data))

    if not extracted_data.empty:
        data = data.merge(extracted_data, on='URL', how='inner')
    
    # default values for missing columns
    required_columns = [
        'LineOfCode', 'NoOfExternalRef', 'NoOfSelfRef', 'NoOfImage', 'NoOfJS', 'NoOfCSS',
        'HasDescription', 'HasSocialNet', 'NoOfOtherSpecialCharsInURL', 'IsHTTPS',
        'HasCopyrightInfo', 'DegitRatioInURL', 'HasSubmitButton'
    ]
    default_values = {
        'LineOfCode': -1, 'NoOfExternalRef': -1, 'NoOfSelfRef': -1, 'NoOfImage': -1,
        'NoOfJS': -1, 'NoOfCSS': -1, 'HasDescription': -1, 'HasSocialNet': -1,
        'NoOfOtherSpecialCharsInURL': -1, 'IsHTTPS': -1, 'HasCopyrightInfo': -1,
        'DegitRatioInURL': -1.0, 'HasSubmitButton': -1
    }

    for column in required_columns:
        if column not in data.columns:
            data[column] = default_values[column]

    return data

def process_chunk(chunk, chunk_idx, output_folder, reference_urls):
    """Process a single chunk of the dataset."""
    data = extract_features(chunk, reference_urls)
    output_filepath = os.path.join(output_folder, f'url_features_chunk_{chunk_idx}.csv')
    print(f"Data after feature extraction for chunk {chunk_idx}:")
    print(data.head())
    data.to_csv(output_filepath, index=False)
    print(f"Feature extraction completed for chunk {chunk_idx}. Features saved to '{output_filepath}'")

def main(input_filepath, output_folder, reference_filepath, start_chunk=0, chunk_size=100):
    """Main function to perform feature extraction in chunks."""
    data = load_data(input_filepath)
    reference_urls = load_reference_urls(reference_filepath)

    os.makedirs(output_folder, exist_ok=True)

    start_index = start_chunk * chunk_size

    for chunk_idx, start in enumerate(range(start_index, len(data), chunk_size)):
        data_chunk = data.iloc[start:start + chunk_size]
        process_chunk(data_chunk, chunk_idx + start_chunk, output_folder, reference_urls)
    
    # Play the MP3 file after all chunks are processed
    mp3_file = os.path.join(os.path.dirname(__file__), 'sound', 'Achievement_Sound_Effect.mp3')
    play_mp3(mp3_file)

if __name__ == '__main__':
    current_directory = os.path.dirname(__file__)
    input_filepath = os.path.join(current_directory, 'temp', 'ReferenceURLs.csv')
    output_folder = os.path.join(current_directory, 'processed_data')
    reference_filepath = os.path.join(current_directory, 'temp', 'ReferenceURLs.csv')
    start_chunk = 0  # Start at which chunk?
    main(input_filepath, output_folder, reference_filepath, start_chunk)
