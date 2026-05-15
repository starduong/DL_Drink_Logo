import os
import csv
import time
import hashlib
import argparse
import logging
from io import BytesIO
from datetime import datetime
from urllib.parse import quote_plus
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Set

import requests
from PIL import Image
import imagehash
from tqdm import tqdm
import random
import shutil
import json

# Fix Unicode encoding for Windows
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# Third-party imports
try:
    from ddgs import DDGS

    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    try:
        from duckduckgo_search import DDGS

        DDGS_AVAILABLE = True
    except ImportError:
        DDGS_AVAILABLE = False

# Selenium imports for Google search
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options

    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False


# ---------------------- CONFIGURATION ----------------------
class Config:
    # Download settings
    TARGET_PER_BRAND = 1000
    MIN_SIDE = 128
    PHASH_THRESHOLD = 6
    MAX_WORKERS = 10
    REQUEST_TIMEOUT = 45
    RATE_LIMIT_DELAY = 1.5

    # Sources configuration (priority order)
    ENABLED_SOURCES = {
        "duckduckgo": True,
        "selenium": True,
    }

    # Paths
    OUTPUT_ROOT = "logo_dataset_drink_v5"
    IMAGES_DIR = os.path.join(OUTPUT_ROOT, "images")
    METADATA_CSV = os.path.join(OUTPUT_ROOT, "metadata.csv")
    LOG_FILE = os.path.join(OUTPUT_ROOT, "download.log")


BRANDS_QUERIES = {
    "coca_cola": [
        "coca cola logo",
        "coca cola official logo",
        "coca cola logo png",
        "coca cola logo transparent",
        "coca cola logo svg",
        "coca cola brand logo",
        "coca cola red white logo",
        "coca cola 3d logo",
        "coca cola logo 3d",
        "coca cola minimal logo",
        "coca cola flat logo",
        "coca cola flat design logo",
        "Coca-Cola logo centered high resolution isolated on white background",
        "Official Coca-Cola red and white ribbon logo close up",
        "Coca-Cola logo dominant large occupying over 60% of frame",
        "Coca-Cola red disc logo transparent PNG centered",
        "Coca-Cola logo 4K pure flat design minimal background",
        "Coca-Cola official logo 2024-2025 isolated clean",
        "Coca-Cola logo metallic silver and red centered close up",
        "Coca-Cola classic red and white script logo high detail",
        "Coca-Cola logo flat clean large centered on plain background",
        "Coca-Cola logo only no bottle no can isolated dominant",
        "Coca-Cola logo high resolution centered pure brand mark",
        "Coca-Cola contour bottle logo isolated on white background",
        "Coca-Cola logo PNG transparent large dominant in frame",
        "Coca-Cola logo with standard colors white background minimal",
        "Coca-Cola logo silver red metallic close up isolated",
        "Coca-Cola logo 2025 version clean large centered",
    ],
    "pepsi": [
        "pepsi logo",
        "pepsi official logo",
        "pepsi logo png",
        "pepsi logo transparent",
        "pepsi logo svg",
        "pepsi brand logo",
        "pepsi globe logo",
        "pepsi 3d logo",
        "pepsi logo 3d",
        "pepsi minimal logo",
        "pepsi flat logo",
        "pepsi flat design logo",
        "Pepsi globe logo centered high resolution isolated",
        "Pepsi logo only white background dominant clean large",
        "Pepsi blue red white globe logo close up 2024-2025",
        "Pepsi logo red white blue PNG transparent large centered",
        "Pepsi logo 4K minimal plain background centered",
        "Pepsi official globe logo isolated dominant >60% frame",
        "Pepsi globe metallic 3D centered clean on white",
        "Pepsi logo no bottle pure globe large standard colors",
        "Pepsi logo dominant occupying over 60% of frame high quality",
        "Pepsi logo flat vector style large centered plain background",
        "Pepsi logo on plain white background PNG isolated",
        "Pepsi globe only no text close up large centered",
        "Pepsi logo 2024-2025 version clean dominant centered",
        "Pepsi logo high resolution centered transparent background",
        "Pepsi logo blue silver metallic dominant isolated",
        "Pepsi globe logo clean large white background centered",
    ],
}


# ---------------------- LOGGING SETUP ----------------------
def setup_logging():
    os.makedirs(Config.OUTPUT_ROOT, exist_ok=True)

    # Fix Unicode logging for Windows
    try:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(Config.LOG_FILE, encoding="utf-8"),
                logging.StreamHandler(sys.stdout),  # Use stdout with UTF-8
            ],
        )
    except Exception as e:
        # Fallback logging without Unicode
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(Config.LOG_FILE), logging.StreamHandler()],
        )


# ---------------------- IMAGE PROCESSING ----------------------
class ImageProcessor:
    @staticmethod
    def download_image(url: str, timeout: int = 30) -> Image.Image:
        """Download and validate image"""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=timeout, stream=True)
            response.raise_for_status()

            # Check content type
            content_type = response.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                return None

            img = Image.open(BytesIO(response.content)).convert("RGB")

            # Validate image is not corrupted
            img.verify()
            img = Image.open(BytesIO(response.content)).convert("RGB")

            return img
        except Exception as e:
            logging.debug(f"Download failed for {url}: {str(e)}")
            return None

    @staticmethod
    def calculate_phash(img: Image.Image) -> imagehash.ImageHash:
        """Calculate perceptual hash for duplicate detection"""
        try:
            return imagehash.phash(img)
        except Exception:
            return None

    @staticmethod
    def is_duplicate(
        hashes: Set,
        new_hash: imagehash.ImageHash,
        threshold: int = Config.PHASH_THRESHOLD,
    ) -> bool:
        """Check if image is duplicate"""
        if new_hash is None:
            return False
        for existing_hash in hashes:
            if new_hash - existing_hash <= threshold:
                return True
        return False

    @staticmethod
    def save_image_jpeg(img: Image.Image, path: str, quality: int = 85):
        """Save image as JPEG"""
        img.save(path, "JPEG", quality=quality, optimize=True)


# ---------------------- SEARCH SOURCES ----------------------
class SearchSources:
    def __init__(self):
        self.ddgs = DDGS() if DDGS_AVAILABLE else None
        self.selenium_driver = None
        self.setup_selenium()

    def setup_selenium(self):
        """Setup Selenium WebDriver for Google search"""
        if not SELENIUM_AVAILABLE or not Config.ENABLED_SOURCES.get("selenium"):
            return

        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless=new")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument(
                "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )

            self.selenium_driver = webdriver.Chrome(options=chrome_options)
            logging.info("Selenium WebDriver initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize Selenium: {str(e)}")
            self.selenium_driver = None

    def duckduckgo_search(self, query: str, max_results: int = 50) -> List[Dict]:
        """Search using DuckDuckGo (free, no API key)"""
        if not Config.ENABLED_SOURCES.get("duckduckgo") or not DDGS_AVAILABLE:
            return []

        try:
            results = []
            # Add random delay to avoid rate limiting
            time.sleep(random.uniform(1.0, 2.0))

            for result in self.ddgs.images(query, max_results=max_results):
                if result.get("image"):
                    results.append(
                        {
                            "url": result["image"],
                            "title": result.get("title", ""),
                            "source": "duckduckgo",
                            "query": query,
                        }
                    )

                    # Break early if we have enough results
                    if len(results) >= max_results:
                        break

            logging.info(f"DDG found {len(results)} results for '{query}'")
            return results
        except Exception as e:
            logging.error(f"DuckDuckGo search error for '{query}': {str(e)}")
            # Wait longer on error
            time.sleep(5.0)
            return []

    def selenium_google_search(self, query: str, max_results: int = 30) -> List[Dict]:
        """Search using Selenium to crawl Google Images"""
        if not Config.ENABLED_SOURCES.get("selenium") or not self.selenium_driver:
            return []

        try:
            results = []
            # Encode query properly for URL
            encoded_query = quote_plus(query)
            url = f"https://www.google.com/search?q={encoded_query}&tbm=isch"

            logging.info(f"Selenium navigating to Google Images")
            self.selenium_driver.get(url)
            time.sleep(3)  # Wait for page to load

            # Try multiple selectors for images
            selectors = [
                "img.Q4LuWd",  # Primary selector
                "img.rg_i",  # Alternative selector
                "img.yWs4tf",  # Another alternative
                "img[alt*='Image result']",  # Generic image selector
            ]

            images = []
            for selector in selectors:
                try:
                    found_images = self.selenium_driver.find_elements(
                        By.CSS_SELECTOR, selector
                    )
                    if found_images:
                        images.extend(found_images)
                        break
                except:
                    continue

            # If no images found with CSS selectors, try by XPath
            if not images:
                try:
                    images = self.selenium_driver.find_elements(
                        By.XPATH, "//img[contains(@src, 'http')]"
                    )
                except:
                    pass

            logging.info(f"Selenium found {len(images)} image elements")

            for i, img in enumerate(images[:max_results]):
                try:
                    src = img.get_attribute("src")
                    if src and src.startswith("http") and not src.startswith("data:"):
                        results.append(
                            {
                                "url": src,
                                "title": f"{query} result {i+1}",
                                "source": "selenium",
                                "query": query,
                            }
                        )
                except Exception as e:
                    logging.debug(f"Failed to process image {i+1}: {str(e)}")
                    continue

            logging.info(f"Selenium extracted {len(results)} results")
            return results

        except Exception as e:
            logging.error(f"Selenium Google search error: {str(e)}")
            return []

    def close_selenium(self):
        """Close Selenium WebDriver"""
        if self.selenium_driver:
            self.selenium_driver.quit()
            logging.info("Selenium WebDriver closed")


# ---------------------- MAIN DOWNLOADER CLASS ----------------------
class AdvancedLogoDownloader:
    def __init__(self):
        self.setup_directories()
        self.image_processor = ImageProcessor()
        self.search_sources = SearchSources()
        self.metadata_file = None
        self.metadata_writer = None
        self.setup_metadata()

        # Tracking state
        self.downloaded_urls = set()
        self.seen_hashes = set()
        self.brand_counts = {}

    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs(Config.IMAGES_DIR, exist_ok=True)
        for brand in BRANDS_QUERIES.keys():
            os.makedirs(os.path.join(Config.IMAGES_DIR, brand), exist_ok=True)

    def setup_metadata(self):
        """Initialize metadata CSV file"""
        file_exists = os.path.exists(Config.METADATA_CSV)
        self.metadata_file = open(
            Config.METADATA_CSV, "a", newline="", encoding="utf-8"
        )
        self.metadata_writer = csv.writer(self.metadata_file)

        if not file_exists:
            self.metadata_writer.writerow(
                [
                    "filename",
                    "brand",
                    "source_url",
                    "source_api",
                    "download_date",
                    "width",
                    "height",
                    "query",
                    "phash",
                ]
            )

    def close_metadata(self):
        """Close metadata file and Selenium"""
        if self.metadata_file:
            self.metadata_file.close()
        self.search_sources.close_selenium()

    def get_current_count(self, brand: str) -> int:
        """Get number of already downloaded images for a brand"""
        if brand not in self.brand_counts:
            brand_dir = os.path.join(Config.IMAGES_DIR, brand)
            if os.path.exists(brand_dir):
                count = len(
                    [
                        f
                        for f in os.listdir(brand_dir)
                        if f.lower().endswith((".jpg", ".jpeg"))
                    ]
                )
                self.brand_counts[brand] = count
            else:
                self.brand_counts[brand] = 0
        return self.brand_counts[brand]

    def search_all_sources(self, query: str) -> List[Dict]:
        """Search all enabled sources for a query"""
        all_results = []

        # Shuffle source order to distribute load
        sources = []
        if Config.ENABLED_SOURCES.get("duckduckgo"):
            sources.append("duckduckgo")
        if Config.ENABLED_SOURCES.get("selenium"):
            sources.append("selenium")

        random.shuffle(sources)

        for source in sources:
            try:
                if source == "duckduckgo":
                    results = self.search_sources.duckduckgo_search(
                        query, max_results=30
                    )
                elif source == "selenium":
                    results = self.search_sources.selenium_google_search(
                        query, max_results=20
                    )
                else:
                    results = []

                all_results.extend(results)

                # Variable delay between sources
                time.sleep(random.uniform(1.5, 3.0))

            except Exception as e:
                logging.error(f"Error searching {source}: {str(e)}")
                time.sleep(5.0)
                continue

        return all_results

    def process_and_save_image(self, result: Dict, brand: str) -> bool:
        """Download, validate, and save an image"""
        if result["url"] in self.downloaded_urls:
            return False

        # Download image
        img = self.image_processor.download_image(result["url"])
        if img is None:
            return False

        # Validate size
        w, h = img.size
        if min(w, h) < Config.MIN_SIDE:
            logging.debug(f"Image too small: {w}x{h}")
            return False

        # Check duplicates
        phash = self.image_processor.calculate_phash(img)
        if self.image_processor.is_duplicate(self.seen_hashes, phash):
            logging.debug("Duplicate image detected")
            return False

        # Save image
        current_count = self.get_current_count(brand)
        filename = f"{brand}_{current_count + 1:06d}.jpg"
        filepath = os.path.join(Config.IMAGES_DIR, brand, filename)

        try:
            self.image_processor.save_image_jpeg(img, filepath)

            # Update tracking
            self.downloaded_urls.add(result["url"])
            if phash:
                self.seen_hashes.add(phash)
            self.brand_counts[brand] = current_count + 1

            # Write metadata
            self.metadata_writer.writerow(
                [
                    os.path.join(brand, filename),
                    brand,
                    result["url"],
                    result["source"],
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    w,
                    h,
                    result["query"],
                    str(phash) if phash else "",
                ]
            )
            self.metadata_file.flush()

            # Safe logging without Unicode characters
            logging.info(f"Saved {filename} from {result['source']}")
            return True

        except Exception as e:
            logging.error(f"Failed to save image: {str(e)}")
            # Remove file if it was partially created
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass
            return False

    def download_brand(self, brand: str) -> int:
        """Download images for a single brand"""
        logging.info(f"Starting download for brand: {brand}")

        initial_count = self.get_current_count(brand)
        target_count = Config.TARGET_PER_BRAND
        needed = max(0, target_count - initial_count)

        if needed == 0:
            logging.info(f"Brand {brand} already has {initial_count} images")
            return 0

        queries = BRANDS_QUERIES[brand]
        total_downloaded = 0

        with tqdm(total=needed, desc=f"{brand:15}") as pbar:
            for query_idx, query in enumerate(queries):
                if total_downloaded >= needed:
                    break

                # Longer delay between queries
                if query_idx > 0:
                    time.sleep(random.uniform(3.0, 6.0))

                # Safe logging without printing the query (to avoid Unicode issues)
                logging.info(
                    f"Searching query {query_idx+1}/{len(queries)} for {brand}"
                )

                # Search all sources for this query
                results = self.search_all_sources(query)
                random.shuffle(results)  # Mix results from different sources

                # Process results
                for result_idx, result in enumerate(results):
                    if total_downloaded >= needed:
                        break

                    # Small delay between image processing
                    if result_idx > 0:
                        time.sleep(random.uniform(0.5, 1.5))

                    if self.process_and_save_image(result, brand):
                        total_downloaded += 1
                        pbar.update(1)

                        # Log progress every 10 images
                        if total_downloaded % 10 == 0:
                            logging.info(
                                f"Downloaded {total_downloaded}/{needed} for {brand}"
                            )

        final_count = self.get_current_count(brand)
        logging.info(
            f"Completed {brand}: {final_count - initial_count} new images "
            f"({final_count} total)"
        )

        return total_downloaded

    def download_all_brands(self):
        """Download images for all brands"""
        logging.info("Starting download for all brands")
        total_new = 0

        for brand in BRANDS_QUERIES.keys():
            new_count = self.download_brand(brand)
            total_new += new_count

            # Longer pause between brands
            if new_count > 0:
                time.sleep(10)

        logging.info(f"Download completed. Total new images: {total_new}")
        self.close_metadata()


# ---------------------- POST-PROCESSING FUNCTIONS ----------------------
def deduplicate_dataset():
    """Remove duplicate images across all brands"""
    logging.info("Starting cross-brand deduplication...")

    processor = ImageProcessor()
    seen_hashes = set()
    removed_count = 0

    for brand in BRANDS_QUERIES.keys():
        brand_dir = os.path.join(Config.IMAGES_DIR, brand)
        if not os.path.exists(brand_dir):
            continue

        for filename in os.listdir(brand_dir):
            if not filename.lower().endswith((".jpg", ".jpeg")):
                continue

            filepath = os.path.join(brand_dir, filename)
            try:
                img = Image.open(filepath).convert("RGB")
                phash = processor.calculate_phash(img)

                if processor.is_duplicate(
                    seen_hashes, phash, threshold=Config.PHASH_THRESHOLD
                ):
                    os.remove(filepath)
                    removed_count += 1
                    logging.info(f"Removed duplicate: {filepath}")
                else:
                    seen_hashes.add(phash)

            except Exception as e:
                logging.warning(f"Failed to process {filepath}: {str(e)}")
                # Remove corrupt files
                try:
                    os.remove(filepath)
                    removed_count += 1
                except:
                    pass

    logging.info(f"Deduplication completed. Removed {removed_count} files.")


def create_splits(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Create train/val/test splits"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    logging.info("Creating dataset splits...")
    splits = {"train": [], "val": [], "test": []}

    for brand in BRANDS_QUERIES.keys():
        brand_dir = os.path.join(Config.IMAGES_DIR, brand)
        if not os.path.exists(brand_dir):
            continue

        images = [
            os.path.join(brand, f)
            for f in os.listdir(brand_dir)
            if f.lower().endswith((".jpg", ".jpeg"))
        ]
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        splits["train"].extend(images[:n_train])
        splits["val"].extend(images[n_train : n_train + n_val])
        splits["test"].extend(images[n_train + n_val :])

    # Write split files
    for split_name, files in splits.items():
        split_file = os.path.join(Config.OUTPUT_ROOT, f"{split_name}.txt")
        with open(split_file, "w", encoding="utf-8") as f:
            for filepath in files:
                f.write(filepath + "\n")

        logging.info(f"{split_name}: {len(files)} images")

    logging.info("Dataset splits created successfully.")


# ---------------------- MAIN EXECUTION ----------------------
def main():
    parser = argparse.ArgumentParser(description="Advanced Logo Dataset Downloader")
    parser.add_argument("--download", action="store_true", help="Download images")
    parser.add_argument("--dedupe", action="store_true", help="Remove duplicates")
    parser.add_argument(
        "--splits", action="store_true", help="Create train/val/test splits"
    )
    parser.add_argument(
        "--target",
        type=int,
        default=Config.TARGET_PER_BRAND,
        help="Target images per brand",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="duckduckgo,selenium",
        help="Sources: duckduckgo,selenium",
    )

    args = parser.parse_args()

    # Update config from arguments
    Config.TARGET_PER_BRAND = args.target

    if args.sources:
        enabled_sources = args.sources.split(",")
        for source in Config.ENABLED_SOURCES.keys():
            Config.ENABLED_SOURCES[source] = source in enabled_sources

    # Setup
    setup_logging()

    # Check if any sources are enabled
    if not any(Config.ENABLED_SOURCES.values()):
        logging.error("No search sources enabled! Check your configuration.")
        return

    # Execute requested actions
    if args.download:
        downloader = AdvancedLogoDownloader()
        downloader.download_all_brands()

    if args.dedupe:
        deduplicate_dataset()

    if args.splits:
        create_splits()

    if not any([args.download, args.dedupe, args.splits]):
        logging.info("No actions specified. Use --help for available options.")


if __name__ == "__main__":
    main()
