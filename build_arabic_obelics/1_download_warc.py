import argparse
import logging 
import pandas as pd 
import numpy as np
import boto3
from typing import Dict, List, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import os
from tqdm import tqdm
from datasets import Dataset
from dotenv import load_dotenv
import pyarrow as pa
import pyarrow.parquet as pq
import shutil
from datasets import Dataset, concatenate_datasets
import glob
import multiprocessing


load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_args():
    parser = argparse.ArgumentParser(description="Download warc files from Common Crawl pointers.")
    parser.add_argument(
        "--sample_size",
        type=int,
        help="The size of the sample",
        default=2_500_000
    )
    parser.add_argument(
        "--sample_path",
        type=str,
        help="The Path of the dataset",
        default="./data/common_crawl.csv"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="The output path of the dataset.",
        default="./data/warc_dataset"
    )
    parser.add_argument(
        "--temp_save_path",
        type=str,
        help="A temporary file to store the data",
        default="./data/warc_parquet"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Number of records to process in each batch",
        default=10000
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        help="The batch to start at helpful if there are errors",
        default=0
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        help="Number of processes to use for parallel processing",
        default=multiprocessing.cpu_count()
    )
    parser.add_argument(
    "--skip_rows",
    type=int,
    help="Number of rows to skip since the code is being run on different servers",
    default=0
    )
    
    args = parser.parse_args()
    return args


class CommonCrawlDownloader:
    def __init__(self, max_concurrent_downloads: int = 1000):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name="us-east-1"
        )
        self.max_concurrent_downloads = max_concurrent_downloads
        # Global progress bar
        self.pbar = None

    def download_warc_record(self, warc_metadata: Dict[str, Union[str, int]]) -> Dict[str, Union[str, bool, bytes]]:
        """
        Download specific byte range from WARC file in Common Crawl S3

        :param warc_metadata: Metadata including warc_filename, offset, and length
        :return: Metadata with downloaded record
        """
        try:
            source_bucket = 'commoncrawl'
            source_key = warc_metadata['warc_filename']

            # Byte range request
            byte_range = f"bytes={warc_metadata['warc_record_offset']}-" \
                         f"{warc_metadata['warc_record_offset'] + warc_metadata['warc_record_length'] - 1}"

            response = self.s3_client.get_object(
                Bucket=source_bucket,
                Key=source_key,
                Range=byte_range
            )

            record_data = response['Body'].read()

            warc_metadata['warc'] = record_data
            warc_metadata['warc_error'] = ""
            warc_metadata['download_success'] = True

        except Exception as e:
            warc_metadata['download_success'] = False
            warc_metadata['warc_error'] = str(e)
            warc_metadata['warc'] = b""

        # Update global progress bar
        if self.pbar:
            self.pbar.update(1)
            
        return warc_metadata

    def download_records(self, warc_metadata_list: List[Dict[str, Union[str, int]]]) -> List[Dict]:
        """
        Download multiple WARC records concurrently

        :param warc_metadata_list: List of WARC record metadata
        :return: List of processed metadata
        """
        # Initialize global progress bar
        self.pbar = tqdm(total=len(warc_metadata_list), desc="Downloading WARC Records")
        
        with ThreadPoolExecutor(max_workers=self.max_concurrent_downloads) as executor:
            results = list(executor.map(self.download_warc_record, warc_metadata_list))

        self.pbar.close()
        return results


def process_batch(args_tuple):
    """
    Process a batch of WARC records and save to disk
    
    :param args_tuple: Tuple containing (batch_data, batch_index, save_path, total_batches)
    :return: Number of successful downloads
    """
    batch_data, batch_index, save_path, total_batches = args_tuple
    
    process_id = os.getpid()
    logger.info(f"Process {process_id}: Processing batch {batch_index}/{total_batches}")
    
    downloader = CommonCrawlDownloader(max_concurrent_downloads=10)
    processed_data = downloader.download_records(batch_data)
    
    # Define the schema explicitly
    schema = pa.schema([
        ('warc_filename', pa.string()),
        ('warc_record_offset', pa.int64()),
        ('warc_record_length', pa.int64()),
        ('warc', pa.binary()),
        ('warc_error', pa.string()),
        ('url',pa.string()),
        ('url_host_registered_domain',pa.string()),
        ('download_success', pa.bool_())
        # Add any other fields that are in your data
    ])
    
    # Clean data before conversion
    clean_data = []
    for item in processed_data:
        clean_item = {
            'warc_filename': str(item.get('warc_filename', '')),
            'warc_record_offset': int(item.get('warc_record_offset', 0)),
            'warc_record_length': int(item.get('warc_record_length', 0)),
            'warc': item.get('warc', b'') if isinstance(item.get('warc'), bytes) else b'',
            'warc_error': str(item.get('warc_error', '')),
            'url': str(item.get('url')),
            'url_host_registered_domain': str(item.get('url_host_registered_domain')),
            'download_success': bool(item.get('download_success', False))
            # Handle other fields as needed
        }
        clean_data.append(clean_item)
    
    # Create table with schema
    table = pa.Table.from_pylist(clean_data, schema=schema)
    batch_file_path = os.path.join(save_path, f"batch_{batch_index}.parquet")
    pq.write_table(table, batch_file_path)
    
    # Count successes
    success_count = sum(item['download_success'] for item in clean_data)
    
    # Log success rate for this batch
    logger.info(f"Process {process_id}: Batch {batch_index}/{total_batches} completed. Success rate: {success_count/len(batch_data):.2%}")
    
    # Clear memory
    del processed_data
    del clean_data
    del table
    
    return success_count


def create_hf_ds(temp_save_path, save_path):
    # Find all parquet files in the directory
    parquet_dirs = os.path.join(temp_save_path, "batch_*.parquet")
    parquet_files = glob.glob(parquet_dirs)

    # Create datasets from individual parquet files
    datasets_list = [Dataset.from_parquet(file) for file in parquet_files]

    # Concatenate them into a single dataset
    if datasets_list:
        ds = concatenate_datasets(datasets_list)
        logger.info(f"Successfully loaded dataset with {len(ds)} records")
        logger.info(f"Saving it to {save_path}")
        ds.save_to_disk(save_path)
        return True
    else:
        logger.info("No parquet files found")
        return False 


def humanbytes(B):
    """Return the given bytes as a human-friendly KB, MB, GB, or TB string."""
    B = float(B)
    KB = float(1024)
    MB = float(KB ** 2)  # 1,048,576
    GB = float(KB ** 3)  # 1,073,741,824
    TB = float(KB ** 4)  # 1,099,511,627,776

    if B < KB:
        return f"{B} {'Bytes' if B == 1 else 'Byte'}"
    elif KB <= B < MB:
        return f"{B / KB:.2f} KB"
    elif MB <= B < GB:
        return f"{B / MB:.2f} MB"
    elif GB <= B < TB:
        return f"{B / GB:.2f} GB"
    elif TB <= B:
        return f"{B / TB:.2f} TB"


def main(args):    
    if AWS_ACCESS_KEY_ID: 
        logger.info("Keys loaded successfully.")
    else: 
        logger.info(f"AWS_ACCESS_KEY_ID:{AWS_ACCESS_KEY_ID}")
    
    # Set up multiprocessing method for compatibility across platforms
    multiprocessing.set_start_method('spawn', force=True)
    
    NUM_PROC  = min(args.num_processes,64)
    np.random.seed(42)
    sample_size = args.sample_size
    logger.info(f"Loading CSV with {sample_size} rows from {args.sample_path}")
    logger.info(f"Skipping {args.skip_rows} rows  and reading next {sample_size} rows")
    df = pd.read_csv(args.sample_path, skiprows=range(1, args.skip_rows + 1), nrows=sample_size)

    
    logger.info(f"Size of the test sample {humanbytes(df['warc_record_length'].sum())}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.temp_save_path, exist_ok=True)
    
    # Process in batches
    batch_size = args.batch_size
    all_data = df.to_dict('records')
    total_batches = (len(all_data) + batch_size - 1) // batch_size
    
    logger.info(f"Processing {len(all_data)} records in {total_batches} batches using {NUM_PROC} processes")
    
    # Prepare batch arguments for multiprocessing
    batch_args = []
    for i in range(args.start_idx, total_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(all_data))
        batch_data = all_data[start_idx:end_idx]
        batch_args.append((batch_data, i, args.temp_save_path, total_batches))
    
    # Process batches in parallel using a process pool
    total_success = 0
    with ProcessPoolExecutor(max_workers=NUM_PROC) as executor:
        results = list(tqdm(
            executor.map(process_batch, batch_args),
            total=len(batch_args),
            desc="Processing batches"
        ))
        total_success = sum(results)
    
    # Log overall success rate
    logger.info(f"Overall download success rate: {total_success/len(all_data):.2%}")
    
    logger.info("Starting to convert the data into a hugging face dataset")
    parquet_dirs = os.path.join(args.temp_save_path, "batch_*.parquet")
    parquet_files = glob.glob(parquet_dirs)
    
    if len(parquet_files) == total_batches - args.start_idx:
        converted_to_hf = create_hf_ds(args.temp_save_path, args.save_path)
        if converted_to_hf: 
            shutil.rmtree(args.temp_save_path)
            logger.info(f"Successfully created HuggingFace dataset at {args.save_path}")
    else: 
        logger.info(f"Number of parquet files ({len(parquet_files)}) does not match the expected number of batches ({total_batches - args.start_idx})")
        logger.info("Please re-run the script or check for errors")


if __name__ == "__main__":
    args = get_args()
    main(args)