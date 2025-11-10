import os 
import boto3
import botocore
import argparse
import logging
from dotenv import load_dotenv


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



def get_args():
   args = argparse.ArgumentParser(description="Download the dataset metadata")
   args.add_argument("--save_path",
                     type=str,
                     default="./data/common_crawl.csv",
                     help="The save path of the data."
                     )
   args.add_argument("--dump_name",
                     type=str,
                     default="CC-MAIN-2025-13",
                     help="The name of the dump metadata")
   return args.parse_args()


def main(args):
    load_dotenv()
    AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
    BUCKET_NAME = '' # replace with your bucket name

    # enter authentication credentials
    s3 = boto3.resource('s3', aws_access_key_id = AWS_ACCESS_KEY_ID,
                              aws_secret_access_key= AWS_SECRET_ACCESS_KEY,
                              region_name="us-east-1")



    KEY = f'common_crawl_dumps/{args.dump_name}.csv' # replace with your object key

    try:
      # we are trying to download training dataset from s3 with name `my-training-data.csv`
      logger.info("Starting to download the metadata")
      s3.Bucket(BUCKET_NAME).download_file(KEY, args.save_path)
      logger.info(f"Successfully downloaded the metadata and saved to {args.save_path}")

    except botocore.exceptions.ClientError as e:
      if e.response['Error']['Code'] == "404":
        logger.info("The object does not exist.")
      else:
        raise
      
if __name__ =="__main__":
    args = get_args()
    main(args)