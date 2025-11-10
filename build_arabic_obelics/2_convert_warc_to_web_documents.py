import os
import logging
import yaml
import argparse
import shutil
from datasets import load_from_disk, Dataset
from obelics.processors import (
    DOMTreeSimplificator,
    HtmlExtractor,
    PreExtractionSimplificator,
    CommonCrawlWebDocumentExtractor,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger('bs4.dammit').setLevel(logging.ERROR)


def get_args():
    args = argparse.ArgumentParser(description="Converts the downloaded warc files to HTML and then webdoc and markdown.")
    args.add_argument("--warc_dataset_path",
                      type=str,
                      help="The path of the WARC dataset.",
                      default="./data/warc_dataset")
    args.add_argument("--save_path_web_doc",
                      type=str,
                      help="The path to save the html dataset.",
                      default="./data/web_document_dataset")
    args.add_argument("--path_save_file_image_urls",
                      type=str,
                      help="The path to save the images urls",
                      default="./data/image_urls.txt")

    return args.parse_args()

# Main processing
def main(args):
    # Create dataset
    NUM_PROC = min(os.cpu_count(),64)

    warc_dataset = load_from_disk(args.warc_dataset_path)
    logger.info("Created WARC dataset")

    # Extract HTML
    if ("html" not in warc_dataset.column_names) and ("html_error" not in warc_dataset.column_names):
        warc_dataset = warc_dataset.add_column("html", [""] * len(warc_dataset))
        warc_dataset = warc_dataset.add_column("html_error", [""] * len(warc_dataset))

    html_extractor = HtmlExtractor()
    logger.info("Starting HTML extraction")

    html_dataset = warc_dataset.map(html_extractor, num_proc=NUM_PROC)
    logger.info("Finished HTML extraction")

    # Calculate success rate
    num_successes = len([1 for el in html_dataset["html_error"] if not el])
    logger.info(
        f"Success rate for HTML extraction: {num_successes} /"
        f" {len(html_dataset)} ({num_successes / len(html_dataset) * 100}%)"
    )
    with open("./obelics/configs/config_extract_web_documents.yaml") as f:
        extraction_params = yaml.load(f, Loader=yaml.FullLoader)

    dom_tree_simplificator = DOMTreeSimplificator(
        strip_multiple_linebreaks=extraction_params["dom_tree_simplificator"]["strip_multiple_linebreaks"],
        strip_multiple_spaces=extraction_params["dom_tree_simplificator"]["strip_multiple_spaces"],
        remove_html_comments=extraction_params["dom_tree_simplificator"]["remove_html_comments"],
        replace_line_break_tags=extraction_params["dom_tree_simplificator"]["replace_line_break_tags"],
        unwrap_tags=extraction_params["dom_tree_simplificator"]["unwrap_tags"],
        strip_tags=extraction_params["dom_tree_simplificator"]["strip_tags"],
        strip_special_divs=extraction_params["dom_tree_simplificator"]["strip_special_divs"],
        remove_dates=extraction_params["dom_tree_simplificator"]["remove_dates"],
        remove_empty_leaves=extraction_params["dom_tree_simplificator"]["remove_empty_leaves"],
        unnest_nodes=extraction_params["dom_tree_simplificator"]["unnest_nodes"],
        remake_tree=extraction_params["dom_tree_simplificator"]["remake_tree"],
        css_rules=extraction_params["dom_tree_simplificator"]["css_rules"],
        css_rules_replace_with_text=extraction_params["dom_tree_simplificator"]["css_rules_replace_with_text"],
    )
    pre_extraction_simplificator = PreExtractionSimplificator(
        only_text_image_nodes=extraction_params["pre_extraction_simplificator"]["only_text_image_nodes"],
        format_texts=extraction_params["pre_extraction_simplificator"]["format_texts"],
        merge_consecutive_text_nodes=extraction_params["pre_extraction_simplificator"]["merge_consecutive_text_nodes"],
    )
    web_document_extractor = CommonCrawlWebDocumentExtractor(
        html_dataset=html_dataset,
        dom_tree_simplificator=dom_tree_simplificator,
        pre_extraction_simplificator=pre_extraction_simplificator,
        path_save_dir_dataset=None,
        num_proc=NUM_PROC,
        path_save_file_image_urls=args.path_save_file_image_urls,
        path_save_dir_downloaded_images=None,
        thread_count=None,
        number_sample_per_shard=None,
        image_size=None,
        resize_mode=None,
        path_save_dir_tmp_datasets_images=None,
        path_save_dir_dataset_images=None,
        path_save_file_map_url_idx=None,
        num_proc_urls_to_images=None,
        path_save_dir_sharded_dataset=None,
        shard_size=None,
    )

    # Process and extract image URLs
    web_document_extractor.html_to_web_documents()
    html_dataset_extracted = web_document_extractor.dataset
    web_document_extractor.get_image_urls()


    # Save the processed dataset
    logger.info("Saving the HTML dataset")
    html_dataset_extracted.save_to_disk(args.save_path_web_doc)
    logger.info("Finished saving the HTML dataset")
    
    logger.info("Removing the original WARC dataset to save space")
    shutil.rmtree(args.warc_dataset_path)
    logger.info("Successfully remove the WARC datasets")

if __name__ == "__main__":
    args = get_args()
    main(args)