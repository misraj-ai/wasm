import argparse
import logging
import os
import shutil
from multiprocessing import cpu_count
import requests
from pathlib import Path
import yaml
from datasets import load_from_disk
from PIL import Image, ImageFile
from tqdm.auto import tqdm

from obelics.processors import WebDocumentFilteringDocLevel, WebDocumentFilteringNodeLevel, WebDocumentDocLevelDeDup



from obelics.utils import (
    DIGITS_RE,
    FLAGGED_WORDS,
    NON_PRINTING_CHARACTERS_RE,
    PUNCTUATION,
    SPECIAL_CHARACTERS,
    STOPWORDS,
    UNICODE_PUNCTUATION,
)

# Avoid DecompressionBombError and truncated image error
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)





def download_model(url: str, save_path: str, chunk_size: int = 1024 * 1024) -> None:
    """
    Downloads the given url and saves it to the specified path

    Args:
        url: the url of the model
        save_path: the path of the file
    """
    response = requests.get(url)
    if response.status_code == 200:
        # Create directory if it doesn't exist
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Successfully downloaded the model saving it to ... {save_path}")
        # Write to file in chunks
        with open(save_path, "wb") as f:
            for chunk in tqdm(response.iter_content(chunk_size=chunk_size)):
                if chunk:  # Filter out keep-alive chunks
                    f.write(chunk)
    else:
        raise ValueError(f"Invalid URL or download failed: {url}")


def get_args():
   args = argparse.ArgumentParser(description="Filtering based on some heuristics.")
   args.add_argument("--dataset_web_doc_path",
                     type=str,
                     help="The path of the dataset to be filtered.",
                     default="./data/web_document_dataset")
   args.add_argument("--output_path",
                     type=str,
                     help="Output path of the filtered dataset.",
                     default="./data/filtered_data")
   
   return args.parse_args()


def main(args):
    # Create models directory
    NUM_PROC = min(os.cpu_count(),64)
    models_base_dir = './models'
    if not(os.path.exists(models_base_dir)):
      os.makedirs(models_base_dir)

    # Define model paths
    lang_id_model_path = os.path.join(models_base_dir, "lid.176.bin")

    kenlm_model_path = os.path.join(models_base_dir, "model.binary")


    sentencepiece_model_path = os.path.join(models_base_dir,"tokenizer.model")


    # Download models if requested
    if not(os.path.exists(lang_id_model_path)):
        logger.info("Downloading language model...")
        download_model(
            "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
            lang_id_model_path
        )
    if not(os.path.exists(kenlm_model_path)):
        logger.info("Downloading KenLM model...")
        download_model(
            "https://huggingface.co/edugp/kenlm/resolve/main/wikipedia/ar.arpa.bin?download=true",# The link to the KenLM model trained on 
            kenlm_model_path
        )
    if not(os.path.exists(sentencepiece_model_path)):
        logger.info("Downloading tokenizer model...")
        download_model(
            "https://huggingface.co/edugp/kenlm/resolve/main/wikipedia/ar.sp.model?download=true",
            sentencepiece_model_path
        )

    logger.info("Loading the web document dataset")


    web_document_dataset = load_from_disk(args.dataset_web_doc_path)
    logger.info("Dataset loaded successfully")

    with open("./obelics/configs/config_filter_web_documents.yaml") as f:
        filtering_params = yaml.safe_load(f)


    web_document_filtering_node_level = WebDocumentFilteringNodeLevel(
        cond_check_format=filtering_params["cond_check_format"],
        valid_formats=filtering_params["valid_formats"],
        cond_check_size_image=filtering_params["cond_check_size_image"],
        original_width_min_cutoff=filtering_params["original_width_min_cutoff"],
        original_width_max_cutoff=filtering_params["original_width_max_cutoff"],
        original_height_min_cutoff=filtering_params["original_height_min_cutoff"],
        original_height_max_cutoff=filtering_params["original_height_max_cutoff"],
        rendered_width_min_cutoff=filtering_params["rendered_width_min_cutoff"],
        rendered_width_max_cutoff=filtering_params["rendered_width_max_cutoff"],
        rendered_height_min_cutoff=filtering_params["rendered_height_min_cutoff"],
        rendered_height_max_cutoff=filtering_params["rendered_height_max_cutoff"],
        aspect_ratio_max_cutoff=filtering_params["aspect_ratio_max_cutoff"],
        cond_remove_non_printing_characters=filtering_params["cond_remove_non_printing_characters"],
        non_printing_characters_re=NON_PRINTING_CHARACTERS_RE,
        cond_standardize_whitespace=filtering_params["cond_standardize_whitespace"],
        cond_check_number_words_node_level=filtering_params["cond_check_number_words_node_level"],
        strip_characters=SPECIAL_CHARACTERS,
        number_words_node_level_min_cutoff=filtering_params["number_words_node_level_min_cutoff"],
        number_words_node_level_max_cutoff=filtering_params["number_words_node_level_max_cutoff"],
        cond_check_character_repetition_ratio_node_level=filtering_params["cond_check_character_repetition_ratio_node_level"],
        character_repetition_length_node_level=filtering_params["character_repetition_length_node_level"],
        character_repetition_node_level_max_cutoff=filtering_params["character_repetition_node_level_max_cutoff"],
        cond_check_word_repetition_ratio_node_level=filtering_params["cond_check_word_repetition_ratio_node_level"],
        word_repetition_length_node_level=filtering_params["word_repetition_length_node_level"],
        word_repetition_node_level_max_cutoff=filtering_params["word_repetition_node_level_max_cutoff"],
        cond_check_special_character_ratio_node_level=filtering_params["cond_check_special_character_ratio_node_level"],
        special_character_ratio_node_level_max_cutoff=filtering_params["special_character_ratio_node_level_max_cutoff"],
        cond_check_stopword_ratio_node_level=filtering_params["cond_check_stopword_ratio_node_level"],
        stopwords=STOPWORDS,
        stopword_ratio_node_level_min_cutoff=filtering_params["stopword_ratio_node_level_min_cutoff"],
        cond_check_flagged_word_ratio_node_level=filtering_params["cond_check_flagged_word_ratio_node_level"],
        flagged_words=FLAGGED_WORDS,
        flagged_word_ratio_node_level_max_cutoff=filtering_params["flagged_word_ratio_node_level_max_cutoff"],
        cond_check_punctuation_ratio_node_level=filtering_params["cond_check_punctuation_ratio_node_level"],
        min_number_words_to_check_punctuation_ratio_node_level=filtering_params["min_number_words_to_check_punctuation_ratio_node_level"],
        punctuation=PUNCTUATION,
        punctuation_ratio_node_level_min_cutoff=filtering_params["punctuation_ratio_node_level_min_cutoff"],
        cond_check_common_word_ratio_node_level=filtering_params["cond_check_common_word_ratio_node_level"],
        path_common_words="common_words_path",
        common_word_ratio_node_level_min_cutoff=filtering_params["common_word_ratio_node_level_min_cutoff"],
        cond_check_lang_id_node_level=filtering_params["cond_check_lang_id_node_level"],
        path_lang_id_model=lang_id_model_path,
        lang_id_node_level_min_cutoff=filtering_params["lang_id_node_level_min_cutoff"],
        cond_check_perplexity_score_node_level=filtering_params["cond_check_perplexity_score_node_level"],
        digits_re=DIGITS_RE,
        unicode_punctuation=UNICODE_PUNCTUATION,
        path_sentencepiece_model=sentencepiece_model_path,
        path_kenlm_model=kenlm_model_path,
        perplexity_score_node_level_max_cutoff=filtering_params["perplexity_score_node_level_max_cutoff"],
    )
 

    logger.info("Starting filtering at node level")
    web_document_dataset_filtered = web_document_dataset.map(
        web_document_filtering_node_level,
        num_proc=NUM_PROC
    )
    logger.info("Node level filtering complete")

    deDup = WebDocumentDocLevelDeDup(
        cond_deduplicate_nodes_doc_level = filtering_params["cond_deduplicate_nodes_doc_level"],
        deduplication_max_cuttoff = filtering_params["deduplication_max_cuttoff"],

    )
    logger.info("Starting de-duplication")

    web_document_dataset_filtered = web_document_dataset_filtered.map(
        deDup,
        num_proc=NUM_PROC,
    )
    logger.info("De-duplication complete")



    web_document_filtering_doc_level = WebDocumentFilteringDocLevel(
        cond_check_number_images=filtering_params['cond_check_number_images'],
        number_images_min_cutoff=filtering_params["number_images_min_cutoff"],
        number_images_max_cutoff=filtering_params["number_images_max_cutoff"],
        cond_check_number_words_doc_level=filtering_params["cond_check_number_words_doc_level"],
        strip_characters=SPECIAL_CHARACTERS,
        number_words_doc_level_min_cutoff=filtering_params["number_words_doc_level_min_cutoff"],
        number_words_doc_level_max_cutoff=filtering_params["number_words_doc_level_max_cutoff"],
        cond_check_character_repetition_ratio_doc_level=filtering_params["cond_check_character_repetition_ratio_doc_level"],
        character_repetition_length_doc_level=filtering_params["character_repetition_length_doc_level"],
        character_repetition_doc_level_max_cutoff=filtering_params["character_repetition_doc_level_max_cutoff"],
        cond_check_word_repetition_ratio_doc_level=filtering_params["cond_check_word_repetition_ratio_doc_level"],
        word_repetition_length_doc_level=filtering_params["word_repetition_length_doc_level"],
        word_repetition_doc_level_max_cutoff=filtering_params["word_repetition_doc_level_max_cutoff"],
        cond_check_special_character_ratio_doc_level=filtering_params["cond_check_special_character_ratio_doc_level"],
        special_character_ratio_doc_level_max_cutoff=filtering_params["special_character_ratio_doc_level_max_cutoff"],
        cond_check_stopword_ratio_doc_level=filtering_params["cond_check_stopword_ratio_doc_level"],
        stopwords=STOPWORDS,
        stopword_ratio_doc_level_min_cutoff=filtering_params["stopword_ratio_doc_level_min_cutoff"],
        cond_check_flagged_word_ratio_doc_level=filtering_params["cond_check_flagged_word_ratio_doc_level"],
        flagged_words=FLAGGED_WORDS,
        flagged_word_ratio_doc_level_max_cutoff=filtering_params["flagged_word_ratio_doc_level_max_cutoff"],
        cond_check_punctuation_ratio_doc_level=filtering_params["cond_check_punctuation_ratio_doc_level"],
        punctuation=PUNCTUATION,
        punctuation_ratio_doc_level_min_cutoff=filtering_params["punctuation_ratio_doc_level_min_cutoff"],
        cond_check_common_word_ratio_doc_level=filtering_params["cond_check_common_word_ratio_doc_level"],
        path_common_words="common_words_path",
        common_word_ratio_doc_level_min_cutoff=filtering_params["common_word_ratio_doc_level_min_cutoff"],
        cond_check_lang_id_doc_level=filtering_params["cond_check_lang_id_doc_level"],
        path_lang_id_model=lang_id_model_path,
        lang_id_doc_level_min_cutoff=filtering_params["lang_id_doc_level_min_cutoff"],
        cond_check_perplexity_score_doc_level=filtering_params["cond_check_perplexity_score_doc_level"],
        non_printing_characters_re=NON_PRINTING_CHARACTERS_RE,
        digits_re=DIGITS_RE,
        unicode_punctuation=UNICODE_PUNCTUATION,
        path_sentencepiece_model=sentencepiece_model_path,
        path_kenlm_model=kenlm_model_path,
        perplexity_score_doc_level_max_cutoff=filtering_params["perplexity_score_doc_level_max_cutoff"],
    )

    logger.info("Starting filtering at document level")
    web_document_dataset_filtered = web_document_dataset_filtered.filter(
        web_document_filtering_doc_level,
        num_proc=NUM_PROC
    )

    logger.info("Document level filtering complete")

    logger.info("Saving filtered dataset")
    if web_document_dataset_filtered.num_rows:
      web_document_dataset_filtered.save_to_disk(
          args.output_path
      )

      logger.info("Filtered dataset saved successfully")

    logger.info(f"Original dataset size: {web_document_dataset.num_rows}")
    logger.info(f"Filtered dataset size: {web_document_dataset_filtered.num_rows}")
    
    logger.info("Removing the original dataset")
    shutil.rmtree(args.dataset_web_doc_path)
    logger.info("Successfully removed the original dataset")


if __name__ =="__main__":
   args = get_args()
   main(args)