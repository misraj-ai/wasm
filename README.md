# WASM

**WASM is an open source pipeline for creating and curating multimodal Arabic Data from common crawl**

**Dataset Sample:** https://huggingface.co/datasets/Misraj/msdd

**Paper:** Coming soon


## Usage 

```
git clone https://repo/url 
cd wasm 
pip install -r requirements.txt
```

If you want to download the common crawl data first you need to query the data using AWS Athena. The query should look like this.  
```
SELECT url,
       warc_filename,
       warc_record_offset,
       warc_record_length,
       content_languages,
       fetch_status,
       url_host_registered_domain,
FROM "ccindex"."ccindex"
WHERE crawl = 'CRAWL_NAME'
  AND subset = 'warc'
  AND content_languages LIKE '%ara%'
  AND fetch_status = 200
GROUP BY url,
         warc_filename,
         warc_record_offset,
         warc_record_length,
         content_languages,
         fetch_status,
         url_host_registered_domain;
```

After storing the results of the query on an S3 Bucket you have to add it to  `build_arabic_obelics/0_download_metadata.py` line 37. 

You also need to create a `.env` file and add the following: 
``` 
AWS_ACCESS_KEY_ID=YOUR_AWS_ACCESS_KEY
AWS_SECRET_ACCESS_KEY=YOUR_AWS_SECRET_ACCESS_KEY
``` 

**Then the pipeline is ready to launch** 

1. Downloading the metadata 
``` 
python build_arabic_obelics/0_download_metadata.py
``` 
This script takes two arguments: 
* save_path: where to save the downloaded data 
* Dump name the name of the dump stored on AWS S3

2. Downloading the WARC files: 
The script downloads the warc files given the WARC filename, WARC record offset, WARC record length. It downloads the data in batches and saves them in case something happens it does not need to start from scratch. 
``` 
python build_arabic_obelics/1_download_warc.py
``` 
The script takes eight arguments: 
* sample_size: The size of the sample to be downloaded in case the instance running can't handle huge amounts of data 
* sample_path: The path of the metadata downloaded by the previous  script  
* save_path: The path to store the resulting WARC data set 
* temp_save_path: The path to store the temporary WARC files after download 
* batch_size: The number of samples to store in the `temp_save_path` 
* start_idx: In case data was previously downloaded from this metadata this argument ensures they are not downloaded again 
* num_processes: The number of processors to use 
* skip_rows: Number of rows to skip

3. Converting War to web document 

```
python -m build_arabic_obelics.2_convert_warc_to_web_documents
``` 
The script takes three arguments: 
* warc_dataset_path: The path of the WARC dataset 
* save_path_web_doc: Where to store the resulting web doc dataset 
* path_save_file_image_urls: Where to store the images urls 

4. Filtering the data

```
python -m build_arabic_obelics.3_filtering
``` 
The script takes two arguments: 
* dataset_web_doc_path: The path of the web doc dataset 
* output_path: Where to store the resulting filtered dataset

**NOTE:** all the arguments have default values 

## Citation

If you are using this code, please cite
```
@misc{hennara2025wasmpipelineconstructingstructured,
      title={Wasm: A Pipeline for Constructing Structured Arabic Interleaved Multimodal Corpora}, 
      author={Khalil Hennara and Ahmad Bastati and Muhammad Hreden and Mohamed Motasim Hamed and Zeina Aldallal and Sara Chrouf and Safwan AlModhayan},
      year={2025},
      eprint={2511.07080},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.07080}, 
}
```
