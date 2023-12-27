# ADL-FInal-Group20

## Guide

### To run the inference code

**sh run.sh:** 

- Parameters \
    --model_path # model path for text2prompt model \
    --output_path # output folder path \
    --sd_model_id # model path for stable diffusion model \
    --sd_lora_path # model path for the lora model \
    --data_path # data path for the input movie plot \
    --movie_title # in dataset movie tile

### Folder Structure

```bash
.
├── download.sh
├── main.py
├── movie_plot.csv
├── output
├── README.md
├── run.sh
├── train.sh
└── train_text_to_image_lora.py
```
- `download.sh` is the script to download the lora model from google drive
- `train.sh` is the script to run `train_text_to_image_lora.py`
- `run.sh` is the script to run the inference code in `main.py`
- `movie_plot.csv` is our local movie dataset
- `output` is the default output folder for generated images