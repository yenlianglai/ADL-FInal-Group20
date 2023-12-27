python3 ./main.py \
    --model_path alibaba-pai/pai-bloom-1b1-text2prompt-sd \
    --sd_lora_path ./lora_model \
    --output_path ./finetuned \
    --sd_model_id stabilityai/stable-diffusion-2 \
    --data_path ./movie_plot.csv \
    --movie_title "Fast & Furious" \
