#!/bin/bash

export MODEL_NAME="stabilityai/stable-diffusion-2"
export DATASET_NAME="Norod78/cartoon-blip-captions"

accelerate launch --mixed_precision="bf16"  diffusers/examples/text_to_image/train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=4 \
  --gradient_accumulation_steps=8\
  --gradient_checkpointing \
  --num_train_epochs=10\
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="linear" --lr_warmup_steps=0 \
  --output_dir="output"\
  --logging_dir="logs"\
  --rank=16 \
  --allow_tf32 \
  --validation_prompt="the avatar characters with two men, one in front of the image and one holding a stick"\
  --enable_xformers_memory_efficient_attention \
