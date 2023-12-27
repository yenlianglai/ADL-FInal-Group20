from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.tokenize.texttiling import TextTilingTokenizer
import nltk
nltk.download("stopwords")
from diffusers import StableDiffusionPipeline
from typing import List
import pandas as pd
import argparse
import torch
import re
import os


class ImageGenerator:
    def __init__(self, model_id):
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        self.pipe = self.pipe.to("cuda")
    
    def generate(self, prompt: str):
        assert isinstance(prompt, str)
        image = self.pipe(prompt).images[0]  
        return image



class PromptGenerator:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
    
    def generate(self, raw: str):
        assert isinstance(raw, str)
        input = f'Instruction: Give a simple description of the image to the script.\nInput: {raw}\nOutput:'
        
        input_ids = tokenizer.encode(input, return_tensors='pt').cuda()
        outputs = model.generate(
            input_ids,
            max_length=384,
            do_sample=True,
            temperature=1.0,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            num_return_sequences=1)
        
        prompt = tokenizer.batch_decode(outputs[:, input_ids.size(1):], skip_special_tokens=True)
        return prompt


class Pipeline:
    def __init__(self, tokenizer, model, sd_model_id, data_path, output_path):
        self.data = pd.read_csv(data_path)
        self.promptGenerator = PromptGenerator(tokenizer, model)
        self.imageGenerator = ImageGenerator(sd_model_id)
        self.output_path = output_path
        
    def get_movie(self, title):
        matching_rows = self.data[self.data["Title"] == title]
        if matching_rows.empty:
            raise Exception(f"Movie titled '{title}' not found.")
        return matching_rows.index[0]


    def split_into_sentences(self, text):
        text = text.replace("\n", "")
        sentences = re.split(r"(?<=[.!?]) +", text)
        return "\n\n".join(sentences)


    def preprocess_tiling_output(self, tt_text):
        tt_text = tt_text.replace("\n\n", "<|end_sentence|>") + "<|end_sentence|>"
        if tt_text.startswith("<|end_sentence|>"):
            return tt_text[len("<|end_sentence|>") :]
        return tt_text


    def preprocess_tiling_input(self, text):
        return text.replace("<|end_sentence|>", "\n\n")


    def preprocessing_output(self, text):
        return text.replace("<|end_sentence|>", "")


    def text_segmentation(
        self,
        title: str,
        w=40,
        k=20,
    ) -> List[int]:
        idx = self.get_movie(title)
        processed_text = self.split_into_sentences(self.data.loc[idx, "plot_synopsis"])
        tiling_tokenizer = TextTilingTokenizer(w=w, k=k)
        sections = tiling_tokenizer.tokenize(self.preprocess_tiling_input(processed_text))

        pred_list = []
        for section in sections:
            pp_section = self.preprocess_tiling_output(section)
            pred_list.append(pp_section)
        return [self.preprocessing_output(item) for item in pred_list]
    
    def generate(self, movie_title):
        plot_list = self.text_segmentation(movie_title)
        accumulated_section = ""
        for i, section in enumerate(plot_list):
            # accumulated_section += section
            prompt = self.promptGenerator.generate(section)
            print(f"prompt: {prompt}")
            image = self.imageGenerator.generate(movie_title + "style" + prompt[0])
            image.save(os.path.join(self.output_path, f"{movie_title}_{i}.png"))
            print(f"Image saved as {movie_title}_{i}.png")

        print("Finished ✅")
                  



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate images from text')
    parser.add_argument('--movie_title', type=str, help='Movie title')
    parser.add_argument('--data_path', type=str, help='Path to movie plot data')
    parser.add_argument('--model_path', type=str, help='Path to model')
    parser.add_argument('--sd_model_id', type=str, help='Stable Diffusion model ID')
    parser.add_argument('--output_path', type=str, help='Path to output images')
    args = parser.parse_args()
    
    movie_title = args.movie_title
    data_path = args.data_path
    model_path = args.model_path
    sd_model_id = args.sd_model_id
    output_path = args.output_path
    
    assert os.path.exists(data_path)
    os.makedirs(output_path, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).eval().cuda()
    
    
    generator = Pipeline(tokenizer, model_path, sd_model_id, data_path, output_path)
    generator.generate(movie_title)
