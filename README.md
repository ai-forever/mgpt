# mGPT

Multilingual Generative Pretrained Transformer

 - 1.3 billion parameter model
 - Trained on 60 languages
 - HuggingFace compatible [model card](https://huggingface.co/sberbank-ai/mGPT)

 ## Setting up environment

`pip install -r requirements.txt`  

## Checkpoint backup

Download checkpoints to load model from disk:
```
!wget https://files.sberdisk.ru/s/NzeBqYE84TAQDiS/download -O model.zip
!unzip model.zip -d mgptxl
model_name = "./mgptxl" 
```

## Transformers usage 🤗

```
from transformers import GPT2LMHeadModel, GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/mGPT")
model = GPT2LMHeadModel.from_pretrained("sberbank-ai/mGPT")

text = "Александр Сергеевич Пушкин родился в "
input_ids = tokenizer.encode(text, return_tensors="pt").cuda(device)
out = model.generate(
        input_ids, 
        min_length=100, 
        max_length=100, 
        eos_token_id=5, 
        pad_token=1,
        top_k=10,
        top_p=0.0,
        no_repeat_ngram_size=5
)
generated_text = list(map(tokenizer.decode, out))[0]
print(generated_text)
Александр Сергеевич Пушкин родился в  г. Санкт-Петербурге.
```

## Choosing best parameters:

In general:
```min_length=100,
eos_token_id=5, 
pad_token=1,
do_sample=True,
top_k=0,
top_p=0.8,
no_repeat_ngram_size=4
```

English Generation: 
```top_p=0.95, top_k=0```



## Examples


#### mGPT Generation Examples
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Vd3TEh1ojBvE7q8BDLmcA9RXeq0aQIlf?usp=sharing)

#### mGPT Fine-tuning example
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qkDhzEab2MXvohOuQYgKixHHimlh1Oh2?usp=sharing)

## Languages supported

 - **Languages:** Afrikaans, Azerbaijani, Belarusian, Bengali, Chuvash, German, English, Basque, Finnish, Hebrew (modern), Hungarian, Indonesian, Japanese, Kazakh, Kirghiz, Kyrgyz, Latvian, Mongolian, Malay, Dutch, Polish, Romanian, Moldavan, Yakut, Swahili, Telugu, Thai, Turkish, Tuvinian, Urdu, Vietnamese, Yoruba, Arabic, Bashkir, Bulgarian, Buriat, Danish, Greek, Modern, Spanish; Castilian, Persian, French, Hindi, Armenian, Italian, Georgian, Korean, Lithuanian, Malayalam, Marathi, Burmese, Ossetian, Ossetic, Portuguese, Russian, Swedish, Tamil, Tajik, Turkmen, Tatar, Ukrainian, Uzbek, Kalmyk, Chinese
  - **ISO codes:** az, sw, af, ar, ba, be, bxr, bg, bn, cv, hy, da, de, el, es, eu, fa, fi, fr, he, hi, hu, kk, id, it, ja, ka, ky, ko, lt, lv, mn, ml, os, mr, ms, my, nl, ro, pl, pt, sah, ru, tg, sv, ta, te, tk, th, tr, tl, tt, tyv, uk, en, ur, vi, uz, yo, zh, xal

 ## Cite Us 

 mGPT: Few-Shot Learners Go Multilingual

 [Abstract](https://arxiv.org/abs/2204.07580) [PDF](https://arxiv.org/pdf/2204.07580.pdf)

 ![](https://habrastorage.org/webt/1q/ru/yt/1qruytul6m2m-upyk9frq3pgrds.png)

 ```
@misc{https://doi.org/10.48550/arxiv.2204.07580,
  doi = {10.48550/ARXIV.2204.07580},
  
  url = {https://arxiv.org/abs/2204.07580},
  
  author = {Shliazhko, Oleh and Fenogenova, Alena and Tikhonova, Maria and Mikhailov, Vladislav and Kozlova, Anastasia and Shavrina, Tatiana},
  
  keywords = {Computation and Language (cs.CL), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences, I.2; I.2.7, 68-06, 68-04, 68T50, 68T01},
  
  title = {mGPT: Few-Shot Learners Go Multilingual},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}

 ```



## Contributing

We welcome community contributions to the model, and celebrate both its inference and training technique enhencements

