# mGPT Armenian finetune

We introduce a monolingual GPT-3-based model for Armenian language. 

The model is based on [mGPT](https://huggingface.co/sberbank-ai/mGPT/), a family of autoregressive GPT-like models with 1.3 billion parameters trained on 60 languages from 25 language families using Wikipedia and Colossal Clean Crawled Corpus. 


![](https://habrastorage.org/webt/4h/pp/tq/4hpptqkgytnoi9ax58wdrymsxx4.png)
What happens on this image? The model is originally trained with sparse attention masks, then fine-tuned with no sparsity on the last steps (perplexity and loss peak). Getting rid of the sparsity in the end of the training helps to integrate the model into the GPT2 HF class.

### Multilingual Generative Pretrained Transformer - Armenian monolingual finetune
[![Apache license](https://img.shields.io/badge/License-Apache-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mgpt-few-shot-learners-go-multilingual/few-shot-ner-on-xglue)](https://paperswithcode.com/sota/few-shot-ner-on-xglue?p=mgpt-few-shot-learners-go-multilingual)
	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mgpt-few-shot-learners-go-multilingual/part-of-speech-tagging-on-xglue)](https://paperswithcode.com/sota/part-of-speech-tagging-on-xglue?p=mgpt-few-shot-learners-go-multilingual)
	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mgpt-few-shot-learners-go-multilingual/cross-lingual-transfer-on-xcopa)](https://paperswithcode.com/sota/cross-lingual-transfer-on-xcopa?p=mgpt-few-shot-learners-go-multilingual)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mgpt-few-shot-learners-go-multilingual/cross-lingual-paraphrase-identification-on)](https://paperswithcode.com/sota/cross-lingual-paraphrase-identification-on?p=mgpt-few-shot-learners-go-multilingual)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mgpt-few-shot-learners-go-multilingual/cross-lingual-natural-language-inference-on-4)](https://paperswithcode.com/sota/cross-lingual-natural-language-inference-on-4?p=mgpt-few-shot-learners-go-multilingual)


[[Paper]](https://arxiv.org/abs/2204.07580) [[Habr]](https://habr.com/ru/company/sberdevices/blog/662195/) [[Model Card]](https://huggingface.co/sberbank-ai/mGPT) 

 - 1.3 billion parameter model
 - Trained on 60 languages
 - HuggingFace compatible [model card](https://huggingface.co/sberbank-ai/mGPT)

## Training 

We reproduce the GPT-3 architecture using GPT-2 sources and the sparse attention mechanism, [Deepspeed](https://github.com/microsoft/DeepSpeed) and [Megatron](https://github.com/NVIDIA/Megatron-LM) frameworks allows us to effectively parallelize the training and inference steps. The resulting models show performance on par with the recently released [XGLM](https://arxiv.org/pdf/2112.10668.pdf) models at the same time covering more languages and enhancing NLP possibilities for low resource languages. 

The model was fine-tuned on 170GB of Armenian texts, including MC4, Archive.org fiction, EANC public data, OpenSubtitles, OSCAR corpus and blog texts.

Val perplexity is 2.046.

The mGPT model was pre-trained for 12 days x 256 GPU (Tesla NVidia V100), 4 epochs, then 9 days x 64 GPU, 1 epoch

The Armenian finetune was around 7 days with 4 Tesla NVidia V100 and has made 160k steps.

## Web Demo
Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the Web Demo for generation: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/sberbank-ai/mGPT-armenian/) 
![](https://habrastorage.org/webt/ud/a8/mn/uda8mnufnq7oavqbivs6ciwdhb8.png)

 ## Setting up environment

`pip install -r requirements.txt`  


## Transformers usage 🤗

```
from transformers import GPT2LMHeadModel, GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/mGPT-armenian")
model = GPT2LMHeadModel.from_pretrained("sberbank-ai/mGPT-armenian")

text = "Նորություններ. Բրազիլացի գիտնականները հայտնաբերել են Ուութլանդի արեւմուտքում ապրող գաճաճ միաեղջյուրների հազվագյուտ տեսակին:"
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
Նորություններ. Բրազիլացի գիտնականները հայտնաբերել են Ուութլանդի արեւմուտքում ապրող գաճաճ միաեղջյուրների հազվագյուտ տեսակին: Այս մասին գրում է «Ֆրանս Պրեսը»: Նշվում է, որ գիտնականների հետազոտությունները կրկին հաջողությամբ են ավարտվել:
Գիտնականների պնդմամբ, գաճաճ միաեղջյուրների մեծ մասը ունի ուղիղ ճակատ, եւ գտնվում են երկարակյաց եւ առատ ձմեռներով գյուղատնտեսական բուսատեսակների տնկման վայրերում, ինչը նպաստում է ձմեռային բնույթի սառնամանիքների եւ քամիների դեմ պայքարելու հմտությանը:
Ուութլանդի արեւելյան ափի այս հին վայրի բնակիչները ապրել են հնագույն ժամանակներից եւ բազմաթիվ տարիների ընթացքում կուտակել են ավանդույթներ եւ դիցաբանություններ: Այնտեղ բնակվող նեոգաճաճ կենդանիների թվին են պատկանում գերմանացի ուսումնասիրողները: Մեկ տարվա ընթացքում նրանք հայտնաբերել են մոտ 25 միաեղջյուր եւ դրանց մեծ մասը նեոգաճաճ ձկներ: Նրանք չորացել են ամառային եւ գարնանային արեւադարձային սառնամանիքների ժամանակ եւ իրենց ձուկը հավաքել են գաճաճ միաեղջյուրների վրա:
```

## Choosing best parameters:

In general:
```min_length=500,
eos_token_id=5, 
pad_token=1,
do_sample=True,
top_k=0,
top_p=0.9,
temperature=0.9,
no_repeat_ngram_size=4
```


## Examples


#### mGPT Armenian Generation Examples
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/195i9264INAJDd4HqV5kfUC_-VafO0cu8?usp=sharing)

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

We welcome community contributions to the model, and celebrate both its inference and training technique enhancements

