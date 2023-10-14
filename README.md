# mGPT

### Multilingual Generative Pretrained Transformer
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://mit-license.org/)

We introduce mGPT, a multilingual variant of GPT-3, pretrained on 61 languages from linguistically diverse 25 language families using Wikipedia and C4 Corpus. We detail the design and pretraining procedure. The models undergo an intrinsic and extrinsic evaluation: language modeling in all languages, downstream evaluation on cross-lingual NLU datasets and benchmarks in 33 languages, and world knowledge probing in 23 languages. The in-context learning abilities are on par with the contemporaneous language models while covering a larger amount of languages, including underrepresented and low-resource languages of the Commonwealth of Independent States and the small peoples in Russia. The source code and the language models are available under the MIT license.

[[Paper]](https://arxiv.org/abs/2204.07580) [[Habr (Russian)]](https://habr.com/ru/company/sberdevices/blog/662195/) [[HugginFace mGPT-1.3B Model Card]](https://huggingface.co/sberbank-ai/mGPT)  [[HugginFace mGPT-13B Model Card]](https://huggingface.co/sberbank-ai/mGPT-13B) 
[[Papers With Code]](https://paperswithcode.com/paper/mgpt-few-shot-learners-go-multilingual)

 ### Setting up environment
`pip install -r requirements.txt`  

## Pretrain data
The model was pretrained on a 600Gb of texts, mainly from MC4 and Wikipedia. 
- [MC4](https://www.tensorflow.org/datasets/catalog/c4?hl=ru#c4multilingual)
- Wikipedia (version 20201101)

The Wikipedia texts are extracted from the dumps (v. 20201101) with WikiExtractor (Attardi, 2015).
Training data was deduplicated, and the text deduplication includes 64-bit hashing of each text in the corpus for keeping texts with a unique hash. We also filter the documents based on their text compression rate using zlib4. The most strongly and weakly compressing deduplicated texts are discarded.

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
        pad_token_id=1,
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
pad_token_id=1,
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
 Afrikaans (af), Arabic (ar), Armenian (hy), Azerbaijani (az), Basque (eu), Bashkir (ba), Belarusian (be), Bengali (bn), Bulgarian (bg), Burmese (my), Buryat (bxr), Chuvash (cv), Danish (da), English (en), Estonian (et), Finnish (fi), French (fr), Georgian (ka), German (de), Greek (el), Hebrew (he), Hindi (hi), Hungarian (hu), Indonesian (id), Italian (it), Japanese (ja), Javanese (jv), Kalmyk (xal), Kazakh (kk), Korean (ko), Kyrgyz (ky), Latvian (lv), Lithuanian (lt), Malay (ms), Malayalam (ml), Marathi (mr), Mongolian (mn), Ossetian (os), Persian (fa), Polish (pl), Portuguese (pt), Romanian (ro), Russian (ru), Spanish (es), Swedish (sv), Swahili (sw), Tatar (tt), Telugu (te), Thai (th), Turkish (tr), Turkmen (tk), Tuvan (tyv), Ukrainian (uk), Uzbek (uz), Vietnamese (vi), Yakut (sax), Yoruba (yo)

 ## Cite Us 

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

## Pretraining
[[mGPT-1.3B Model Card]](https://huggingface.co/ai-forever/mGPT)
[[mGPT-13B Model Card]](https://huggingface.co/ai-forever/mGPT-13B)

We utilize the DeepSpeed library and Megatron-LM. We pretrain our LMs with a total batch size of 2048 and the context window of 512 tokens. The total number of the training steps is 600k, and the models have seen $400$B tokens during pretraining. The pretraining took 14 days on a cluster of 256 V100 GPUs for mGPT-1.3B and 22 days on 512 V100 GPUs for mGPT-13B.

## Monoligual models:
[Habr article about the monoligual mGPT-1.3B models (Russian)](https://habr.com/ru/companies/sberdevices/articles/755108/)


Monolingual models on HuggingFace:
- [🇦🇲 mGPT-1.3B Armenian](https://huggingface.co/ai-forever/mGPT-1.3B-armenian)
- [🇦🇿 mGPT-1.3B Azerbaijan](https://huggingface.co/ai-forever/mGPT-1.3B-azerbaijan)
- [🍯 mGPT-1.3B Bashkir](https://huggingface.co/ai-forever/mGPT-1.3B-bashkir)
- [🇧🇾 mGPT-1.3B Belorussian](https://huggingface.co/ai-forever/mGPT-1.3B-belorussian)
- [🇧🇬 mGPT-1.3B Bulgarian](https://huggingface.co/ai-forever/mGPT-1.3B-bulgarian)
- [🌞 mGPT-1.3B Buryat](https://huggingface.co/ai-forever/mGPT-1.3B-buryat)
- [🌳 mGPT-1.3B Chuvash](https://huggingface.co/ai-forever/mGPT-1.3B-chuvash)
- [🇬🇪 mGPT-1.3B Georgian](https://huggingface.co/ai-forever/mGPT-1.3B-georgian)
- [🌸 mGPT-1.3B Kalmyk](https://huggingface.co/ai-forever/mGPT-1.3B-kalmyk)
- [🇰🇿 mGPT-1.3B Kazakh](https://huggingface.co/ai-forever/mGPT-1.3B-kazakh)
- [🇰🇬 mGPT-1.3B Kirgiz](https://huggingface.co/ai-forever/mGPT-1.3B-kirgiz)
- [🐻 mGPT-1.3B Mari](https://huggingface.co/ai-forever/mGPT-1.3B-mari)
- [🇲🇳 mGPT-1.3B Mongol](https://huggingface.co/ai-forever/mGPT-1.3B-mongol)
- [🐆 mGPT-1.3B Ossetian](https://huggingface.co/ai-forever/mGPT-1.3B-ossetian)
- [🇮🇷 mGPT-1.3B Persian](https://huggingface.co/ai-forever/mGPT-1.3B-persian)
- [🇷🇴 mGPT-1.3B Romanian](https://huggingface.co/ai-forever/mGPT-1.3B-romanian)
- [🇹🇯 mGPT-1.3B Tajik](https://huggingface.co/ai-forever/mGPT-1.3B-tajik)
- [☕ mGPT-1.3B Tatar](https://huggingface.co/ai-forever/mGPT-1.3B-tatar)
- [🇹🇲 mGPT-1.3B Turkmen](https://huggingface.co/ai-forever/mGPT-1.3B-turkmen)
- [🐎 mGPT-1.3B Tuvan](https://huggingface.co/ai-forever/mGPT-1.3B-tuvan)
- [🇺🇦 mGPT-1.3B Ukranian](https://huggingface.co/ai-forever/mGPT-1.3B-ukranian)
- [🇺🇿 mGPT-1.3B Uzbek](https://huggingface.co/ai-forever/mGPT-1.3B-uzbek)
- [💎 mGPT-1.3B Yakut](https://huggingface.co/ai-forever/mGPT-1.3B-yakut)


## Contributing
We welcome community contributions to the model, and celebrate both its inference and training technique enhancements.
