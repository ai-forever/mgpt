# -*- coding: utf-8 -*-
from dataclasses import dataclass


@dataclass
class TaskConfig:
    cache_dir: str = "datasets_cache"
    batch_size: int = 4
    shot_nums: int = 0
    seq_length: int = 512


@dataclass
class XNLITaskConfig(TaskConfig):
    task_name: str = "XNLI"
    prompts = {
        "entailment": "entailment",
        "contradiction": "contradiction",
        "neutral": "neutral",
    }
    langs = [
        "ru",
        "en",
        "ar",
        "bg",
        "de",
        "el",
        "es",
        "fr",
        "hi",
        "sw",
        "th",
        "tr",
        "ur",
        "vi",
        "zh",
    ]
    prompt_components_per_lang = {
        "en": {
            "entailment": "Yes",
            "neutral": "Also",
            "contradiction": "No",
            "question": "right",
        },
        "ru": {
            "entailment": "Да",
            "neutral": "Также",
            "contradiction": "Нет",
            "question": "правда",
        },
        "ar": {
            "entailment": "نعم",
            "neutral": "ايضا",
            "contradiction": "رقم",
            "question": "حق",
        },
        "bg": {
            "entailment": "Да",
            "neutral": "Също",
            "contradiction": "Не",
            "question": "нали така",
        },
        "de": {
            "entailment": "Ja",
            "neutral": "Ebenfalls",
            "contradiction": "Nein",
            "question": "rechts",  # echt
        },
        "el": {  # греческий
            "entailment": "Ναί",
            "neutral": "Επίσης",
            "contradiction": "Οχι",
            "question": "σωστά",
        },
        "es": {
            "entailment": "Sí",
            "neutral": "también",
            "contradiction": "No",
            "question": "¿derecho",
        },
        "fr": {
            "entailment": "Oui",
            "neutral": "Aussi",
            "contradiction": "Non",
            "question": "à droite",
        },
        "hi": {
            "entailment": "हां",
            "neutral": "भी",
            "contradiction": "नहीं",
            "question": "अधिकार",
        },
        "sw": {
            "entailment": "Ndiyo",
            "neutral": "Pia",
            "contradiction": "Hapana",
            "question": "haki",
        },
        "th": {
            "entailment": "ใช่",
            "neutral": "อีกด้วย",
            "contradiction": "ไม่",
            "question": "ขวา",
        },
        "tr": {
            "entailment": "Evet",
            "neutral": "Ayrıca",
            "contradiction": "Numara",
            "question": "sağ",
        },
        "ur": {  # urdu
            "entailment": "جی ہاں",
            "neutral": "بھی",
            "contradiction": "نہیں",
            "question": "ٹھیک ہے",
        },
        "vi": {
            "entailment": "Đúng",
            "neutral": "Cũng thế",
            "contradiction": "Không",
            "question": "bên phải",
        },
        "zh": {
            "entailment": "是的",
            "neutral": "还",
            "contradiction": "不",
            "question": "对",
        },
    }


@dataclass
class PAWSXTaskConfig(TaskConfig):
    task_name: str = "PAWSX"
    prompts = {"0": "0", "1": "1"}
    prompt_components_per_lang = {
        "en": {"1": "Yes,", "0": "No,", "question": "right? "},
        "ja": {"1": "はい", "0": "いいえ", "question": ""},
        "de": {"1": "Ja,", "0": "Nein,", "question": "rechts? "},
        "ko": {"1": "네", "0": "아니다", "question": ""},
        "es": {"1": "Sí,", "0": "No,", "question": "¿derecho? "},
        "fr": {"1": "Oui,", "0": "Non,", "question": "à droite? "},
        "zh": {"1": "是的,", "0": "不,", "question": "对? "},
    }
    langs = ["en", "de", "es", "fr", "ja", "ko", "zh"]


@dataclass
class AMAZONTaskConfig(TaskConfig):
    task_name: str = "AMAZON"
    review_colm: str = "review_title"
    prompts = {
        1: "Rating 1. ",
        2: "Rating 2. ",
        3: "Rating 3. ",
        4: "Rating 4. ",
        5: "Rating 5. ",
    }
    langs = ["en", "de", "es", "fr", "ja", "zh"]


@dataclass
class XCOPATaskConfig(TaskConfig):
    task_name: str = "XCOPA"
    data_dir = "xcopa/data/"
    prompts_ques = {
        "et": {"cause": " sest ", "effect": " nõnda "},
        "ht": {"cause": " paske ", "effect": " konsa "},
        "it": {"cause": " perché ", "effect": " così "},
        "id": {"cause": " karena ", "effect": " jadi "},
        "qu": {"cause": " because ", "effect": " so "},
        "sw": {"cause": " kwa sababu ", "effect": " hivyo "},
        "zh": {"cause": " 因為 ", "effect": " 這樣 "},
        "ta": {"cause": " ஏனெனில் ", "effect": " அதனால் "},
        "th": {"cause": " เพราะ ", "effect": " ดังนั้น "},
        "tr": {"cause": " çünkü ", "effect": " böyle "},
        "vi": {"cause": " bởi vì ", "effect": " như thế "},
    }
    prompts = {0: "choice1", 1: "choice2"}
    langs = ["et", "ht", "it", "id", "qu", "sw", "zh", "ta", "th", "tr", "vi"]


@dataclass
class XWINOTaskConfig(TaskConfig):
    task_name: str = "XWINO"
    data_dir: str = "crosslingual_winograd/dataset.tsv"
    prompts = {0: "choice1", 1: "choice2"}
    langs = ["en", "jp", "pt", "ru"]
