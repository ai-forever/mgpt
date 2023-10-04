# -*- coding: utf-8 -*-
import json
import os
import pickle

import numpy as np
from datasets import Dataset, concatenate_datasets, load_dataset
from sklearn.metrics import accuracy_score
from evaluate.task import RuGPTEvaluationTask

from evaluate.mgpt_classification_configs import (
    XNLITaskConfig,
    PAWSXTaskConfig,
    AMAZONTaskConfig,
    XCOPATaskConfig,
    XWINOTaskConfig,
)

GROUP2TASK = {
    "classification": ["XNLI", "PAWSX", "AMAZON", "XCOPA", "XWINO"],
}


def merge_lang_datasets(benchmark, task_lang_name, split, langs, cache_dir):
    ds = []
    for lang in langs:
        dataset = load_dataset(
            benchmark,
            "{}{}".format(task_lang_name, lang),
            split=split,
            cache_dir=cache_dir,
        ).map(lambda example: {"language": lang})
        ds.append(dataset)
    ds = concatenate_datasets(ds)
    return ds


class Metrics:
    def __init__(self, task_name):
        self.task_name = task_name

    def calculate_metric(self, predictions, answers):
        if self.task_name in GROUP2TASK["classification"]:
            metrics = self.calculate_accuracies(answers, predictions)
        return metrics

    def calculate_accuracies(self, answers, predictions):
        langs = answers.keys()
        results = []
        for lang in langs:
            results.append([lang, accuracy_score(answers[lang], predictions[lang])])
        return results


class MultilingualClassificationTask(RuGPTEvaluationTask):
    def __init__(self, config):
        super().__init__(config)
        self.prompts = config.prompts
        self.langs = config.langs
        self.cache_dir = config.cache_dir
        self.task_name = config.task_name
        self.shot_nums = config.shot_nums
        self.seq_length = config.seq_length

    def verbalize_samples(self, lang, dataset, prompt):
        raise NotImplementedError

    def load_data(self, split):
        raise NotImplementedError

    def verbalize_train_examples(self, ds_train, ds_test, lang):
        num_rows = ds_test.num_rows
        ds_train = ds_train.add_column("idx_row", range(ds_train.num_rows))

        train_texts = []
        idx_texts = []
        for label, prompt in self.prompts.items():
            ds = ds_train.filter(lambda example: example["label"] == label)
            idx_texts.extend(ds["idx_row"])
            samples = self.verbalize_samples(lang, ds, prompt)
            train_texts.extend(samples)
        idx_texts = np.array(idx_texts)
        train_texts = np.array(train_texts)

        train_samples = []
        for row in range(num_rows):
            if self.task_name == "XWINO":
                idx_rows = np.where(idx_texts != row)[0]
            else:
                idx_rows = idx_texts
            sample = np.random.choice(
                idx_rows, min(len(idx_rows), self.shot_nums), replace=False
            )
            train_samples.append(train_texts[sample])

        if len(train_samples[0]) < self.shot_nums:
            print(f"few-shot examples: {len(train_samples[0])} (not {self.shot_nums})")
        return ["\n".join(sample) + "\n" for sample in train_samples]

    def calculate_scores(self, data, model):
        losses, _ = model.forward(
            data, loss_per_pos=True, batch_size=self.config.batch_size
        )
        return np.asarray([sum(loss) for loss in losses])

    def predict_subset(self, lang, dataset_lang, model):
        scores, labels = [], []
        prompts = self.prompts
        for i, prompt in prompts.items():
            labels.append(i)
            text_samples = self.verbalize_samples(lang, dataset_lang, prompt)
            if self.shot_nums > 0:
                samples = [
                    shot + sample
                    for shot, sample in zip(dataset_lang["shots"], text_samples)
                ]
                text_samples = []
                for sample in samples:
                    tokens = model.tokenizer.encode(sample)
                    if len(tokens) > self.seq_length:
                        tokens = tokens[-self.seq_length :]
                        text_sample = model.tokenizer.decode(tokens)
                        sample = text_sample.split("\n", maxsplit=1).pop()
                    text_samples.append(sample)
            print(lang, 'example: "' + text_samples[0] + '"')
            scores.append(self.calculate_scores(text_samples, model))

        scores = np.array(scores).T
        labels = np.array(labels)

        idx = np.argmin(scores, axis=1)
        pred_label = np.take_along_axis(labels, idx, axis=0)
        true_label = dataset_lang["label"]
        scores = np.concatenate((labels.reshape(1, -1), scores))

        return true_label, pred_label.tolist(), scores

    def predict(self, model):
        path_scores = None
        if self.shot_nums > 0:
            ds_train = self.load_data(split="validation")
        dataset = self.load_data(split="test")
        y_true = {}
        y_pred = {}
        for i, lang in enumerate(self.langs):
            print(i + 1, "/", len(self.langs), ":", lang)
            dataset_lang = dataset.filter(lambda example: example["language"] == lang)
            if self.shot_nums > 0:
                print("Preprocess few-shot examples")
                ds_train_lang = ds_train.filter(
                    lambda example: example["language"] == lang
                )
                train_examples = self.verbalize_train_examples(
                    ds_train_lang, dataset_lang, lang
                )
                dataset_lang = dataset_lang.add_column("shots", train_examples)

            true_label, pred_label, scores = self.predict_subset(
                lang, dataset_lang, model
            )
            y_true[lang] = true_label
            y_pred[lang] = pred_label
            print("%.3f" % accuracy_score(true_label, pred_label))
        return y_true, y_pred


class AMAZONTask(MultilingualClassificationTask):
    def __init__(self, config):
        super().__init__(config)
        self.review_colm = config.review_colm

    def verbalize_samples(self, lang, dataset, prompt):
        data = [prompt + text for text in dataset["text"]]
        return data

    def load_data(self, split):
        return (
            merge_lang_datasets(
                benchmark="amazon_reviews_multi",
                task_lang_name="",
                split=split,
                langs=self.langs,
                cache_dir=self.cache_dir,
            )
            .rename_column(self.review_colm, "text")
            .rename_column("stars", "label")
        )


class PAWSXTask(MultilingualClassificationTask):
    def __init__(self, config):
        self.prompt_components_per_lang = config.prompt_components_per_lang
        super().__init__(config)

    def verbalize_samples(self, lang, dataset, prompt):
        q = self.prompt_components_per_lang[lang]["question"]
        a = self.prompt_components_per_lang[lang][prompt]
        prompt = " " + q + a + " "
        data = []
        for sen1, sen2 in zip(dataset["sentence1"], dataset["sentence2"]):
            sen1 = sen1.strip()
            if sen1.endswith("."):
                sen1 = sen1[:-1] + ","
            sen1 = sen1.strip()
            line = sen1 + prompt + sen2
            line = (
                line.replace(" .", ".")
                .replace(" ,", ",")
                .replace(" '", "'")
                .replace(" n't", "n't")
            )
            data.append(line)
        return data

    def load_data(self, split):
        return merge_lang_datasets(
            benchmark="xtreme",
            task_lang_name="PAWS-X.",
            split=split,
            langs=self.langs,
            cache_dir=self.cache_dir,
        )


class XNLITask(MultilingualClassificationTask):
    def __init__(self, config):
        self.prompt_components_per_lang = config.prompt_components_per_lang
        super().__init__(config)

    def verbalize_samples(self, lang, dataset, prompt):
        q = self.prompt_components_per_lang[lang]["question"]
        a = self.prompt_components_per_lang[lang][prompt]
        prompt = " " + q + "? " + a + ", "
        data = []
        for sen1, sen2 in zip(dataset["sentence1"], dataset["sentence2"]):
            sen1 = sen1.strip()
            if sen1.endswith("."):
                sen1 = sen1[:-1] + ","
            sen1 = sen1.strip()
            line = sen1 + prompt + sen2
            line = (
                line.replace(" .", ".")
                .replace(" ,", ",")
                .replace(" '", "'")
                .replace(" n't", "n't")
            )
            data.append(line)
        return data

    def load_data(self, split):
        return (
            load_dataset("xtreme", "XNLI", split=split, cache_dir=self.cache_dir)
            .filter(lambda example: example["language"] in self.langs)
            .rename_column("gold_label", "label")
        )


class XCOPATask(MultilingualClassificationTask):
    def __init__(self, config):
        super().__init__(config)
        self.prompts_ques = config.prompts_ques
        self.data_dir = os.path.join(config.cache_dir, config.data_dir)

    def verbalize_samples(self, lang, dataset, prompt):
        dataset = dataset.map(
            lambda example: {
                "question": self.prompts_ques[example["language"]][example["question"]]
            }
        )
        spacer = "" if lang == "zh" else " "
        data = [
            premise.strip()
            + spacer
            + ques.strip().capitalize()
            + spacer
            + choice.strip().lower()
            for premise, ques, choice in zip(
                dataset["premise"], dataset["question"], dataset[prompt]
            )
        ]
        data = [d.replace(". ", " ").lower().capitalize() for d in data]
        return data

    def load_data(self, split):
        ds = []
        for lang in self.langs:
            ds.append(
                load_dataset(
                    "xcopa",
                    lang,
                    split=split,
                    ignore_verifications=True,
                    cache_dir=self.cache_dir,
                ).map(lambda example: {"language": lang})
            )
        ds = concatenate_datasets(ds)
        return ds


class XWINOTask(MultilingualClassificationTask):
    def __init__(self, config):
        super().__init__(config)
        self.data_dir = os.path.join(config.cache_dir, config.data_dir)

    def verbalize_samples(self, lang, dataset, prompt):
        data = dataset[prompt]
        return data

    def preprocess_sentence(self, sent, idxs, candidate, lang):
        processed_sent = sent.copy()
        processed_sent[idxs[0] : idxs[1]] = [c.lower() for c in candidate[2]]
        spacer = "" if lang in ["zh", "jp"] else " "
        result = (
            spacer.join(processed_sent)
            .replace(" .", ".")
            .replace(" ,", ",")
            .replace(" '", "'")
            .replace(" n't", "n't")
        )
        return result

    def load_data(self, split):
        with open(self.data_dir, encoding="utf-8") as ifh:
            xwino_dict = {}
            choice1 = []
            choice2 = []
            langs = []
            labels = []
            docs = []
            for line in ifh:
                chunks = line.strip().split("\t")

                lang = chunks[0]
                langs.append(lang)
                idxs = json.loads(chunks[5])[1]
                candidates = json.loads(chunks[6])
                sent_tokens = json.loads(chunks[4])
                docs.append(chunks[3])

                sents = []
                for i, candidate in enumerate(candidates):
                    if candidate[-1]:
                        labels.append(i)
                    processed_sent = self.preprocess_sentence(
                        sent_tokens, idxs, candidate, lang
                    )
                    sents.append(processed_sent)

                choice1.append(sents[0])
                choice2.append(sents[1])

            xwino_dict["language"] = langs
            xwino_dict["sentence"] = docs
            xwino_dict["choice1"] = choice1
            xwino_dict["choice2"] = choice2
            xwino_dict["label"] = labels

        return Dataset.from_dict(xwino_dict)


def evaluate_task(task_name, model, shots=0):
    if task_name == "XNLI":
        task = XNLITask(XNLITaskConfig(shot_nums=shots))

    elif task_name == "PAWSX":
        task = PAWSXTask(PAWSXTaskConfig(shot_nums=shots))

    elif task_name == "AMAZON":
        task = AMAZONTask(AMAZONTaskConfig(shot_nums=shots))

    elif task_name == "XCOPA":
        task = XCOPATask(XCOPATaskConfig(shot_nums=shots))

    elif task_name == "XWINO":
        task = XWINOTask(XWINOTaskConfig(shot_nums=shots))

    metric = Metrics(task_name)
    y_true, y_pred = task.predict(model)
    result = metric.calculate_metric(y_true, y_pred)
    return result
