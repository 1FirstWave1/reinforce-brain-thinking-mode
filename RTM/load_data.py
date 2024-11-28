import json
import os
import numpy as np
from loguru import logger
import math
import re
import csv
from collections import Counter, defaultdict
from fractions import Fraction
from loguru import logger
import numpy as np
from pathlib import Path
import random

INVALID_ANS = "[invalid]"

    
def load_data(frn, format="list", mode="r"):
    """Load data from a file.
    :param frn (str): The dataset file name.

    :return: The dataset (a list of examples, each as a dictionary).
    """
    if not os.path.exists(frn):
        filename = Path(frn)
        filename.parent.mkdir(parents=True, exist_ok=True)
        return []
    if frn.endswith(".jsonl") or frn.endswith(".txt"):
        with open(frn, mode) as fr:
            rtns = [] if format == "list" else {}
            for i, line in enumerate(fr):
                if line.strip() == "":
                    continue
                try:
                    line = json.loads(line)
                    if format == "list":
                        rtns.append(line)
                    else:
                        idx = line["idx"]
                        rtns[idx] = line
                except json.decoder.JSONDecodeError as e:
                    print(f"Error in line {i}: {line}\n {e}")
                    exit(-1)
        return rtns
    elif frn.endswith(".csv"):
        with open(frn, encoding='utf-8') as fr:
            reader = csv.DictReader(fr)
            return [line for line in reader]
    elif frn.endswith(".json"):
        with open(frn, mode) as fr:
            rtns = json.load(fr)
        return rtns


class Dataset:
    def __init__(self, data_path) -> None:
        self.data = load_data(data_path)

    def __iter__(self):
        raise NotImplementedError

    def _parse_data(self, *args, **kwargs):
        raise NotImplementedError
    
class GSM8KDataset(Dataset):
    def __iter__(self):
        for idx, idata in enumerate(self.data):
            try:
                data = self._parse_data(idata)
                data["idx"] = idx
            except Exception as e:
                logger.error(e)
                continue
            yield idx, data

    def _parse_data(self, idata):
        question, answer = idata["question"], idata["answer"]
        ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
        #匹配
        match = ANS_RE.search(answer)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            gold_answer = int(match_str)
        else:
            gold_answer = INVALID_ANS
        return {"question": question, "answer": gold_answer}

class TruthfulQADataset(Dataset):
    def __iter__(self):
        for idx, idata in enumerate(self.data):
            try:
                data = self._parse_data(idata)
                data["idx"] = idx
            except Exception as e:
                logger.error(e)
                continue
            yield idx, data
    def _parse_data(self, idata):
        question, mc1_targets = idata["question"], idata["mc1_targets"]
        options = []
        for key, value in mc1_targets.items():
            options.append(key)
            if value == 1:
                correct_answer = key 
        random.shuffle(options)
        choice = [f"Opt{i}: {opt}" for i, opt in zip("ABCD", options)]
        option_text = "\n".join(choice)
        question = question + "\nOptions:\n" + option_text  
        for i,option in enumerate(options):
            if option == correct_answer:
                gold_answer = chr(i+65)
        #number_options用于后续确认选项个数
        return {"question": question, "answer": gold_answer, "number_options" : len(options)}

class GPQADataset(Dataset):

    def __iter__(self):
        for idx, idata in enumerate(self.data):
            try:
                data = self._parse_data(idata)
                data["idx"] = idx
            except Exception as e:
                logger.error(e)
                continue
            yield idx, data

    def _parse_data(self, idata):
        question, correct_answer = idata["Question"], idata["Correct Answer"]
        incorrect_answer1 = idata["Incorrect Answer 1"]
        incorrect_answer2,incorrect_answer3 = idata["Incorrect Answer 2"],idata["Incorrect Answer 3"]
        options = [correct_answer,incorrect_answer1,incorrect_answer2,incorrect_answer3]
        random.shuffle(options)
        choice = [f"Opt{i}: {opt}" for i, opt in zip("ABCD", options)]
        option_text = "\n".join(choice)
        question = question + "\nOptions:\n" + option_text  
        for i,option in enumerate(options):
            if option == correct_answer:
                gold_answer = chr(i+65)
        return {"question": question, "answer": gold_answer}   

DATASETS = {
    "GSM8K": GSM8KDataset,
    "TruthfulQA": TruthfulQADataset,
    "GPQA": GPQADataset,
}