from loguru import logger
import re
import math
from fractions import Fraction
from collections import Counter, defaultdict
import wandb
import numpy as np
import json
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain.prompts import ChatPromptTemplate

INVALID_ANS = "[invalid]"

def compute_answer_callback(final_answer, data, dataset_name, chat):
    logger.debug(final_answer)
    #抽取出模型所给出的答案
    if isinstance(final_answer, dict) and 'reply' in final_answer:
        final_answer = final_answer['reply']
    answer_default = extract_pred_answer(
        dataset_name, data, final_answer, chat 
    )    
    data.update(
        {"answer_default": answer_default,"final_answer": final_answer}
    )
    return data
def save_data(data, path):
    if path is None:
        return
    with open(path, "a+") as fout:
        data_json = json.dumps(data)
        fout.write(data_json + "\n")
        logger.info(f"Saved one line to {path}")
    return

def extract_pred_answer(dataset_name, data, pred_completion, chat, rounding="int", abs_val=True):
    """从回复中抽取预测回答
    :param dataset_name (str): The name of the dataset.
    :param pred_completion (str): The predicted completion.
    :param rounding (str): The rounding method for the predicted answer. Can be "int", "ceil", or "floor".
    :param abs_val (bool): Whether to take the absolute value of the predicted answer.

    :return: The predicted answer.
    """
    pred_answer = INVALID_ANS
    if INVALID_ANS in str(pred_completion):
        return INVALID_ANS
    #抽取数字（仍然有可能搞错）
    if dataset_name in ["GSM8K"]:
        option = ResponseSchema(name="final_answer", description="Extract the numerical (int or float) result.")             
        output_parser = StructuredOutputParser.from_response_schemas([option])
        format_instructions = output_parser.get_format_instructions()
        extract_format = """Next, a math question response will be provided. Please extract the numerical (int or float) result.
        
        Question: {question}
        Answer: {answer}

        {format_instructions}
        """
        prompt = ChatPromptTemplate.from_messages([extract_format])
        messages = prompt.invoke({"question":data["question"],"answer": pred_completion, "format_instructions": format_instructions})
        response = chat.invoke(messages)  
        try:
            output_dict = output_parser.parse(response.content)
        #无法提取就置为空
        except:
            output_dict ={"final_answer":""}
              
        if type(output_dict["final_answer"]) == int:
            pred_answer = output_dict["final_answer"]
        elif type(output_dict["final_answer"]) == str:
            ANS_RE = re.compile(r"(\-?[0-9\.\,]+)")
            match = ANS_RE.search(output_dict["final_answer"])
            if match:
                match_str = match.group(1).strip()
                match_str = match_str.replace(",", "")
                try:
                    pred_answer = str2num(match_str, rounding, abs_val)
                except:
                    pred_answer = INVALID_ANS
            else:
                pred_answer = INVALID_ANS
        return pred_answer

    elif dataset_name in ["GPQA", "TruthfulQA"]:
        if dataset_name == "TruthfulQA":
            op_str = ""
            for i in range(data["number_options"]-1):
                op_str += "Opt"+chr(i+65)+", " 
            op_str = "["+op_str+"Opt"+chr(data["number_options"]+65)+"]"
            description = "Extract the selected option from the response, and return only one option indicator from the list {op_str}."
            description = description.replace("{op_str}", op_str)
            option = ResponseSchema(name="correct_option", description=description)
            output_parser = StructuredOutputParser.from_response_schemas([option])
            format_instructions = output_parser.get_format_instructions()
            extract_format = """Next, a single-choice question response will be provided. Please return only one option indicator from the list {op_str}.
            
            Question: {question}
            Answer: {answer}

            {format_instructions}
            """  
            extract_format = extract_format.replace("{op_str}", op_str)
        elif dataset_name == "GPQA":
            option = ResponseSchema(name="correct_option", description="Extract the selected option from the response, and return only one option indicator from the list [OptA, OptB, OptC, OptD].")             
            output_parser = StructuredOutputParser.from_response_schemas([option])
            format_instructions = output_parser.get_format_instructions()
            extract_format = """Next, a single-choice question response will be provided. Please return only one option indicator from the list [OptA, OptB, OptC, OptD].
            
            Question: {question}
            Answer: {answer}

            {format_instructions}
            """
        prompt = ChatPromptTemplate.from_messages([extract_format])
        messages = prompt.invoke({"question":data["question"],"answer": pred_completion, "format_instructions": format_instructions})
        response = chat.invoke(messages)
        try:
            output_dict = output_parser.parse(response.content)
        except Exception as e:
            new_parser = OutputFixingParser.from_llm(parser=output_parser, llm=chat)
            output_dict = new_parser.parse(response.content)
        #小写转大写需不要添加？
        ANS_RE = re.compile(r"Opt([a-zA-Z])")
        finds = ANS_RE.findall(output_dict["correct_option"])
        try:
            answer = finds[-1]
        except:
            answer = INVALID_ANS
        return answer
    



def str2num(answer_str, rounding="int", abs_val=True):
    """Convert a string to a number.
    @:param answer_str (str): The string to convert.
    @:param rounding (str): 取整的方式 Can be "int", "ceil", or "floor".
    @:param abs_val (bool): 绝对值Whether to take the absolute value of the answer.

    @:return The converted number.
    """
    if "/" in answer_str:
        answer_str = float(sum(Fraction(s) for s in answer_str.split()))
    answer_str = float(answer_str)

    if rounding == "int":
        answer_str = int(answer_str)
    elif rounding == "ceil":
        answer_str = math.ceil(answer_str)
    elif rounding == "floor":
        answer_str = math.floor(answer_str)

    if abs_val:
        answer_str = abs(answer_str)

    return answer_str


#比较答案，应该是一些数据集特殊才会暂时写成这样（多选题）
def compare_results(answers, final_answer):
    #比较两个答案是否相等
    def compare(answer, final_answer):
        if isinstance(answer, str) and isinstance(final_answer, str):
            correctness = f"{answer}".strip() == f"{final_answer}".strip()
        else:
            try:
                correctness = np.abs(answer - final_answer) < 1e-6
            except Exception as e:
                logger.error(f"Exception: {e}\n Answer is {answer}")
                correctness = f"{answer}".strip() == f"{final_answer}".strip()
        return correctness
    #answer不是列表变成列表再调用
    if not isinstance(answers, list):
        return compare_results([answers], final_answer)[0]
    #原答案为列表，逐个跟final_answer比较？
    ret = [compare(answer, final_answer) for answer in answers]
    return ret

class BaseMetrics:
    #记录序号与准确率
    def __init__(self, **kwargs) -> None:
        self._idxs = []
        self._data = None
        self._correctnesses = []
        self.use_wandb = kwargs["use_wandb"]
        if self.use_wandb:
            #进行实验跟踪
            wandb.init(
                project="brain_net",
                # Track hyperparameters and run metadata
                config=kwargs,
            )
    @property
    def correctnesses(self):
        return np.array(self._correctnesses)

    @property
    def acc(self):
        return self.correctnesses.mean(axis=0)
    #返回字典，包含索引和acc
    @property
    def reports(self):
        reports = {
            "idx": self._data["idx"],
            "acc": self.acc,
        }
        return reports
    #对数据进行更新
    def update(self, data):
        idx = data["idx"]
        self._idxs.append(idx)
        #answer是真答案
        logger.debug(data)
        answer_default, answer = (
            data["answer_default"],
            data["answer"],
        )
        #布尔值
        correctness = compare_results(answers=answer_default, final_answer=answer)
        self._correctnesses.append(correctness)
        self._data = data
        return

    def log(self):
        if self.use_wandb:
            reports = self.reports
            wandb.log(reports)
        #记录debug级别信息
        logger.debug(self.reports)
        return