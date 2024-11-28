import re
import json
import time

from loguru import logger
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
import networkx as nx
from random import choice, randrange
from sys2 import prompts
from sys2 import extract_prompts
from sys2.load_data import load_data
import os
import matplotlib.pyplot as plt
import numpy as np

class baseAgent:
    def __init__(self,chat,output=None):
        self.chat = chat
        self.output = output
        self.datas = load_data(self.output, format="dict")
        self.sys_template = ChatPromptTemplate.from_messages(
            [("human", prompts.base_sys_prompt)]
        )
    
    def reset(self, data):
        logger.info(f"Reset for new question.")
        self.data = data
        self.terminate = False
        self.sys_msg = self.sys_template.invoke({"question": data["question"]})
        self.answer = None
        
        self.reload(data)
        
        state = "reload"
        return state, self.terminate
    
    def reload(self,data):
        #从output的特定文件中进行匹配
        path = os.path.join(self.output, "output_dev.jsonl")
        self.datas = load_data(path, format="dict")
        idx = data["idx"]
        if idx in self.datas:
            logger.warning(
                f"Loading existing data from {self.output} with idx {idx}. Terminate early."
            )
            data = self.datas[idx]
            self.data = data
            self.terminate = True
        return        
 
    def policy(self,state):
        reply = self.chat.invoke(self.sys_msg)
        self.answer = reply.content
        self.terminate = True
        return None, self.terminate    
    
    


class cotAgent(baseAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sys_template = ChatPromptTemplate.from_messages(
            [("human", prompts.cot_sys_prompt)]
        )




class shotAgent(baseAgent):
    def __init__(self, chat, output=None, dataset_name, number_shots=3):
        super().__init__(chat, output)
        shots = self.load_shots(dataset_name, number_shots)
        s = prompts.shot_sys_prompt.replace("{few_shots}", shots)
        self.sys_template = ChatPromptTemplate.from_messages(
            [("human", s)]
        )
    def reset(self, data):
        logger.info(f"Reset for new question.")
        self.data = data
        self.terminate = False
        self.sys_msg = self.sys_template.invoke({"question": data["question"]})
        self.answer = None
        
        self.reload(data)
        
        state = "reload"
        return state, self.terminate

    def load_shots(self, dataset_name, number_shot):
        #todo: load examples 
        dataset = None

        shots = ""
        for idx, data in dataset:
            shot = "Question:"+data["question"] + "\n" +"Answer:"+data["answer"]
            if shots:
                shots = shots+"\n\n"+shot
            else:
                shots = shots+shot
            if idx > number_shot-2:
                break    
        return shots    
