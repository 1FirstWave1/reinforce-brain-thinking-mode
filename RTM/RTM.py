#active设置有问题需改动
#提取的提示词也单独放一个文件中
#extract_information
#association问题
#602行报错
import re
import json
import time

from loguru import logger
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
import networkx as nx
from random import choice, randrange
#这种实现全部import出来是不是不太优雅
from sys2 import prompts
from sys2 import extract_prompts
import os
import matplotlib.pyplot as plt
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.output_parsers import OutputFixingParser
import numpy as np

COLORMAP = {
    "root": "yellow",
    "regeneration": "green",
    "subquestion": "green",
    "information": "red",
    "comment": "lightgreen",
    "never": "gray",
    "done": "purple",
    "item": "gray",
    "normal": "blue"
}
REVERSE_COLORMAP = {
    "yellow": "root",
    "green": "subquestion",
    "red": "information"
}


#唯一问题graph没办法动态生成，只能暂时用networkx
class sys2Agent:
    #G都得改成self.G
    def __init__(self, chat, output):
        self.chat = chat
        #输出路径
        self.output = output
        self.sys_template = ChatPromptTemplate.from_messages(
            [("system", prompts.SYS_PROMPTS)]
        )
        self.MSG = []

    def reset(self, data):
        logger.info(f"Reset for new question.")
        self.data = data
        self.G = nx.Graph()
        self.terminate = False
        #生成本轮最初信息
        self.sys_msg = self.sys_template.invoke({"text": data})
        self.G.add_node(
            "root",
            subtype=None,
            content=self.sys_msg,
            record=None,
            active=[],
            kwargs=None,
            confidence=True,
            color=COLORMAP["root"]
        )
        self.answer = None
        state = "overall", "TASK_RECOGNITION", ["root"], {"regeneration": False}
        return state, self.terminate

    def policy(self, state):
        type, subtype, index_list, kwargs = state
        if type == "overall":
            return self.policy_overall(subtype, index_list, kwargs)
        elif type == "decompose":
            return self.policy_decompose(subtype, index_list, kwargs)
        elif type == "reflection":
            return self.policy_reflection(subtype, index_list, kwargs)
        elif type == "association":
            return self.policy_association(subtype, index_list, kwargs)
        elif type == "compare":
            return self.policy_compare(subtype, index_list, kwargs)
        elif type == "importance":
            return self.policy_importance(subtype, index_list, kwargs)
        elif type == "inference":
            return self.policy_inference(subtype, index_list, kwargs)
        elif type == "difficulty":
            return self.policy_difficulty(subtype, index_list, kwargs)

    def policy_overall(self, subtype, index_list, kwargs):
        if subtype == "TASK_RECOGNITION":
            pre_index = index_list[0]
            regeneration = False
            if 'regeneration' in kwargs:
                regeneration = kwargs['regeneration']
            record = self.gain_record(subtype, pre_index, kwargs)
            confidence = self.confidence_metrics(record, ["root"])
            #提取分类结果
            node = self.extract_information(record, ["type of task"], ["Based on the reply, specify the type of task."])
            cur_index = self.deal_node_generation(regeneration, subtype, node, record, index_list, kwargs, confidence)
            state = "decompose", "DECOMPOSE_TASK", ["root"], {"problem": self.G.nodes["root"]["content"], "regeneration": False}
            state = self.confidence_redirect(confidence, cur_index, state)
            return state, self.terminate
        #得到最终回答！ 
        elif subtype == "ANSWER":
            pre_index = index_list[0]
            record = self.gain_record(subtype, pre_index, kwargs)
            node = self.extract_information(record, ["final_answer"], ["Extract the final answer.Note that you only need to extract the final answer, no additional information is required"])
            pre_index = "root"
            count = self.count_node(subtype.lower())
            node_info = subtype, node, record, [pre_index], subtype, kwargs, None
            cur_index = self.draw_graph(f"{count}{subtype.lower()}", node_info)
            colors = nx.get_node_attributes(self.G, 'color').values()
            nx.draw(self.G, with_labels=True, node_color=colors)
            path_1 = os.path.join(self.output,
                                  time.strftime("graph_%Y_%m_%d_%H_%M_%S.png", time.localtime(time.time())))
            plt.savefig(path_1)
            path_2 = os.path.join(self.output,
                                  time.strftime("test_txt_%Y_%m_%d_%H_%M_%S.txt", time.localtime(time.time())))
            with open(path_2, "w") as f:
                for msg in self.MSG:
                    f.write(msg + "\n")
            state = None
            self.terminate = True
            self.answer = node["final_answer"]
            return state, self.terminate

    #kwargs作为prompt的参数，也不一定非要在上一层指定的感觉，只是传递一个信息吧。。。。
    def policy_decompose(self, subtype, index_list, kwargs):
        if subtype == "DECOMPOSE_TASK":
            pre_index = index_list[0]
            regeneration = False
            if 'regeneration' in kwargs:
                regeneration = kwargs['regeneration']
            record = self.gain_record(subtype, pre_index, kwargs)
            confidence = self.confidence_metrics(record, index_list)
            #统一都作为字符串表达式存入node中
            node = self.extract_information(record, ["judgment"], ["Based on the reply, does this question need to be broken down into several sub-questions to be solved? Answer True if yes, False if not or unknown."])
            judgment = node["judgment"] == "True" or node["judgment"] == True
            if judgment:
                # this intends to make the steps attach to the root node
                num_dict = self.extract_information(record, ["number_step"], [
                    "Into how many steps can the question be broken down? Just give a number."])
                number_step = int(num_dict["number_step"])
                node.update(num_dict)
                objects = [f'step{i + 1}' for i in range(number_step)]
                descriptions = [f'what should we do in the step{i + 1}' for i in range(number_step)]
                steps = self.extract_information(record, objects, descriptions)
                #字典的更新
                node.update(steps)
                cur_index = self.deal_node_generation(regeneration, subtype, node, record, index_list, kwargs, confidence)
                state = "decompose", "STEP", [cur_index], {"i": 1, "regeneration": False}
                state = self.confidence_redirect(confidence, cur_index, state)
                return state, self.terminate
            else:
                cur_index = self.deal_node_generation(regeneration, subtype, node, record, index_list, kwargs, confidence)
                state = "overall", "ANSWER", [cur_index], {"regeneration": False}
                state = self.confidence_redirect(confidence, cur_index, state)
                return state, self.terminate

        # elif subtype == "FIRST_STEP":
        #     pre_index = index_list[0]
        #     record = self.gain_record(subtype, index, kwargs)
        #     node = self.extract_information(record, ["the solution of step1"], ["the solution of step1"])
        #     index = "root"
        #     count = self.count_node(subtype.lower())
        #     index = self.draw_graph(f"{count}{subtype.lower()}", subtype, node, record, [index], subtype)

        #     confidence = self.confidence_metrics(record)
        #     if confidence:
        #         state = "decompose", "STEP", [index], {"i": 1, "regeneration":False}
        #         return state, self.terminate
        #     else:
        #         next_type, next_task, next_task_kwargs = self.not_sure_method(index)
        #         next_task_kwargs["regeneration"] = False
        #         state = next_type, next_task, [index], next_task_kwargs
        #         return state, self.terminate

        elif subtype == "STEP":
            pre_index = index_list[0]
            regeneration = False
            if 'regeneration' in kwargs:
                regeneration = kwargs['regeneration']
            record = self.gain_record(subtype, pre_index, kwargs)
            confidence = self.confidence_metrics(record,self.G.nodes[pre_index]["active"])
            i = kwargs["i"]
            node = self.extract_information(record, [f"the solution of step{i}"], [f"the solution of step{i}"])
            num_list = re.findall(r"\d+", pre_index)
            num_list = [int(x) for x in num_list]
            count = num_list[0]
            if regeneration:
                cur_index = pre_index
                self.regenerate_node(cur_index, record, node,
                                     {**kwargs,
                                      'answer1': self.G.nodes[cur_index]['content'],
                                      'answer2': node,
                                      'regeneration': False
                                      })
            else:
                #active与上一个节点相一致
                node_info = subtype, node, record, [pre_index], subtype, kwargs, confidence
                cur_index = self.draw_graph(f"{count}step{i}", node_info, self.G.nodes[pre_index]["active"])
            if confidence == "sure" or confidence == True:
                if i + 1 < int(self.G.nodes[f"{count}decompose_task"]["content"]["number_step"]):
                    state = "decompose", "STEP", [cur_index], {"i": i + 1, "regeneration": False}
                    return state, self.terminate
                else:
                    if self.G.nodes[cur_index]["active"][-1] == "root":
                        state = "overall", "ANSWER", [cur_index], {"regeneration": False}
                        return state, self.terminate
                    else:
                        state = self.back_to_top(self.G.nodes[cur_index]["active"][-1])
                        return state, self.terminate
            else:
                next_type, next_task, next_task_kwargs = self.not_sure_method(cur_index)
                next_task_kwargs["regeneration"] = False
                next_task_kwargs["i"] = i + 1
                state = next_type, next_task, [cur_index], next_task_kwargs
                return state, self.terminate

        # elif subtype == "DECOMPOSE_ITEM":
        #     #该方法禁调用not_sure_method
        #     pre_index = index_list[0]
        #     record = self.gain_record(subtype, index, kwargs)
        #     #这样写不好，直接改成用符号分割，正则表达式匹配吧。。。。
        #     num_dict = self.extract_information(record, ["number_item"], [
        #         "Based on the answer, how many items are decomposed? Just give a number."])
        #     number_item = int(num_dict["number_item"])
        #     objects = [f'item{i}' for i in range(number_item)]
        #     descriptions = [f'what should we do in the step{i}' for i in range(number_item)]
        #     steps = self.extract_information(record, objects, descriptions)
        #     # 字典的更新
        #     node = self.extract_information(record, ["important_item"],
        #                                     ["What is the most important item in this question?"])
        #     node.update(steps)
        #     count = self.count_node(subtype.lower())
        #     center_index = index
        #     index = self.draw_graph(f"{count}{subtype.lower()}",
        #                             subtype, node, record, [index],
        #                             subtype, active=center_index)

        #     # 生成这些点并判断激活程度
        #     for key, value in steps:
        #         node = {key: value}
        #         count = self.count_node(subtype.lower())
        #         current_index = self.draw_graph(f"{count}{subtype.lower()}", subtype, node, record, [index],
        #                                         subtype,
        #                                         active=[center_index])

        #     subtype = self.G.nodes[index_list[0]]["subtype"]
        #     type = self.gain_type(subtype)
        #     state = type, subtype, [index_list[0]], {"regeneration": True}
        #     return state, self.terminate

    #下面很多内容未设置"regeneration"参数，同时返回原任务所需kwargs缺失
    def policy_association(self, subtype, index_list, kwargs):
        regeneration = False
        if 'regeneration' in kwargs:
            regeneration = kwargs['regeneration']
        if subtype == "ASSOCIATION":
            '''input:kwargs{item : 需要联想的问题或实体}
               output:多个相似的节点与index直接连接
               联想本身不调用redirect（这个没法调而且相似没啥正确与否）'''
            pre_index = index_list[0]
            record = self.gain_record(subtype, pre_index, kwargs)
            confidence = self.confidence_metrics(record, index_list)
            number = self.extract_information(record, ["number_associate_item"], [
                "How many items are mentioned related to the main item? Just give a number."])

            number_associate_item = int(number["number_associate_item"])
            objects = [f'item{i}' for i in range(number_associate_item)]
            #可再生成一下对应的reason
            descriptions = [f'Write the name of item{i} in order.' for i in range(number_associate_item)]
            items = self.extract_information(record, objects, descriptions)

            # count = self.count_node(subtype.lower())
            # center_index = index
            # index = self.draw_gragh(f"{count}{subtype.lower()}" + "_task", subtype, node, record, [index],
            #                         "ASSOCIATION_TASK", active=center_index)
            #生成这些点并判断激活程度
            for key, value in items.items():
                node = {key: value}
                count = self.count_node(subtype.lower())
                node_info = subtype, node, record, [pre_index], subtype, kwargs, confidence
                #每个点都储存了本次产生的record，active需要修改
                cur_index = self.draw_graph(f"{count}{subtype.lower()}", node_info, active=[pre_index])
                argument = {"problem": self.G.nodes[pre_index]["record"]["ask"], "item": value}
                argument.update(kwargs)
                judgment = self.policy_association("HELP_JUDGMENT", cur_index, argument)
                if not judgment:
                    self.G.nodes[cur_index]["active"].remove(pre_index)
            #此处有问题记得修改
            state = self.back_to_top(pre_index)
            state = self.confidence_redirect(confidence, cur_index, state, center_index=pre_index)
            # 生成辅助信息返回
            return state, self.terminate
            # #根据相似事物生成新的原节点答案 
            # type ,subtype = self.node_type(G[index])
            # record = self.send_and_record(subtype,node)
            # answer1 = G[index]["content"]
            # answer2 = record["reply"]
            # kwargs = {"answer1":answer1,"answer2":answer2}
            # #获得两个答案的区别
            # differences = self.policy_compare("DIFFERENCE_ANSWER",index,G,kwargs)
            # #选择答案
            # better_answer,reason = self.policy_compare("CHOOSE_ANSWER",index,G,differences)
            # G[index]["content"] = better_answer
        elif subtype == "SIMILAR_PROBLEM":
            pre_index = index_list[0]
            record = self.gain_record(subtype, pre_index, kwargs)
            confidence = self.confidence_metrics(record, index_list)
            node = self.extract_information(record, ["example", "answer"],
                                            ["the example of similar question", "the correspond answer"])
            cur_index = self.deal_node_generation(regeneration, subtype, node, record, index_list, kwargs, confidence)
            state = self.back_to_top(pre_index)
            state = self.confidence_redirect(confidence, cur_index, state)
            return state, self.terminate

    #compare还需要大改
    def policy_compare(self, subtype, index_list, kwargs):
        regeneration = False
        if 'regeneration' in kwargs:
            regeneration = kwargs['regeneration']
        if subtype == "COMPARE":
            # 暂定为两件事
            # TODO: expecting two things in kwargs
            pre_index = index_list[0]
            record = self.gain_record(subtype, pre_index, kwargs)
            confidence = self.confidence_metrics(record, index_list)
            node = self.extract_information(record, ["similarities", "differences"],
                                            ["the similarities between two things",
                                             "the differences between two things"])
            cur_index = self.deal_node_generation(regeneration, subtype, node, record, index_list, kwargs, confidence)
            kwargs = {"differences": node["differences"], "problem": kwargs["problem"], "regeneration": False}
            state = "compare", "DIFFERENCE_IMPACT", [cur_index], kwargs
            state = self.confidence_redirect(confidence, cur_index, state)
            return state, self.terminate
        elif subtype == "COMPARE_ORDINARY":
            pre_index = index_list[0]
            record = self.gain_record(subtype, pre_index, kwargs)
            confidence = self.confidence_metrics(record, index_list)
            node = self.extract_information(record, ["differences"],
                                            ["the differences between usual tasks and this specific task"])
            cur_index = self.deal_node_generation(regeneration, subtype, node, record, index_list, kwargs, confidence)
            kwargs = node.update({"problem": kwargs["problem"], "regeneration": False})
            state = "compare", "DIFFERENCE_IMPACT", [cur_index], kwargs
            state = self.confidence_redirect(confidence, cur_index, state)
            return state, self.terminate
        elif subtype == "DIFFERENCE_IMPACT":
            pre_index = index_list[0]
            #active是其父节点
            active = self.G.nodes[pre_index]["active"]
            record = self.gain_record(subtype, pre_index, kwargs)
            confidence = self.confidence_metrics(record, active)
            node = self.extract_information(record, ["impact"],
                                            ["What impact do these differences have on the problem"])
            cur_index = self.deal_node_generation(regeneration, subtype, node, record, index_list, kwargs, confidence, active=active)
            state = self.back_to_top(active[0])
            state = self.confidence_redirect(confidence, cur_index, state)
            #生成辅助信息返回
            return state, self.terminate
        elif subtype == "DIFFERENCE_ANSWER":
            pre_index = index_list[0]
            record = self.gain_record(subtype, pre_index, kwargs)
            node = self.extract_information(record, ["differences"], ["the differences between answer1 and answer2"])
            kwargs.update(node)
            kwargs["regeneration"] = False
            return self.policy_compare("CHOOSE_ANSWER", index_list, kwargs)
        # 当然我们也可以通过”自信度指标来做尝试
        elif subtype == "CHOOSE_ANSWER":
            pre_index = index_list[0]
            record = self.gain_record(subtype, pre_index, kwargs)
            node = self.extract_information(record, ["better_answer"],
                                            ["Which answer is better under this question? Answer answer1 if answer1 is better, answer2 if answer2 is better."])
            #容易抽风可以适当添加鲁棒性
            return node

    def policy_importance(self, subtype, index_list, kwargs):
        regeneration = False
        if 'regeneration' in kwargs:
            regeneration = kwargs['regeneration']
        if subtype == "IMPORTANCE":
            ''''获得某一项中最重要的事物（感觉prompt）可以改的更好一些'''
            pre_index = index_list[0]
            record = self.gain_record(subtype, pre_index, kwargs)
            # 先按有一项重要事务来写吧
            node = self.extract_information(record, ["important_item"],
                                            ["What is the most important item in this question?"])
            cur_index = self.deal_node_generation(regeneration, subtype, node, record, index_list, kwargs, None)
            state = self.back_to_top(pre_index)
            return state, self.terminate
        #看最后需要加入与否
        elif subtype == "CONDENSE_TEXT":
            ''''对文本中的无关信息进行删减，目前只感觉可以跟根节点相连
                输入：一段文本
                输出：压缩过文本
                不生成节点'''
            pre_index = index_list[0]
            record = self.gain_record(subtype, pre_index, kwargs)
            node = self.extract_information(record, ["condensed_text"], ["Give the condensed version of the text."])
            return node["condensed_text"]
        # 将非重点的点设为失活的
        elif subtype == "UNIMPORTANT_POINT":
            '''两种使用方式：1、index_list只传入一个index时，判断active为该index的点哪些是不必须的；
            2、index_list多个参数时，判断这些点关于第一个点问题哪些是不必须的'''
            pre_index = index_list[0]
            index_list.remove(pre_index)
            node_text = ""
            i = 0
            node_number_dict = {}
            if len(index_list):
                for node in index_list:
                    i += 1
                    node_number_dict[i] = node
                    node_str_list = [f"{name}:{value}" for name, value in self.G.nodes[node]["content"].items()]
                    result = " ".join(node_str_list)
                    result = "node number{i}:" + result + "\n"
                    node_text += result
            else:
                for node in self.G.nodes():
                    if (pre_index in self.G.nodes[node]["active"]):
                        i += 1
                        node_number_dict[i] = node
                        node_str_list = [f"{name}:{value}" for name, value in self.G.nodes[node]["content"].items()]
                        result = " ".join(node_str_list)
                        result = "node number{i}:" + result + "\n"
                        node_text += result
            if pre_index == "root":
                problem = self.data
            else:
                problem = self.G.nodes[pre_index]["record"]["ask"]
            dict_node = {"node_text": node_text, "problem": problem}
            record = self.gain_record(subtype, pre_index, dict_node)
            #用序号来回答哪些不重要
            node = self.extract_information(record, ["irrelevant_point"], [
                "What are irrelevant nodes in the answer? Just give the number of the node."])
            #此处直接正则表达式提取了
            num_list = re.findall(r"\d+", pre_index)
            num_list = [int(x) for x in num_list]
            node_list = [node_number_dict[j+1] for j in range(i) if j+1 not in num_list]
            for node in node_list:
                self.G.nodes[node]["active"].remove(pre_index)
            #只进行一次active的判断，并不返回值
            return
        elif subtype == "HELP_JUDGMENT":
            pre_index = index_list[0]
            record = self.gain_record(subtype, pre_index, kwargs)
            judgment = self.extract_information(record, ["judgment"], ["Based on the reply,is it helpful in solving the problem?\
                                                                     Answer True if yes, False if not or unknown."])
            return judgment["judgment"] == "True"

    def policy_inference(self, subtype, index_list, kwargs):
        regeneration = False
        if 'regeneration' in kwargs:
            regeneration = kwargs['regeneration']
        if subtype == "REASON":
            ''''该策略获取某节点陈述的可能原因并与其相连接'''
            pre_index = index_list[0]
            record = self.gain_record(subtype, pre_index, kwargs)
            confidence = self.confidence_metrics(record, index_list)
            node = self.extract_information(record, ["reason"], ["The reasons mentioned in the answer"])
            cur_index = self.deal_node_generation(regeneration, subtype, node, record, index_list, kwargs, confidence)
            state = self.back_to_top(pre_index)
            state = self.confidence_redirect(confidence, cur_index, state)
            return state, self.terminate
        elif subtype == "DEFINE":
            ''''获得对应事物的定义并与其相连
                输入:thing(需要定义的事物)
                输出:跳转到提问节点'''
            pre_index = index_list[0]
            record = self.gain_record(subtype, pre_index, kwargs)
            confidence = self.confidence_metrics(record, index_list)
            node = self.extract_information(record, ["definition"], ["The definition mentioned in the answer"])
            cur_index = self.deal_node_generation(regeneration, subtype, node, record, index_list, kwargs, confidence)
            state = self.back_to_top(pre_index)
            state = self.confidence_redirect(confidence, cur_index, state)
            return state, self.terminate
        elif subtype == "RESULT":
            '''获得对应事物的结果并与其相连(当然感觉结果和影响可以分成两个策略来使用)
                输入：thing（需被探究造成结果的概念）
                输出：跳转到提问节点 '''
            pre_index = index_list[0]
            record = self.gain_record(subtype, pre_index, kwargs)
            confidence = self.confidence_metrics(record, index_list)
            node = self.extract_information(record, ["impact_or_outcome"],
                                            ["The impact or outcome mentioned in the answer"])
            cur_index = self.deal_node_generation(regeneration, subtype, node, record, index_list, kwargs, confidence)
            state = self.back_to_top(pre_index)
            state = self.confidence_redirect(confidence, cur_index, state)
            return state, self.terminate
            #反事实：若不发生会产生怎样的结果
        elif subtype == "COUNTER_FACTUAL1" or "COUNTER_FACTUAL2":
            ''' 某一因素相反或不存在对结果的影响
                输入：thing（影响），result原本结果
                输出：跳转到提问节点 '''
            pre_index = index_list[0]
            record = self.gain_record(subtype, pre_index, kwargs)
            confidence = self.confidence_metrics(record, index_list)
            node = self.extract_information(record, ["impact"], ["The impact mentioned in the answer"])
            cur_index = self.deal_node_generation(regeneration, subtype, node, record, index_list, kwargs, confidence)
            state = self.back_to_top(pre_index)
            state = self.confidence_redirect(confidence, cur_index, state)
            return state, self.terminate

    def confidence_metrics(self, record, active):
        i = 0
        if active:
            while active[0] != "root":
                i += 1
                active = self.G.nodes[active[0]]["active"]
            reply = record["original_reply"]
            logprobs = [token["logprob"] for token in reply.response_metadata["logprobs"]["content"]]
            perplexity_score = np.exp(-np.mean(logprobs))
        limit = 1.2 + i*0.1
        if perplexity_score < limit:
            return True
        else:
            return False

    def choose_policy(self, list):
        '''选择一个方案
            输入：一个包含各种策略的列表
            输出：给定的某个策略'''
        pass

    def deal_node_generation(self, regeneration, subtype, node, record, index_list, kwargs, confidence, active=None, color="normal"):
        if active is None:
            active = index_list
        if regeneration:
            cur_index = index_list[0]
            # 重生成的节点哪些属性有可能改动？
            self.regenerate_node(cur_index, record, node,
                                 {**kwargs,
                                  'answer1': json.dumps(self.G.nodes[cur_index]['content']),
                                  'answer2': json.dumps(node),
                                  'regeneration': False
                                  })
        else:
            count = self.count_node(subtype.lower())
            node_info = subtype, node, record, index_list, subtype, kwargs, confidence
            cur_index = self.draw_graph(f"{count}{subtype.lower()}", node_info, active, color)
        return cur_index

    def confidence_redirect(self, confidence, index, state, center_index=None):
        if not center_index:
            center_index = self.G.nodes[index]["active"][0]
        if confidence:
            candidate = [center_index]
            for node in self.G.nodes():
                if (index in self.G.nodes[node]["active"]):
                    self.G.nodes[node]["active"].append(center_index)
                    candidate.append(node)
            if len(candidate) >2:
                #是否应继承上文
                self.policy_importance("UNIMPORTANT_POINT", candidate, {})
            return state
            #不确定则转其他信息获取，并在获取完成后重复循环
        else:
            #深度优先原则,也有可能confidence是None
            if self.G.nodes[center_index]["confidence"]:
                next_type, next_task, next_task_kwargs = self.not_sure_method(index)
                next_task_kwargs["regeneration"] = False
                state = next_type, next_task, [index], next_task_kwargs
                return state
            else:
                return self.back_to_top(center_index)

    def back_to_top(self, index):
        subtype = self.G.nodes[index]["subtype"]
        type = self.gain_type(subtype)
        back_dict = self.G.nodes[index]["kwargs"]
        back_dict["regeneration"] = True
        state = type, subtype, [index], back_dict
        return state

    def regenerate_node(self, index, record, node, kwargs):
        judgment = self.policy_compare("DIFFERENCE_ANSWER", [index], kwargs)
        #防止字符串抽风 eg. “answer1better"
        if 'answer2' in judgment["better_answer"]:
            self.G.nodes[index]['record'] = record
            self.G.nodes[index]['content'] = node
            self.G.nodes[index]['color'] = COLORMAP['regeneration']
        elif 'answer1' in judgment["better_answer"]:
            pass
        else:
            print('Warning::RegenerateNode: neither answer or answer2 is returned')
            self.G.nodes[index]['record'] = record
            self.G.nodes[index]['content'] = node

    def not_sure_method(self, index, node=None):
        decompose = ["DECOMPOSE_TASK"]
        association = ["ASSOCIATION", "SIMILAR_PROBLEM"]
        compare = ["COMPARE", "COMPARE_ORDINARY"]
        importance = ["IMPORTANCE"]
        inference = ["REASON", "DEFINE", "RESULT", "COUNTER_FACTUAL1", "COUNTER_FACTUAL2"]
        seq = [decompose, association, compare, importance, inference]


        for node in self.G.nodes():
            node_dict = self.G.nodes[node]
            for subtype_list in seq:
                if (node_dict["active"] == index) & (node_dict["subtype"] in subtype_list):
                    subtype_list.remove(node_dict["subtype"])

        thing_list = []
        for node in self.G.neighbors(index):
            if self.G.nodes[node]["subtype"] == "ASSOCIATION" or False:
                for key, value in self.G.nodes[node]["content"].items():
                    thing_list.append(value)
        if len(thing_list) < 1:
            seq[4].remove("COUNTER_FACTUAL1")
            seq[4].remove("COUNTER_FACTUAL2")
        if len(thing_list) < 2:
            seq[2].remove("COMPARE")            

        choice_number = randrange(len(seq))
        candidate = choice(seq[choice_number])
        if candidate == "DECOMPOSE_TASK":
            return "decompose", "DECOMPOSE_TASK", {"problem": self.G.nodes[index]["content"], "regeneration": False}
        elif candidate == "ASSOCIATION":
            return "association", "ASSOCIATION", {"item": self.G.nodes[index]["content"]}
        elif candidate == "SIMILAR_PROBLEM":
            return "association", "SIMILAR_PROBLEM",  {"problem": self.G.nodes[index]["content"], "regeneration": False}
        elif candidate == "COMPARE":
            # TODO: 如果thing不够也可以考虑主动调用生成
            nodes = self.G.neighbors(index)
            thing1 = choice(thing_list)
            thing_list.remove(thing1)
            thing2 = choice(thing_list)
            return "compare", "COMPARE", {"thing1": thing1, "thing2": thing2, "regeneration": False}
        elif candidate == "COMPARE_ORDINARY":
            return "compare", "COMPARE_ORDINARY", {"problem": self.G.nodes[index]["content"], "regeneration": False}
        elif candidate == "IMPORTANCE":
            return "importance", "IMPORTANCE", {"problem": self.G.nodes[index]["content"], "regeneration": False}
        elif candidate in ["REASON", "DEFINE", "RESULT"]:
            return "inference", candidate, {"thing": self.G.nodes[index]["content"], "regeneration": False}
        elif candidate in ["COUNTER_FACTUAL1", "COUNTER_FACTUAL2"]:
            #需要有一个thing
            thing = choice(thing_list)
            return "inference", candidate, {"thing": thing, "result": self.G.nodes[index]["record"]["reply"], "regeneration": False}

    def extract_information(self, record, objects, descriptions, model=None):
        schemas = []
        for item, description in zip(objects, descriptions):
            schemas.append(ResponseSchema(name=item, description=description))
        output_parser = StructuredOutputParser.from_response_schemas(schemas)
        format_instructions = output_parser.get_format_instructions()
        extract_format = """Next, a segment of Q&A text will be provided. Please extract information according to the following format.
        
        chatting records = {text}

        {format_instructions}
        """
        prompt = ChatPromptTemplate.from_messages([extract_format])

        text = f"ask:{record['ask']} \n reply:{record['reply']}"
        messages = prompt.invoke({"text": text, "format_instructions": format_instructions})
        response = self.chat.invoke(messages)
        try:
            output_dict = output_parser.parse(response.content)
        except Exception as e:
            new_parser = OutputFixingParser.from_llm(parser=output_parser, llm=self.chat)
            output_dict = new_parser.parse(response.content)

        #输出的是一个字典
        return output_dict

    def gain_type(self, subtype):
        overall = ["ANSWER", "TASK_RECOGNITION", "TASK_DIFFICULTY"]
        decompose = ["DECOMPOSE_TASK", "STEP"]
        association = ["ASSOCIATION", "SIMILAR_PROBLEM"]
        compare = ["COMPARE", "COMPARE_ORDINARY", "DIFFERENCE_IMPACT", "DIFFERENCE_ANSWER", "CHOOSE_ANSWER"]
        importance = ["IMPORTANCE", "CONDENSE_TEXT", "UNIMPORTANT_POINT"]
        inference = ["REASON", "DEFINE", "RESULT", "COUNTER_FACTUAL1", "COUNTER_FACTUAL2"]
        if subtype in overall:
            return "overall"
        elif subtype in decompose:
            return "decompose"
        elif subtype in association:
            return "association"
        elif subtype in compare:
            return "compare"
        elif subtype in importance:
            return "importance"
        elif subtype in inference:
            return "inference"

    def count_node(self, name):
        count = sum(name in node for node in self.G.nodes())
        return count

    def gain_record(self, subtype, index, kwargs):
        regeneration = False
        if "regeneration" in kwargs:
            regeneration = kwargs["regeneration"]
            del kwargs["regeneration"]
        if regeneration:
            kwargs["index"] = index
        record = self.send_and_record(subtype, regeneration, **kwargs)
        return record

    def send_and_record(self, subtype, regeneration=False, **kwargs):
        if regeneration:
            additional_msg = self.create_tips([kwargs["index"]])
            #这样会删除原函数的参数么？
            del kwargs["index"]
        else:
            additional_msg = self.create_tips()
        if kwargs:
            #暂时这么写吧,但愿不报错
            subtype_template = ChatPromptTemplate.from_messages([("human", getattr(prompts, subtype))])
            subtype_temp_msg = subtype_template.invoke(kwargs)
            subtype_temp_msg = subtype_temp_msg.messages

        else:
            subtype_temp_msg = [HumanMessage(content=getattr(prompts, subtype))]

        if additional_msg:
            tip_msg = [SystemMessage(content=additional_msg)]
            subtype_msg = self.sys_msg.messages + tip_msg + subtype_temp_msg
        else:
            subtype_msg = self.sys_msg.messages + subtype_temp_msg
        reply = self.chat.invoke(subtype_msg, logprobs=True)
        qs = ""
        for rd in subtype_msg:
            qs += rd.content
        self.MSG.append(qs)
        self.MSG.append(reply.content)
        # return {"ask":qs, "reply":reply.content}
        return {"ask": qs, "reply": reply.content, "original_reply": reply}

    def create_tips(self, mode=None):
        if mode is None:
            mode = ["root"]
        additional_msg = ""
        for node in self.G.nodes():
            # 不是根节点且两个集合有重复元素
            if (node != 'root' and set(mode) & set(self.G.nodes[node]["active"])):
                node_str_list = [f"{name}:{value}" for name, value in self.G.nodes[node]["content"].items()]
                result = " ".join(node_str_list)
                additional_msg += result + "\n"
        return additional_msg

    # 绘制图结构
    def draw_graph(self, name, node_info, active=None, color="normal"):
        if active is None:
            active = ["root"]
        (subtype, node, record, index_list, edge, kwargs, confidence) = node_info
        # 添加属性：激活状态（来判断最终该节点信息传递何处调用
        self.G.add_node(
            name,
            subtype=subtype,
            content=node,
            record=record,
            kwargs=kwargs,
            confidence=confidence,
            active = active,
            color=COLORMAP[color]
        )
        for index in index_list:
            self.G.add_edge(index, name, label=edge)
        return name

    def save_chat(self):
        path = os.path.join(self.output, "test_txt.txt")
        with open(path, "w") as f:
            for msg, i in enumerate(self.MSG):
                if i % 2 == 0:
                    f.write("Human" + msg + "\n")
                else:
                    f.write("AI" + msg + "\n")
        return

#目前任务：
#colormap（更高级的属性）的属性

#探索（反复提问获得新信息，方法）与回顾（已知信息的关联再挖掘，错误信息的平反或删除）的平衡
#任务的不断细化以及总结
#分成细碎的点，因为每个点都有可能继续向外延申
#多线程同时进行，减少生成时间
