#brain_net
import argparse
import os
from tqdm import tqdm
from sys2.load_data import DATASETS
from sys2.model_test1 import sys2Agent
from sys2.baseline import baseAgent, cotAgent, shotAgent
from sys2.output import compute_answer_callback, BaseMetrics, save_data
from langchain_openai import ChatOpenAI
# from langsmith import Client
from dotenv import load_dotenv

_ = load_dotenv(dotenv_path='env.env')

#生成命令行参数的默认值
parser = argparse.ArgumentParser(
    prog="sys2",
    description="interaction with llm",
    epilog="Text at the bottom of help",
)
parser.add_argument("--agent", default="sys2", choices=["base", "sys2", "cot", "few_shots"])
parser.add_argument("--input", default="data/gpqa-main/dataset/gpqa_main.csv", choices=["data/GSM8K/test.jsonl", "data/gpqa-main/dataset/gpqa_main.csv", "data/TruthfulQA-main/data/mc_task.json"])
parser.add_argument("--output", default="./result/sys2/GPQA")
parser.add_argument("--dataset_name", default="GPQA", choices=["GSM8K", "GPQA", "TruthfulQA"])
parser.add_argument("--max_steps", default=30, type=int)
parser.add_argument("--temperature", default=0, type=float)
parser.add_argument("--prompt_version", default=0, type=int)
parser.add_argument("--max_tokens", default=1024, type=int)
parser.add_argument("--debug", action="store_true")
#gpt-4o mini输出规范性比较强，价格非常香
parser.add_argument("--model_name", default="gpt-4o-mini", choices=["gpt-3.5-turbo", "gpt-4", "gpt-4o-mini"])
parser.add_argument("--use_wandb")
# parser.add_argument("--data", default="There are only three people on the playground Xiao Ming,\
# Xiao Hong, and Xiao Li. Xiao Ming is running, Xiao Hong is playing tennis with someone, so what is Xiao Li doing?")
# parser.add_argument("--data", default="Which number is bigger between 3.9 and 3.11?")


def main(args):
    # 加载数据集
    Dataset = DATASETS[args.dataset_name]
    #为什么这里直接就运行了？
    dataset = Dataset(data_path=args.input)    
    chat = ChatOpenAI(
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    # #暂时注销
    # client = Client()
    if args.agent == "sys2":
        agent = sys2Agent(
            chat=chat,
            output=args.output,
        )
    elif args.agent == "base":
        agent = baseAgent(
            chat=chat,
            output=args.output,
        )
    elif args.agent == "cot":    
        agent = cotAgent(
            chat=chat,
            output=args.output,
        )
    elif args.agent == "few_shots":
        shots = ""
        for idx, data in dataset:
            shot = "Question:"+data["question"] + "\n" +"Answer:"+data["answer"]
            if shots:
                shots = shots+"\n\n"+shot
            else:
                shots = shots+shot
            if idx >=2:
                break
        agent = shotAgent(
            chat=chat,
            output=args.output,
            shots=shots
        )
    #vars返回对象属性和属性值的字典
    #idx有点怪
    #data键值：idx question answer answer_default
    metrics = BaseMetrics(**vars(args))
    for idx, data in tqdm(dataset):
        del idx
        state, terminate = agent.reset(data)
        for i in tqdm(range(30)):
            # 若终止则更新度量
            if terminate:
                if state == "reload":
                    data = agent.data
                else:
                    data = compute_answer_callback(agent.answer, agent.data, args.dataset_name,chat)
                    #生成的节点数
                    data["number_node"] = i                
                    path = os.path.join(args.output, "output_dev.jsonl")
                    #已完成结果存一个json文件
                    save_data(data, path)

                metrics.update(data)
                metrics.log()
                break
            # 否则执行策略,并更新状态
            # try:
            state, terminate = agent.policy(state,i)
            # 若出现异常则恢复并终止
            # except Exception as e:
            #     print("程序异常终止！")
            #     terminate = True


#之后想办法搞定API的问题
if __name__ == "__main__":
    #解析命令行参数
    args = parser.parse_args()
    main(args)
