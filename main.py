#reinforce_thinking_mode
#used for final test(not train)
#RTM and other baselines
import argparse
import os
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from RTM.utils import gain_path
from RTM.load_data import DATASETS
from RTM.model_test1 import sys2Agent
from baselines.easy_baselines import baseAgent, cotAgent, shotAgent
from RTM.output import compute_answer_callback, BaseMetrics, save_data



_ = load_dotenv(dotenv_path='env.env')

parser = argparse.ArgumentParser(
    prog="RTM",
    description="interaction with llm",
    epilog="Text at the bottom of help",
)
parser.add_argument("--agent", default="RTM", choices=["base", "RTM", "cot", "3-shot"])
parser.add_argument("--dataset_name", default="GPQA", choices=["GPQA"])
parser.add_argument("--actor_name", default="gpt-4o-mini", choices=["gpt-3.5-turbo", "gpt-4", "gpt-4o-mini", "llama-7b", "Vicuna-13b"])
parser.add_argument("--max_steps", default=30, type=int)
parser.add_argument("--temperature", default=0, type=float)
parser.add_argument("--max_tokens", default=1024, type=int)
#used for log
parser.add_argument("--debug", action="store_true")
parser.add_argument("--use_wandb")



def main(args):
    Dataset = DATASETS[args.dataset_name]
    input_path, out_path = gain_path(args.agent, args.dataset_name)
    dataset = Dataset(data_path=input_path)    
    actor_llm = ChatOpenAI(
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    if args.agent == "RTM":
        env = RTMEnv(
            chat=chat,
            input=input_path,
        )
        agent = RTMAgent(
            chat=actor_llm,
            output=out_path,
        )

    elif args.agent == "base":
        agent = baseAgent(
            chat=actor_llm,
            output=out_path,
        )
    elif args.agent == "cot":    
        agent = cotAgent(
            chat=actor_llm,
            output=out_path,
        )
    elif args.agent == "3-shot":
        
        agent = shotAgent(
            chat=actor_llm,
            output=out_path,
            dataset_name=args.dataset_name,
            number_shots=3,
        )        


    metrics = BaseMetrics(**vars(args))
    only_agent = ["base", "cot", "3-shot"]
    agent_env = ["RTM"]
    #only agent
    if args.agent in only_agent:
        for idx, data in dataset:
            del idx
            state, terminate = agent.reset(data)
            for i in tqdm(range(100)):
                if terminate:
                    if state == "reload":
                        data = agent.data
                    else:
                        data = compute_answer_callback(agent.answer, agent.data, args.dataset_name,chat)          
                        path = os.path.join(args.output, "output_dev.jsonl")
                        save_data(data, path)

                    metrics.update(data)
                    metrics.log()
                    break
                try:
                    state, terminate = agent.policy(state,i)
                except Exception as e:
                    print("Program terminated abnormally!")
                    terminate = True
    #agent and env
    if args.agent in agent_env:
        for idx, data in tqdm(dataset):
            del idx
            state, terminate = env.reset(data)
            for _ in range(100):
                if terminate:
                    metrics.update(env.data)
                    metrics.log()
                    break
                try:
                    action = agent.policy(state)
                    state, terminate = env.step(action)
                except Exception as e:
                    print("Program terminated abnormally!")
                    terminate = True
    

if __name__ == "__main__":
    args = parser.parse_args()
    if args.actor_name == "Vicuna-13b":
        openai.api_key = "EMPTY"
        openai.api_base = "http://localhost:8000/v1"
    elif args.actor_name == "llama-7b":
        openai.api_key = "EMPTY"
        openai.api_base = "http://localhost:8000/v1"    
    main(args)
