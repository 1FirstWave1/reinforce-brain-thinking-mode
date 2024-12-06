import os

def gain_path(agent, dataset):
    if dataset == "GPQA":
        input_path = "data/GPQA/gpqa_main.csv"
    #todo: Add other datasets' path
    
   
    output_path = os.path.join("data", dataset)
    output_path = os.path.join(output_path, "result")
    output_path = os.path.join(output_path, agent)

    return input_path, output_path