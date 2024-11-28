import os

def gain_path(agent, dataset):
    if dataset == "GPQA":
        input_path = "data/GPQA/gpqa_main.csv"
    #todo: Add other datasets' path
    
   
    out_path = os.path.join("data", dataset)
    out_path = os.path.join(out_path, "result")
    out_path = os.path.join(out_path, agent)

    return input_path, output_path