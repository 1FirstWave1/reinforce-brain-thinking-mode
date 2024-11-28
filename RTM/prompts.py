#提示词之整体识别

SYS_PROMPTS = """For the following text, you do not need to answer the questions directly now, but pay attention to the following questions about the text.

text: {text}
"""
#答案
ANSWER = "Based on the question and all the above information, provide the final answer."
#之后可能做成选项选择的形式
#任务类型
TASK_RECOGNITION = "What is the task mentioned above? specify the type of task (e.g., mathematics, biology, general knowledge and so on).Explain the approach to solving this type of question and the key considerations to keep in mind."
#提取具体的任务
TASK_EXTRACTION = "Extract specific questions from the text and refer directly to the original text, taking care not to add extraneous information"
#任务的难度，可能做成选择题的形式
TASK_DIFFICULTY = "What is the difficulty of the problem?"

#是否，质疑问题
QUESTION_DOUBT = "Are there any pitfalls to be aware of for this task? Please provide your reasons and your final answer.(Yes/No)"
#摘出相应概念
EXTRACT = "Please extract the nouns, verbs, adjectives, adverbs, and formulas with actual meaning from the text above."

#提示词具体

#分解任务
DECOMPOSE_TASK = "Please break down the <question>{problem}</question> into several steps and briefly describe the work that should be done in each step. Note that you do not need to provide the answer for each step at this time."
# #or
# DECOMPOSE_TASK = "Let's think step by step about how to solve this problem. Please provide the number of steps and explain the tasks of each step in detail."

FIRST_STEP = "For the first step, please provide an answer based on the issue and the goal of the first step."

STEP = "For the step{i}, what should we do? Please provide an answer based on the issue, the answers to the previous steps, and the goal of step{i}."
DECOMPOSE_ITEM = ""


#反思推断
REFLECTION_OTHER_CONCLUSIONS ="Given <question>{problem}</question> and <result>{result}</result>, what other conclusions can we draw?Please answer in separate parts."
#反思推断（只含结果）
REFLECTION_OTHER_CONCLUSIONS_RESULT ="Given <result>{result}</result>, what other conclusions can we draw (forward and backward inference)? Please answer in separate parts."
#是否违反常识
VIOLATION_CONCLUSION ="In the context of the problem, do these conclusions violate common sense or rules? Please provide your reasons and your final answer. (Yes/No)"
#违反常识的具体内容
WHAT_VIOLATION_CONCLUSION = "What common sense or rules are violated?"

NEGATIVE_CONCLUSION = "If the answer to <question>{problem}</question> is not <result>{result}</result>, what could be the possible reasons?"

REFLECTION_REASON = "The answer to <question>{problem}</question> is <real_result>{real_result}</real_result>, but our answer is <our_result>{result}</our_result>, what could be the possible reason?"




#联想
ASSOCIATION = "what words/stories/rules/theorems does <item>{item}</item> remind us of? Please explain each one."

FUTHER_ASSOCIATION = "In the context of <question>{problem}</question>, what other related content can these words/stories/rules/theorems make us think of?"
#可以反复提问得到一个集合
SIMILAR_PROBLEM = "What other problems are similar to <question>{problem}</question>? Please provide an example and its solution."




#对比
COMPARE = "What are the similarities and differences between <thing1>{thing1}</thing1> and <thing2>{thing2}</thing2>? Please provide a detailed explanation and answer in separate parts"

COMPARE_ORDINARY ="Compared to the usual tasks, what are the differences in this <question>{problem}</question>? Please answer in separate parts."

DIFFERENCE_IMPACT = "What impact do these <differences>{differences}</differences> have on the <question>{problem}</question>?"
#or
# DIFFERENCE_IMPACT = "Based on the task and its answer, what modifications should be made to the original answer for this task?"
#选择答案的语句中没有问题。。。。。
DIFFERENCE_ANSWER = "For the specific <question>{problem}</question>, What are the differences between <answer1>{answer1}</answer1> and <answer2>{answer2}</answer2>?"

CHOOSE_ANSWER = "Which answer is better under this <question>{problem}</question>? answer1:{answer1} \n answer2:{answer2} \n The diferences between two answers \
is <differences>{differences}</differences>. Please provide your reasons.Finally, choose the better one."




#重要性
IMPORTANCE = "What is the most important aspect of <question>{problem}</question>?"
#针对文本的删减
CONDENSE_TEXT = "Please remove the information that is irrelevant to this issue from the text.The text is <text>{text}</text>."

UNIMPORTANT_POINT = "The following text is the relative point to the problem.\
    Please select some unimportant or irrelevant nodes in solving the problem.The problem is <question>{problem}</question>.The nodes text is <node_text>{node_text}</node_text>."

HELP_JUDGMENT = "This <item>{item}</item> is related to <main_thing>{main_thing}</main_thing>. Is it helpful in solving the <question>{problem}</question>?"



#因果与推理
COUNTER_FACTUAL1 = "If <thing>{thing}</thing> does not exist, what impact would it have on the <result>{result}</result>?"
COUNTER_FACTUAL2 = "If <thing>{thing}</thing> is opposite, what impact would it have on the <result>{result}</result>?"

REASON = "What is the reason for <thing>{thing}</thing> occurring?"
RESULT = "What kind of impact or outcome will this <thing>{thing}</thing> bring?"
DEFINE = "What is the definition of <thing>{thing}</thing>?"



#归纳
INDUCTION = "Based on the above text, what kind of rule/theorem/formula can we derive?"

#base(严格来说没啥提示词)
base_sys_prompt = """Question: {question}
Answer:
"""
#cot基础提示词
cot_sys_prompt = """Question: {question}

Let's think step by step.
Answer:
"""

shot_sys_prompt = """Question: {question}

Here are few examples:
{few_shots}

Answer:
"""