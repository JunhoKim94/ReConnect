# Templates for the system instructions.
# * Each explanations are seperated by "*".
CSQA = {
    "mcq": """\
You are a helpful assistant for question answering. \
You are given a question and 5 options (labeled A, B, C, D, and E). \
Your task is to choose the label corresponding to the best answer for the question. \
Do you understand the task?\
""",
    "mcq_with_kg": """\
You are a helpful assistant for question answering. \
You are given a question, 5 options (labeled A, B, C, D, and E), and explanations. \
Based on the given explanations, your task is to choose the label corresponding to the best answer for the question. \
Do you understand the task?\
""",
    "knowledge_generation": """\
You are given a question and 5 options. \
Based on the reasoning skills, your task is to write comprehensive explanations that support the most likely option. \
Note that:
* There is always one option that is correct and more likely than the others.
* The explanations must support only the most likely option and refute all the others.
* The explanations must be simple and concise (max 15 words).
Do you understand the task?\
""",
    "knowledge_generation_external": """\
You are given a question, 5 options, and external knowledge references. \
Based on the given external knowledge references, your task is to extract the most relevant information that supports the most likely option. \
Note that:
* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to the question.
* Do not simply replicate the given knowledge, but you should offer a refined and comprehensive explanation.
* The explanations must be simple and concise (max 15 words).
Do you understand the task?\
""",
#Your task is to select the most relevant external knowledge for answering the given question. \
'knowledge_selection': """\
You are given a question, 5 options, and a list of explanations. \
Based on the given explanations, your task is to synthesize these explanations into a single, high-quality explanation that supports the most likely option. \
Note that:
* It is crucial to critically evaluate the information provided in these explanations, recognizing that some of it may be incorrect.
* Do not simply replicate the given explanations, but you should offer a refined and accurate explanation.
* There is always one option that is correct.
Do you understand the task?\
""",
#Based on the external knowledge, your task is to refine the given explanations into high-quality explanations that support the most likely option. \
#* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to question.
'knowledge_refinement': """\
""",
}
#* The explanations must be concise and accurate (max 15 words).

CSQA2 = {
    "mcq": """\
You are a helpful assistant for question answering. \
You are given a question and 2 options (labeled A and B). \
Your task is to choose the label corresponding to the best answer for the question. \
Do you understand the task?\
""",
    "mcq_with_kg": """\
You are a helpful assistant for question answering. \
You are given a question, 2 options (labeled A and B), and explanations. \
Based on the given explanations, your task is to choose the label corresponding to the best answer for the question. \
Do you understand the task?\
""",
    "knowledge_generation": """\
You are given a question and 2 options. \
Based on the reasoning skills, your task is to write comprehensive explanations that support the most likely option. \
Note that:
* There is always one option that is correct and more likely than the others.
* The explanations must support only the most likely option and refute all the others.
* The explanations must be simple and concise (max 15 words).
Do you understand the task?\
""",
    "knowledge_generation_external": """\
You are given a question, 2 options, and external knowledge references. \
Based on the given external knowledge references, your task is to extract the most relevant information that supports the most likely option. \
Note that:
* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to the question.
* Do not simply replicate the given knowledge, but you should offer a refined and comprehensive explanation.
* The explanations must be simple and concise (max 15 words).
Do you understand the task?\
""",
'knowledge_selection': """\
You are given a question, 2 options, and a list of explanations. \
Based on the given explanations, your task is to synthesize these explanations into a single, high-quality explanation that supports the most likely option. \
Note that:
* It is crucial to critically evaluate the information provided in these explanations, recognizing that some of it may be incorrect.
* Do not simply replicate the given explanations, but you should offer a refined and accurate explanation.
* There is always one option that is correct.
Do you understand the task?\
""",
'knowledge_refinement': """\
""",
}


PIQA = {
    "mcq": """\
You are a helpful assistant for question answering. \
You are given a question or a text to complete and 2 possible solutions (labeled A and B). \
Your task is to choose the label corresponding to the best solution. \
Do you understand the task?\
""",
    "mcq_with_kg": """\
You are a helpful assistant for question answering. \
You are given a question or a text to complete, 2 possible solutions (labeled A and B), and explanations. \
Based on the given explanations, your task is to choose the label corresponding to the best solution. \
Do you understand the task?\
""",
    "knowledge_generation": """\
You are given a question or a text to complete and 2 possible solutions. \
Based on the reasoning skills, your task is to write comprehensive explanations that support the most likely solution. \
Note that:
* There is always one option that is correct and more likely than the others.
* The explanations must support only the most likely option and refute all the others.
* The explanations must be simple and concise (max 15 words).
Do you understand the task?\
""",
    "knowledge_generation_external": """\
You are given a question or a text to complete, 2 possible solutions, and external knowledge references. \
Based on the given external knowledge references, your task is to extract the most relevant information that supports the most likely option. \
Note that:
* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to the question.
* Do not simply replicate the given knowledge, but you should offer a refined and comprehensive explanation.
* The explanations must be simple and concise (max 15 words).
Do you understand the task?\
""",
'knowledge_selection': """\
You are given a question or a text to complete, 2 possible solutions, and a list of explanations. \
Based on the explanations, your task is to synthesize these explanations into a single, high-quality explanation that supports the most likely option. \
Note that:
* It is crucial to critically evaluate the information provided in these explanations, recognizing that some of it may be incorrect.
* Do not simply replicate the given explanations, but you should offer a refined and accurate explanation.
* There is always one option that is correct.
Do you understand the task?\
""",
'knowledge_refinement': """\
""",
}

ARC = {
    "mcq": """\
You are a helpful assistant for question answering. \
You are given a question and up to 5 options (labeled A, B, C, D, and E). \
Your task is to choose the label corresponding to the best answer for the question. \
Do you understand the task?\
""",
    "mcq_with_kg": """\
You are a helpful assistant for question answering. \
You are given a question, up to 5 options (labeled A, B, C, D, and E), and explanations. \
Based on the given explanations, your task is to choose the label corresponding to the best answer for the question. \
Do you understand the task?\
""",
    "knowledge_generation": """\
You are given a question and up to 5 options. \
Based on the reasoning skills, your task is to write comprehensive explanations that support the most likely option. \
Note that:
* There is always one option that is correct and more likely than the others.
* The explanations must support only the most likely option and refute all the others.
* The explanations must be simple and concise (max 15 words).
Do you understand the task?\
""",
    "knowledge_generation_external": """\
You are given a question, 5 options, and external knowledge references. \
Based on the external knowledge references, your task is to extract the most relevant information that supports the most likely option. \
Note that:
* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to the question.
* Do not simply replicate the given knowledge, but you should offer a refined and comprehensive explanation.
* The explanations must be simple and concise (max 15 words).
Do you understand the task?\
""",
'knowledge_selection': """\
You are given a question, 5 options, and a list of explanations. \
Based on the given explanations, your task is to synthesize these explanations into a single, high-quality explanation that support the most likely option. \
Note that:
* It is crucial to critically evaluate the information provided in these explanations, recognizing that some of it may be incorrect.
* Do not simply replicate the given explanations, but you should offer a refined and accurate explanation.
* There is always one option that is correct.
Do you understand the task?\
""",
#Based on the external knowledge, your task is to refine the given explanations into high-quality explanations that support the most likely option. \
#* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to question.
'knowledge_refinement': """\
""",
}

OBQA = {
    "mcq": """\
You are a helpful assistant for question answering. \
You are given a question and 4 options (labeled A, B, C, and D). \
Your task is to choose the label corresponding to the best answer for the question. \
Do you understand the task?\
""",
    "mcq_with_kg": """\
You are a helpful assistant for question answering. \
You are given a question, 4 options (labeled A, B, C, and D), and explanations. \
Based on the given explanations, your task is to choose the label corresponding to the best answer for the question. \
Do you understand the task?\
""",
    "knowledge_generation": """\
You are given a question and 4 options. \
Based on the reasoning skills, your task is to write comprehensive explanations that support the most likely option. \
Note that:
* There is always one option that is correct and more likely than the others.
* The explanations must support only the most likely option and refute all the others.
* The explanations must be simple and concise (max 15 words).
Do you understand the task?\
""",
    "knowledge_generation_external": """\
You are given a question, 4 options, and external knowledge references. \
Based on the external knowledge references, your task is to extract the most relevant information that supports the most likely option. \
Note that:
* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to the question.
* Do not simply replicate the given knowledge, but you should offer a refined and comprehensive explanation.
* The explanations must be simple and concise (max 15 words).
Do you understand the task?\
""",
'knowledge_selection': """\
You are given a question, 4 options, and a list of explanations. \
Based on the given explanations, your task is to synthesize these explanations into a single, high-quality explanation that support the most likely option. \
Note that:
* It is crucial to critically evaluate the information provided in these explanations, recognizing that some of it may be incorrect.
* Do not simply replicate the given explanations, but you should offer a refined and accurate explanation.
* There is always one option that is correct.
Do you understand the task?\
""",
#Based on the external knowledge, your task is to refine the given explanations into high-quality explanations that support the most likely option. \
#* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to question.
'knowledge_refinement': """\
""",
}


QASC = {
    "mcq": """\
You are a helpful assistant for question answering. \
You are given a question and 8 options (labeled A, B, C, D, E, F, G and H). \
Your task is to choose the label corresponding to the best answer for the question. \
Do you understand the task?\
""",
    "mcq_with_kg": """\
You are a helpful assistant for question answering. \
You are given a question, 8 options (labeled A, B, C, D, E, F, G and H), and explanations. \
Based on the given explanations, your task is to choose the label corresponding to the best answer for the question. \
Do you understand the task?\
""",
    "knowledge_generation": """\
You are given a question and 8 options. \
Based on the reasoning skills, your task is to write comprehensive explanations that support the most likely option. \
Note that:
* There is always one option that is correct and more likely than the others.
* The explanations must support only the most likely option and refute all the others.
* The explanations must be simple and concise (max 15 words).
Do you understand the task?\
""",    
"knowledge_generation_external": """\
You are given a question, 8 options, and external knowledge references. \
Based on the given external knowledge references, your task is to extract the most relevant information that supports the most likely option. \
Note that:
* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to the question.
* Do not simply replicate the given knowledge, but you should offer a refined and comprehensive explanation.
* The explanations must be simple and concise (max 15 words).
Do you understand the task?\
""",
'knowledge_selection': """\
You are given a question, 8 options, and a list of explanations. \
Based on the given explanations, your task is to synthesize these explanations into a single, high-quality explanation that support the most likely option. \
Note that:
* It is crucial to critically evaluate the information provided in these explanations, recognizing that some of it may be incorrect.
* Do not simply replicate the given explanations, but you should offer a refined and accurate explanation.
* There is always one option that is correct.
Do you understand the task?\
""",
#Based on the external knowledge, your task is to refine the given explanations into high-quality explanations that support the most likely option. \
#* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to question.
'knowledge_refinement': """\
""",
}

WG = {
    "mcq": """\
You are a helpful assistant for question answering. \
You are given a question where one word has been replaced with \"_\" and 2 options (labeled A and B) to replace \"_\". \
Your task is to choose the label corresponding to the best answer for the question. \
Do you understand the task?\
""",
    "mcq_with_kg": """\
You are a helpful assistant for question answering. \
You are given a question, 2 options (labeled A and B) to replace \"_\", and explanations. \
Based on the given explanations, your task is to choose the label corresponding to the best answer for the question. \
Do you understand the task?\
""",
    "knowledge_generation": """\
You are given a question where one word has been replaced with \"_\" and 2 options to replace \"_\". \
Based on the reasoning skills, your task is to write comprehensive explanations that support the most likely option. \
Note that:
* There is always one option that is correct and more likely than the others.
* The explanations must support only the most likely option and refute all the others.
* The explanations must be simple and concise (max 15 words).
Do you understand the task?\
""",
 "knowledge_generation_external": """\
You are given a question where one word has been replaced with \"_\", 2 options to replace \"_\", and external knowledge references. \
Based on the given external knowledge references, your task is to extract the most relevant information that supports the most likely option. \
Note that:
* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to the question.
* Do not simply replicate the given knowledge, but you should offer a refined and comprehensive explanation.
* The explanations must be simple and concise (max 15 words).
Do you understand the task?\
""",
'knowledge_selection': """\
You are given a question where one word has been replaced with \"_\", 2 options to replace \"_\", and a list of explanations. \
Based on the given explanations, your task is to synthesize these explanations into a single, high-quality explanation that support the most likely option. \
Note that:
* It is crucial to critically evaluate the information provided in these explanations, recognizing that some of it may be incorrect.
* Do not simply replicate the given explanations, but you should offer a refined and accurate explanation.
* There is always one option that is correct.
Do you understand the task?\
""",
#Based on the external knowledge, your task is to refine the given explanations into high-quality explanations that support the most likely option. \
#* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to question.
'knowledge_refinement': """\
""",
}


##############OOD Dataset Prompts #######################

SIQA = {
    "mcq": """\
You are a helpful assistant for question answering. \
You are given a question and 3 options (labeled A, B and C). \
Your task is to choose the label corresponding to the best answer for the question. \
Do you understand the task?\
""",
    "mcq_with_kg": """\
You are a helpful assistant for question answering. \
You are given a question, 3 options (labeled A, B and C), and explanations. \
Based on the given explanations, your task is to choose the label corresponding to the best answer for the question. \
Do you understand the task?\
""",
    "knowledge_generation": """\
You are given a question and 3 options. \
Based on the reasoning skills, your task is to write comprehensive explanations that support the most likely option. \
Note that:
* There is always one option that is correct and more likely than the others.
* The explanations must support only the most likely option and refute all the others.
* The explanations must be simple and concise (max 15 words).
Do you understand the task?\
""",
    "knowledge_generation_external": """\
You are given a question, 3 options, and external knowledge references. \
Based on the given external knowledge references, your task is to synthesize these knowledge into one or more high-quality explanations that support the most likely option. \
Note that:
* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to the question.
* Do not simply replicate the given knowledge, but you should offer a refined and comprehensive explanation.
* The explanations must be simple and concise (max 15 words).
Do you understand the task?\
""",
'knowledge_selection': """\
You are given a question or a text to complete, 3 options, and a list of explanations. \
Based on the given explanations, your task is to synthesize these explanations into a single, high-quality explanation that supports the most likely option. \
Note that:
* It is crucial to critically evaluate the information provided in these explanations, recognizing that some of it may be incorrect.
* Do not simply replicate the given explanations, but you should offer a refined and accurate explanation.
* There is always one option that is correct.
Do you understand the task?\
""",
'knowledge_refinement': """\
""",
}


HellaSWAG = {
    "mcq": """\
You are a helpful assistant for question answering. \
You are given a question and 4 options (labeled A, B, C, and D). \
Your task is to choose the label corresponding to the best answer for the question. \
Do you understand the task?\
""",
    "mcq_with_kg": """\
You are a helpful assistant for question answering. \
You are given a question, 4 options (labeled A, B, C, and D), and explanations. \
Based on the given explanations, your task is to choose the label corresponding to the best answer for the question. \
Do you understand the task?\
""",
    "knowledge_generation": """\
You are given a question and 4 options. \
Based on the reasoning skills, your task is to write comprehensive explanations that support the most likely option. \
Note that:
* There is always one option that is correct and more likely than the others.
* The explanations must support only the most likely option and refute all the others.
* The explanations must be simple and concise (max 15 words).
Do you understand the task?\
""",
    "knowledge_generation_external": """\
You are given a question, 4 options, and external knowledge references. \
Based on the external knowledge references, your task is to extract the most relevant information that supports the most likely option. \
Note that:
* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to the question.
* Do not simply replicate the given knowledge, but you should offer a refined and accurate explanation.
* The explanations must be simple and concise (max 15 words).
Do you understand the task?\
""",
'knowledge_selection': """\
You are given a question or a text to complete, 4 options, and a list of explanations. \
Based on the given explanations, your task is to synthesize these explanations into a single, high-quality explanation that supports the most likely option. \
Note that:
* It is crucial to critically evaluate the information provided in these explanations, recognizing that some of it may be incorrect.
* Do not simply replicate the given explanations, but you should offer a refined and accurate explanation.
* There is always one option that is correct.
Do you understand the task?\
""",
'knowledge_refinement': """\
""",
}

COM2SENSE = {
    "mcq": """\
You are a helpful assistant for question answering. \
You are given a question and 2 options (labeled A and B). \
Your task is to choose the label corresponding to the best answer for the question. \
Do you understand the task?\
""",
    "mcq_with_kg": """\
You are a helpful assistant for question answering. \
You are given a question, 2 options (labeled A and B), and explanations. \
Based on the given explanations, your task is to choose the label corresponding to the best answer for the question. \
Do you understand the task?\
""",
    "knowledge_generation": """\
You are given a question and 2 options. \
Based on the reasoning skills, your task is to write comprehensive explanations that support the most likely option. \
Note that:
* There is always one option that is correct and more likely than the others.
* The explanations must support only the most likely option and refute all the others.
* The explanations must be simple and concise (max 15 words).
Do you understand the task?\
""",
    "knowledge_generation_external": """\
You are given a question, 2 options, and external knowledge references. \
Based on the external knowledge references, your task is to extract the most relevant information that supports the most likely option. \
Note that:
* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to the question.
* Do not simply replicate the given knowledge, but you should offer a refined and comprehensive explanation.
* The explanations must be simple and concise (max 15 words).
Do you understand the task?\
""",
'knowledge_selection': """\
You are given a question or a text to complete, 2 options, and a list of explanations. \
Based on the given explanations, your task is to synthesize these explanations into a single, high-quality explanation that support the most likely option. \
Note that:
* It is crucial to critically evaluate the information provided in these explanations, recognizing that some of it may be incorrect.
* Do not simply replicate the given explanations, but you should offer a refined and accurate explanation.
* There is always one option that is correct.
Do you understand the task?\
""",
'knowledge_refinement': """\
""",
}

RIDDLESENSE = {
    "mcq": """\
You are a helpful assistant for question answering. \
You are given a question and 5 options (labeled A, B, C, D, and E). \
Your task is to choose the label corresponding to the best answer for the question. \
Do you understand the task?\
""",
    "mcq_with_kg": """\
You are a helpful assistant for question answering. \
You are given a question, 5 options (labeled A, B, C, D, and E), and explanations. \
Based on the given explanations, your task is to choose the label corresponding to the best answer for the question. \
Do you understand the task?\
""",
    "knowledge_generation": """\
You are given a question and 5 options. \
Based on the reasoning skills, your task is to write comprehensive explanations that support the most likely option. \
Note that:
* There is always one option that is correct and more likely than the others.
* The explanations must support only the most likely option and refute all the others.
* The explanations must be simple and concise (max 15 words).
Do you understand the task?\
""",
    "knowledge_generation_external": """\
You are given a question, 5 options, and external knowledge references. \
Based on the external knowledge references, your task is to extract the most relevant information that supports the most likely option. \
Note that:
* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to the question.
* Do not simply replicate the given knowledge, but you should offer a refined and accurate explanation.
* The explanations must be simple and concise (max 15 words).
Do you understand the task?\
""",
'knowledge_selection': """\
You are given a question or a text to complete, 5 options, and a list of explanations. \
Based on the given explanations, your task is to synthesize these explanations into a single, high-quality explanation that support the most likely option. \
Note that:
* It is crucial to critically evaluate the information provided in these explanations, recognizing that some of it may be incorrect.
* Do not simply replicate the given explanations, but you should offer a refined and accurate explanation.
* There is always one option that is correct.
Do you understand the task?\
""",
'knowledge_refinement': """\
""",
}


NUMERSENSE = {
    "mcq": """\
You are a helpful assistant for question answering. \
You are given a question and 12 options (labeled A, B, C, D, E, F, G, H, I, J, K, and L). \
Your task is to choose the label corresponding to the best answer for the question. \
Do you understand the task?\
""",
    "mcq_with_kg": """\
You are a helpful assistant for question answering. \
You are given a question, 12 options (labeled A, B, C, D, E, F, G, H, I, J, K, and L), and explanations. \
Based on the given explanations, your task is to choose the label corresponding to the best answer for the question. \
Do you understand the task?\
""",
    "knowledge_generation": """\
You are given a question and 12 options. \
Based on the reasoning skills, your task is to write comprehensive explanations that support the most likely option. \
Note that:
* There is always one option that is correct and more likely than the others.
* The explanations must support only the most likely option and refute all the others.
* The explanations must be simple and concise (max 15 words).
Do you understand the task?\
""",
    "knowledge_generation_external": """\
You are given a question, 12 options, and external knowledge references. \
Based on the given external knowledge references, your task is to extract the most relevant information that supports the most likely option. \
Note that:
* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to the question.
* Do not simply replicate the given knowledge, but you should offer a refined and comprehensive explanation.
* The explanations must be concise and accurate (max 15 words).
Do you understand the task?\
""",
'knowledge_selection': """\
You are given a question or a text to complete, 12 options, and a list of explanations. \
Based on the given explanations, your task is to synthesize these explanations into a single, high-quality explanation that supports the most likely option. \
Note that:
* It is crucial to critically evaluate the information provided in these explanations, recognizing that some of it may be incorrect.
* Do not simply replicate the given explanations, but you should offer a refined and accurate explanation.
* There is always one option that is correct and more likely than the others.
Do you understand the task?\
""",
'knowledge_refinement': """\
""",
}


NUMERSENSE2 = {
    "mcq": """\
You are a helpful assistant for question answering. \
You are given a question where one word has been replaced with \"_\" and 12 options (labeled A, B, C, D, E, F, G, H, I, J, K, and L) to replace \"_\". \
Your task is to choose the label corresponding to the best answer for the question. \
Do you understand the task?\
""",
    "mcq_with_kg": """\
You are a helpful assistant for question answering. \
You are given a question where one word has been replaced with \"_\", 12 options (labeled A, B, C, D, E, F, G, H, I, J, K, and L) to replace \"_\", and explanations. \
Based on the given explanations, your task is to choose the label corresponding to the best answer for the question. \
Do you understand the task?\
""",
    "knowledge_generation": """\
You are given a question and 12 options. \
Based on the reasoning skills, your task is to write comprehensive explanations that support the most likely option. \
Note that:
* There is always one option that is correct and more likely than the others.
* The explanations must support only the most likely option and refute all the others.
* The explanations must be simple and concise (max 15 words).
Do you understand the task?\
""",
    "knowledge_generation_external": """\
You are given a question where one word has been replaced with \"_\", 12 options to replace \"_\", and external knowledge references. \
Based on the given external knowledge references, your task is to extract the most relevant information that supports the most likely option. \
Note that:
* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to the question.
* Do not simply replicate the given knowledge, but you should offer a refined and comprehensive explanation.
* The explanations must be concise and accurate (max 15 words).
Do you understand the task?\
""",
'knowledge_selection': """\
You are given a question where one word has been replaced with \"_\", 12 options to replace \"_\", and a list of explanations. \
Based on the given explanations, your task is to synthesize these explanations into a single, high-quality explanation that supports the most likely option. \
Note that:
* It is crucial to critically evaluate the information provided in these explanations, recognizing that some of it may be incorrect.
* Do not simply replicate the given explanations, but you should offer a refined and accurate explanation.
* There is always one option that is correct and more likely than the others.
Do you understand the task?\
""",
'knowledge_refinement': """\
""",
}

QUARTZ = {
    "mcq": """\
You are a helpful assistant for question answering. \
You are given a question and 2 options (labeled A and B). \
Your task is to choose the label corresponding to the best answer for the question. \
Do you understand the task?\
""",
    "mcq_with_kg": """\
You are a helpful assistant for question answering. \
You are given a question, 2 options (labeled A and B), and explanations. \
Based on the given explanations, your task is to choose the label corresponding to the best answer for the question. \
Do you understand the task?\
""",
    "knowledge_generation": """\
You are given a question and 2 options. \
Based on the reasoning skills, your task is to write comprehensive explanations that support the most likely option. \
Note that:
* There is always one option that is correct and more likely than the others.
* The explanations must support only the most likely option and refute all the others.
* The explanations must be simple and concise (max 15 words).
Do you understand the task?\
""",
    "knowledge_generation_external": """\
You are given a question, 2 options, and external knowledge references. \
Based on the given external knowledge references, your task is to extract the most relevant information that supports the most likely option. \
Note that:
* It is crucial to critically evaluate the information provided in the external knowledge, recognizing that some of it may be irrelevant to the question.
* Do not simply replicate the given knowledge, but you should offer a refined and comprehensive explanation.
* The explanations must be simple and concise (max 15 words).
Do you understand the task?\
""",
'knowledge_selection': """\
You are given a question or a text to complete, 2 options, and a list of explanations. \
Based on the given explanations, your task is to synthesize these explanations into a single, high-quality explanation that support the most likely option. \
Note that:
* It is crucial to critically evaluate the information provided in these explanations, recognizing that some of it may be incorrect.
* Do not simply replicate the given explanations, but you should offer a refined and accurate explanation.
* There is always one option that is correct.
Do you understand the task?\
""",
'knowledge_refinement': """\
""",
}



#HellaSwag / Com2sense / Nummersense ... 


# Templates for the fewshot examples.
MCQ_EXAMPLE_TEMPLATE = """\
Question:
{question}

Options:
{choices}\
"""

MCQ_WITH_KNOWLEDGE_EXAMPLE_TEMPLATE = """\
Question:
{question}

Options:
{choices}

Explanations:
{knowledge}\
"""

KNOWLEDGE_GENERATION_EXAMPLE_TEMPLATE = """\
Question:
{question}

Options:
{choices}\
"""

KNOWLEDGE_GENERATION_EXAMPLE_TEMPLATE_EXTERNAL = """\
Question:
{question}

Options:
{choices}\
"""

KNOWLEDGE_SELECTION = """\
Question:
{question}

Options:
{choices}

Explanations:
{knowledge}\
"""


KNOWLEDGE_AGGREGATION = """\
Question:
{question}

Options:
{choices}
"""



SHOT_TEMPLATES = {
    "mcq": MCQ_EXAMPLE_TEMPLATE,
    "mcq_with_kg": MCQ_WITH_KNOWLEDGE_EXAMPLE_TEMPLATE,
    "knowledge_generation": KNOWLEDGE_GENERATION_EXAMPLE_TEMPLATE,
    "knowledge_generation_external" : KNOWLEDGE_GENERATION_EXAMPLE_TEMPLATE_EXTERNAL,
    "knowledge_selection" : KNOWLEDGE_SELECTION,
    "knowledge_aggregation" : KNOWLEDGE_AGGREGATION
    #"knowledge_aggregation" : MCQ_EXAMPLE_TEMPLATE
}

# Mapping from dataset names to their respective instruction prompts.
OUR_DATASET_TAGS = {
    "csqa": CSQA,
    "csqa2": CSQA2,
    "piqa": PIQA,
    "siqa": SIQA,
    "hellaswag" : HellaSWAG,
    "com2sense" : COM2SENSE,
    "riddlesense" : RIDDLESENSE,
    "obqa" : OBQA,
    "qasc" : QASC,
    "wg" : WG,
    "arc-challenge" : ARC,
    "arc-easy" : ARC,
    "numersense" : NUMERSENSE2,
    "quartz" : QUARTZ,
    "quarel" : QUARTZ,
    "prost" : HellaSWAG,
    "sciq" : HellaSWAG,
    "wsc" : QUARTZ,
    "cycic" : CSQA,
    "sct": QUARTZ,
}
