from langchain.prompts.prompt import PromptTemplate


#
# Prompt template for the user request classification
#
#Classify the following QUESTION as Code Review, GitHub task, or Test case. The QUESTION should be classified as a GitHub task if it means executing a task on or related to a specific repository. The QUESTION should be classified as a Test case if it is not related to code and it is related to a provided use case. If it is related to code is always Code Review.
#Provide only the classification.
#Current conversation: 
DC4SE_CLASSIFICATION_PROMPT_TEMPLATE = """
Act as a Human Resource Manager. 
Current conversation: 
[{history}]

QUESTION: {input}
"""

#
# Prompt for the general questions 
#
DC4SE_GENERAL_CHAT_PROMPT_TEMPLATE = """
The following is a friendly conversation between a human and an AI. The AI is an expert software engineer, but it can assist only with coding, GitHub tasks, and test cases. So, it can answer only about this topics. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
Current conversation:
[{history}]
{input}
"""


DC4SE_CHAT_PROMPT = PromptTemplate(
    input_variables=["history", "input"], template=DC4SE_GENERAL_CHAT_PROMPT_TEMPLATE
)


DC4SE_CLASSIFICATION_PROMPT = PromptTemplate(
                template=DC4SE_CLASSIFICATION_PROMPT_TEMPLATE, input_variables=["history", "input"]
            )



#
# For Task Automation
#



DC4SE_TASK_TEMPLATE = """
Act as a GitHub API invoker. 
Follow these rules:
- provide the most appropriate FUNCTION ID and its PARAMETERS with the correct values to answer the QUESTION.
- Your answer should be a well-formatted JSON, as in the example.
- If there is more than one appropriate function, answer each function. The functions must have the same order as in the QUESTION.
- If the QUESTION is not related to any function, answer "It is not related to the GitHub API".

Current conversation: 
[{history}]

FUNCTIONS:
{input}

EXAMPLE: 
{{"FUNCTIONS": [
  {{"ID": "1",  "PARAMETERS": {{"owner": "name", "repo": "REP"}}}}
]}}

ANSWER: """

DC4SE_TASK_PROMPT = PromptTemplate(
    input_variables=["history", "input"], template=DC4SE_TASK_TEMPLATE
)
#
# PROMPTS for coding
#

DC4SE_CODING_TEMPLATE = """
Act as as software engineer. Check first if the question is related to the last code we were previously discussing about in the conversation. Answer the QUESTION at your best. Provide code as markdown.
Current conversation:
[{history}]

QUESTION: {input}
"""

DC4SE_CODING_PROMPT = PromptTemplate(
    input_variables=["history","input"], template=DC4SE_CODING_TEMPLATE
)

#
# PROMPTS for UAT Test cases
#
DC4SE_TEST_CASES_TEMPLATE = """
Act as a test engineer. 
If I ask you to provide a test case for a given use case, produce a markdown table containing the following columns: Test case ID, Test case Description, Preconditions, Steps (markdown table containing an ordered list of tests steps with example input data. Each step is a new row), Expected result.
Use the CONTEXT below if it helps. If the QUESTION does not provide a description of the use case scenario and there is no way to infer that from the following CONTEXT, concisely answer: I don't know the use case '{{USE CASE}}'.

CONTEXT: {history}

QUESTION: {input}

ANSWER:"""

DC4SE_TEST_CASES_PROMPT = PromptTemplate(
    input_variables=["history","input"], template=DC4SE_TEST_CASES_TEMPLATE
) #era context