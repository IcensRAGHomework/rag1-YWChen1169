import json
import traceback

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, FewShotChatMessagePromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.agents import tool, create_openai_functions_agent, AgentExecutor
from langchain.chains import LLMChain


from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

import base64

import calendarific

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

def generate_hw01(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )

    examples = [
        {
        "input":"2024年台灣10月紀念日有哪些?",
        "output":
        """{
                "Result": [
                    {
                        "date": "2024-10-10",
                        "name": "國慶日"
                    },
                    {
                        "date": "2024-10-11",
                        "name": "重陽節"
                    }
                ]
            }"""
        },
    ]

    example_prompt = ChatPromptTemplate.from_messages(
        [
        ("human", "{input}"),
        ("ai", "{output}"),
        ]
        )
    few_shot_promt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples,
        )

    final_prompt = ChatPromptTemplate.from_messages(
            [
            ("system","You are an assistant that provides structured answers to user queries. "
            "Below is the input query and the intermediate steps you have taken. and the result is the json output\n"),
            few_shot_promt,
            ("human", "{input}"),
            ]
        )

    chain = final_prompt | llm

    response = chain.invoke({"input": question })



    return response.content  
    
def generate_hw02(question):
    # define the tool
    @tool
    def check_calendarific(country : str, year : str, month : str) -> dict:
        """  use the calendarific api to check all the holiday with the str parameter country, year, month. 
            year and month is string
        """
        calapi = calendarific.v2('WjJ5eItS7tMLp9x0c1SS1x7HxaGSTq9t')
        parameters = {
            'api_key': 'WjJ5eItS7tMLp9x0c1SS1x7HxaGSTq9t',
            'country': country,
            'year':    year,
            'month':   month,
            }

        holidays = calapi.holidays(parameters)
        output_data = {
            "Result": [
            {
                "date": holiday['date']['iso'],
                "name": holiday['name']
            }
            for holiday in holidays['response']['holidays']
        ]
        }

        return output_data
    tools = [check_calendarific]

    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )


    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )

    examples = [
        {
        "input":"2024年台灣10月紀念日有哪些?",
        "output":
        """
        {
            "Result": [
                {
                    "date": "2024-10-10",
                    "name": "National Day"
                },
                {
                    "date": "2024-10-09",
                    "name": "Double Ninth Day"
                },
                {
                    "date": "2024-10-21",
                    "name": "Overseas Chinese Day"
                },
                {
                    "date": "2024-10-25",
                    "name": "Taiwan's Retrocession Day"
                },
                {
                    "date": "2024-10-31",
                    "name": "Halloween"
                }
            ]
        }
        """
        },
    ]

    example_prompt = ChatPromptTemplate.from_messages(
            [
            ("human", "{input}"),
            ("ai", "{output}"),
            ]
        )
    few_shot_promt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples,
        )

    final_prompt = ChatPromptTemplate.from_messages(
            [
            ("system","You are an assistant."
            "Below is the input query and the intermediate steps you have taken."
            "the output data sturctured was like the examples, provide the json data structure"),
            few_shot_promt,
            ("human", "{input}"),
            ("ai" , "{agent_scratchpad}"),
            ]
        )

    agent = create_openai_functions_agent(llm = llm, tools = tools, prompt = final_prompt)
    agent_excutor = AgentExecutor(agent = agent, tools = tools, verbose = True)
    response = agent_excutor.invoke({"input" : message,"agent_scratchpad" : ""})


    return response["output"].strip('```json\n').strip('```')
    
def generate_hw03(question2, question3):
    pass

def generate_hw04(question):
    pass
def demo(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    response = llm.invoke([message])
    
    return response


print(generate_hw01('2024年台灣10月紀念日有哪些?'))