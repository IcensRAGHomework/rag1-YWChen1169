import json
import traceback
import requests
from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, FewShotChatMessagePromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.agents import tool, create_openai_functions_agent, AgentExecutor
from langchain.chains import LLMChain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

import base64


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
        api_key = 'WjJ5eItS7tMLp9x0c1SS1x7HxaGSTq9t'
        url = 'https://calendarific.com/api/v2/holidays?&api_key={api_key}&country={country}&year={year}&month={month}'

        try:
            response = requests.get(url)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            data = response.json()  # Parse JSON response into a dictionary
            output_data = {
                "Result": [
                {
                    "date": holiday['date']['iso'],
                    "name": holiday['name']
                }
                for holiday in data['response']['holidays']
            ]
            }

            return output_data
        except requests.exceptions.HTTPError as http_err:
            print(f'HTTP error occurred: {http_err}')
        except requests.exceptions.RequestException as req_err:
            print(f'Request error occurred: {req_err}')
        except json.JSONDecodeError as json_err:
            print(f'Error decoding JSON: {json_err}')
        except Exception as err:
            print(f'An error occurred: {err}')

        return {"Result": []}  # Return an empty result in case of error
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

    @tool
    def check_calendarific(country : str, year : str, month : str) -> dict:
        """  use the calendarific api to check all the holiday with the str parameter country, year, month. 
            year and month is string
        """
        api_key = 'WjJ5eItS7tMLp9x0c1SS1x7HxaGSTq9t'
        url = f'https://calendarific.com/api/v2/holidays?&api_key={api_key}&country={country}&year={year}&month={month}'

        try:
            response = requests.get(url)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            data = response.json()  # Parse JSON response into a dictionary
            output_data = {
                "Result": [
                {
                    "date": holiday['date']['iso'],
                    "name": holiday['name']
                }
                for holiday in data['response']['holidays']
            ]
            }

            return output_data
        except requests.exceptions.HTTPError as http_err:
            print(f'HTTP error occurred: {http_err}')
        except requests.exceptions.RequestException as req_err:
            print(f'Request error occurred: {req_err}')
        except json.JSONDecodeError as json_err:
            print(f'Error decoding JSON: {json_err}')
        except Exception as err:
            print(f'An error occurred: {err}')

        return {"Result": []}  # Return an empty result in case of error

    @tool
    def organize_the_answer(add : bool, reason : str ) -> dict:
        """
        after answer the question : Based on the previous holiday lists, the holidays on the list for that mont?. and use this tool to organize the answer.
        add : If the answer is  included in the list of holidays then return true ,  otherwise, it returns false.
        reason : Describe why you do or do not want to add a new holiday, specify whether the holiday already exists in the list, and the contents of the current list.
        """
        output_data = {
            "Result":
            {
                "add": add,
                "reason": reason
            }
        }

        return output_data


    tools = [check_calendarific,organize_the_answer]

    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature'],
            # model_kwargs={ "response_format": { "type": "json_object" } }
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

    # Define the templates
    system_prompt = SystemMessagePromptTemplate(
        prompt=PromptTemplate(input_variables=[], 
            template="You are a helpful assistant,  provide the response in JSON format. "
            "the output data sturctured was like the examples, provide the json data structure")
    )

    # Placeholder for chat history
    chat_history_placeholder = MessagesPlaceholder(variable_name='chat_history', optional=True)

    # Human message template
    human_message_template = HumanMessagePromptTemplate(
        prompt=PromptTemplate(input_variables=['input'], template="{input}")
    )

    # Placeholder for agent scratchpad
    agent_scratchpad_placeholder = MessagesPlaceholder(variable_name='agent_scratchpad')

    # Combine the parts into a full prompt
    final_prompt = system_prompt + chat_history_placeholder + human_message_template + agent_scratchpad_placeholder

    # print(final_prompt.messages)


    agent = create_openai_functions_agent(llm = llm, tools = tools, prompt = final_prompt)
    agent_executor = AgentExecutor(agent = agent, tools = tools, verbose = True)

    #Adding in memory
    store = {}
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    message2 = HumanMessage(
            content=[
                {"type": "text", "text": question2},
            ]
    )

    message3 = HumanMessage(
            content=[
                {"type": "text", "text": question3},
            ]
    )


    response = agent_with_chat_history.invoke({"input" : message2 }, config={"configurable": {"session_id" : "<foo>"}},)
    # print(response)

    response = agent_with_chat_history.invoke({"input" : message3}, config={"configurable": {"session_id" : "<foo>"}},)
    # print(response)

    return response["output"].strip('```json\n').strip('```')


def generate_hw04(question):

    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )



    class ImageInformation(BaseModel):
        score : int = Field(description="填入數字")

    class JsonOutput(BaseModel):
        Result : ImageInformation

    parser = JsonOutputParser(pydantic_object=JsonOutput)

    # encode image
    base64_image = base64.b64encode(open("./baseball.png", 'rb').read()).decode('utf-8')


    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
    )

    system_prompt = SystemMessagePromptTemplate(
        prompt=PromptTemplate(input_variables=[], 
            template=
            """
            You are a helpful assistant,  provide the response in JSON format. "
            the data structure 
            """)
    )

    prompt_template = PromptTemplate(
        template="請解析以下問題並只以 JSON 格式返回結果：{format_instructions}",
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain = llm

    response = chain.invoke([prompt_template.format(),message]).content
    return response.strip('```json\n').strip('```')

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