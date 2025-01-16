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
    pass
    
def generate_hw03(question2, question3):
    pass

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
        Resule : ImageInformation

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