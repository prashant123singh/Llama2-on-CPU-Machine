from langchain import PromptTemplate
from langchain import LLMChain
from langchain.llms import CTransformers
from src.helper import *

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"



#instrutction = "Convert the following text from English to Hindi: \n\n {text}"
instrutction = "Give the proper summary of the of: \n\n {text}"

SYSTEM_PROMPT= B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS
template = B_INST + SYSTEM_PROMPT + instrutction + E_INST

prompt = PromptTemplate(template=template, input_variables=["text"])

llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type='llama',
                    config={
                        'max_new_tokens': 128,
                        'temperature':0.3
                    })


llmchain=LLMChain(prompt=prompt,llm=llm)

print(llmchain.run("Harry Potter"))
