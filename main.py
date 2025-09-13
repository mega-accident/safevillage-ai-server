from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from fastapi import FastAPI
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
app = FastAPI()

llm = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0.1)

template = "아래 질문에 대한 답변을 해주세요. {question}"
prompt = PromptTemplate.from_template(template=template)
chain = prompt | llm | StrOutputParser()


class Item(BaseModel):
    question: str


@app.post("/analyze")
def analyze_chat(question: Item):
    return chain.invoke({"question": question})
