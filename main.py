import os
import boto3
import uuid
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from starlette.concurrency import run_in_threadpool

load_dotenv()
app = FastAPI(title="안전 신고 분석 API", version="0.0.1")

# S3 설정
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION", "ap-northeast-2"),
)
bucket_name = os.getenv("AWS_S3_BUCKET")

llm = ChatOpenAI(
    model="gpt-4.1-nano",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.1,
)


# 반환할 JSON 형식
class ReportAnalysis(BaseModel):
    title: str
    category: str
    description: str
    dangerDegree: Literal["하", "중", "상", "최상"]


examples = [
    {
        "title": "횡단보도 앞 도로 포트홀 발생",
        "category": "시설물파손",
        "description": "초등학교 앞 횡단보도 바로 앞 차도에 지름 약 30cm, 깊이 10cm 정도의 포트홀이 발생했습니다. 아침 등교시간과 하교시간에 차량들이 급하게 피해가면서 보행자 안전에 위험을 초래하고 있으며, 비 오는 날에는 물이 고여 더욱 위험해 보입니다. 특히 어린이들이 주로 이용하는 통학로이므로 신속한 보수가 필요합니다.",
        "dangerDegree": "상",
    },
    {
        "title": "아파트 단지 내 가로등 고장",
        "category": "시설물파손",
        "description": "아파트 단지 내 주요 산책로의 가로등 3개가 연속으로 꺼져 있어 야간에 매우 어둡습니다. 평소 주민들이 산책과 조깅을 즐기던 구간이지만, 현재는 시야 확보가 어려워 넘어짐 사고나 범죄에 노출될 위험이 있습니다. 특히 어르신들과 아이들이 자주 이용하는 구간이라 조속한 수리가 필요해 보입니다.",
        "dangerDegree": "중",
    },
]

# ReportAnalysis 기반으로 JSON 출력 파서 생성
parser = JsonOutputParser(pydantic_object=ReportAnalysis)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 안전한 마을을 만드는 AI 비서입니다.
            이미지를 분석하여 마을 주민의 자세한 안전 신고 정보를 작성해주세요.

            다음 예시들처럼 구체적인 크기, 위치, 위험요소, 이용자층 등을 포함하여 상세하게 작성해주세요:

            {examples}

            {format_instructions}
            """,
        ),
        ("user", [{"type": "image_url", "image_url": {"url": "{image_url}"}}]),
    ]
)


def upload_s3(image_data: bytes, content_type: str) -> str:
    file_name = f"ai/{uuid.uuid4()}"
    s3.put_object(
        Bucket=bucket_name, Key=file_name, Body=image_data, ContentType=content_type
    )
    # 유효기간 5분 URL 생성
    return s3.generate_presigned_url(
        "get_object", Params={"Bucket": bucket_name, "Key": file_name}, ExpiresIn=300
    )


@app.post("/analyze", response_model=ReportAnalysis)
async def analyze_image(file: UploadFile = File(...)):
    image_data = await file.read()
    image_url = await run_in_threadpool(upload_s3, image_data, file.content_type)
    # 동기 I/O 함수는 run_in_threadpool을 사용하여 별도 스레드에서 실행해야 한다

    chain = prompt | llm | parser

    # 이벤트 루프 블로킹 방지 ainvoke 사용
    return await chain.ainvoke(
        {
            "image_url": image_url,
            # format_instructions는 JSON 형식으로 응답하도록 지시
            "format_instructions": parser.get_format_instructions(),
            "examples": "\n".join([f"예시: {example}" for example in examples]),
        }
    )
