import os
import httpx
from anthropic import Anthropic

# 从环境变量中获取 API Key
api_key = os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    raise ValueError("未找到 API Key，请检查环境变量设置！")

# 初始化 Anthropic 客户端
client = Anthropic(
    api_key=api_key,
    base_url="http://43.106.115.130:3000", 
    http_client=httpx.Client(trust_env=False)
)

# 使用 Claude 原生的 messages 接口
response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "你好，请简单介绍一下你自己。"}
    ]
)

# 打印返回的文本内容
print(response.content[0].text)