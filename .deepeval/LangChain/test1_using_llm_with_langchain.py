from langchain_ollama import ChatOllama

llm = ChatOllama(

        base_url="http://localhost:11434",
        model="qwen2.5-coder",
        temperature=0.5,
        max=250

)

output=llm.invoke("Why sky is blue")
print(output.content)

