# -*- coding: utf-8 -*-
"""Lab9_Langchain_LLM_agents.ipynb

"""

# Вначале необходимо установить используемые библиотеки.

# !pip list -v

#!pip install accelerate transformers

# !pip install --upgrade langchain langchain-huggingface langchain-community

# !pip install langgraph

# !pip install wikipedia

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import ChatHuggingFace
import torch
from modelscope import snapshot_download

# Можно использовать разные модели. Чем больше размер модели, тем в среднем лучше результат.
# Также необходимо помнить что некоторые модели специально дообучаются для использования в качесвте агентов,
# их учат использовать созданные для них инструменты и следовать инструкциям.

# Указываем имя модели и загружаем токенизатор и модель
model_name = "Qwen/Qwen3-4B"
model_dir = snapshot_download(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
    dtype=torch.float16,
)

# Создаем pipeline Hugging Face
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    temperature=0.3,
    top_p=0.95,
    repetition_penalty=1.2
)

# На основе pipeline Hugging Face создаем langchain pipeline и langchain chat model
# И то и то можно использовать для построения агентной системы. Основное отличие — структура ввода и вывода.
# pipeline работает принимает и выдает текст, тогда как chat model работает с запросами в виде диалогов.
llm = HuggingFacePipeline(pipeline=pipe)
chat_model = ChatHuggingFace(llm=llm)

prompt = "What is the capital of France?"
response = llm.invoke(prompt)
print(response)

# Ответ chat model содержит специальные теги начала и окончания фразы, метки указывающие кому принадлежит текст
# user - пользовательский контент, assistant -
# Для размышляющих моделей в диалоге может присутствовать тег think помечающий внутренние размышления агента.
prompt = "What is the capital of France?"
response = chat_model.invoke(prompt)
print(response)

# Построим простейшую RAG (Retrieval-Augmented Generation или генерация, дополненная поиском) систему
# Такая система помогает LLM использовать информацию из внешних источников для формирования ответа, а не только собственные знания.

# В качестве источника дополнительной информации будем использовать простой текстовый документ
from langchain_community.document_loaders import TextLoader

# Загружаем документ (убедитесь что путь указан правильно, также возможны проблемы с кодировкой)
loader = TextLoader("Article.txt", encoding='cp1251')
documents = loader.load()

print(f"Loaded {len(documents)} document(s).")

# Для разных версий langchain может меняться формат импорта нужных компонент

# from langchain.text_splitter import CharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter

# Большие объемы информации обычно обрабатываются кусками - чанками.
# Это связано с ограничениями контекстного окна моделей.
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

print(f"Split into {len(texts)} chunks.")

# Убедимся что информация загружена и нарезана, чанки содержат текст.
print(texts[0].page_content)

# Процесс RAG включает несколько этапов:
# Retrieval — поиск и извлечение фрагментов информации из базы знаний, которые наиболее релевантны запросу пользователя.
# Augmentation — дополнение запроса пользователя извлечённой информацией.
# Generation — генерация ответа языковой моделью с учётом дополнительной информации.
# Эти этапы еализованы в классе RetrievalQA
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

model_emb = "sentence-transformers/all-mpnet-base-v2"
emb_dir = snapshot_download(model_emb)
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
# решение актуально на 07.12.25
# далее необходимо !!!вручную!!! скопировать содержимое из каталога из переменной emb_dir
# в каталог C:\Users\ВАШПОЛЬЗОВАТЕЛЬ\.cache\huggingface\hub\models--sentence-transformers--all-mpnet-base-v2\snapshots\e8c3b32edf5434bc2275fc9bab85f82640a19130


# Сама информация в RAG системе хранится в векторном виде для облегчения поиска релевантных фрагментов.
# Для преобразования в векторный вид используюся Embedd-еры
embeddings = HuggingFaceEmbeddings()

# Преобразуем чанки в вектора, которые будут храниться в памяти.
# В реальной системе в памяти хранятся только наиболее часто используемы чанки,
# основной объем хранится в векторной базе данных.
# LangChain поддерживает различные векторные базы данных, например Chroma,
# Weaviate, Qdrant или база данных Яндекса YDB
vectorstore = InMemoryVectorStore.from_documents(texts, embeddings)

# retrievers сравнивают пользовательский запрос с векторизованными чанками (документами)
# и возвращают наиболее релевантные результаты.
# Для определения релевантности могут использоваться различные метрики, например, косинусная мера или совпадение ключевых слов.
retriever = vectorstore.as_retriever()


# Создадим цепочку формирования ответа на основе RAG
#
# Тип цепи "stuff" все данные от retriever добавляет в контекст запроса к языковой модели.
# Этот подход работает с небольшими фрагментами данных.
# При работе с большим количеством данных используются "refine" или "map_reduce", которые
# позволяют обрабатывать найденные документы каждый по отдельности уточняя и объединяя полученную из них информацию.
qa_chain = RetrievalQA.from_chain_type(
    chat_model,
    retriever=retriever,
    chain_type="stuff"
)

print("Question answering chain created.")

# Теперь можно формировать запросы, LLM будет изпользовать загруженные данные для ответа.
# Качество ответов зависит от многих факторов - от используемого Embedd-ера, метрики для поиска релевантных документов,
# поноты информации в самих документах и используемой LLM.
query = 'Какая технологическая компания России наиболее известна своим ИИ?'
result = qa_chain.invoke({"query": query})
print(result['result'])

# Еще один спостоб построения систем на базе LLM - это агенты.
# Основное отличие агентов - это возможность планирования своих действий и активного взаимодействия
# со специально созданными для них инструментами или другими агентами.

# В langchain есть встроенные инструменты, предоставляющие агентам возможность
# делать запросы к поисковым движкам или внешним сервисам, выполнять команды в терминале и т.д.
# Создадим инструмент для доступа к Википедии, чтобы агент мог
# верифицировать свой ответ.

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
wikitool = WikipediaQueryRun(api_wrapper=api_wrapper)

print(wikitool.invoke({"query": "Kolmogorov"}))

# Создадим список инструментов и агента с доступом к ним.
from langgraph.prebuilt import create_react_agent

tools = [wikitool]

agent_executor = create_react_agent(chat_model, tools)

# Ассистенты обрабатывают сообщения. У сообщений есть системная часть
# в которой описываеся его роль, инструкции
# и возможности использовать инструменты.

# Пользовательский запрос содержится в пользовательской части сообщения.
input_message = {
    "role": "user",
    "content": "Who is Komlogorov?"
}

result = agent_executor.invoke({"messages": [input_message]})
print(result["messages"][1])

# Для удобства вывода используется специальный метод
for message in result["messages"]:
  message.pretty_print()

# Системный промпт можно создать самостоятельно.
system_prompt = (
    f"Ты полезный ассистент. Используй {tools} для поиска информации и проверки своего ответа. В ответе явно укажи каким инструментом ты пользовался."
)

agent_executor = create_react_agent(chat_model, tools, prompt=system_prompt)

input_message = {
    "role": "user",
    "content": "Who is Komlogorov?"
}

result = agent_executor.invoke({"messages": [input_message]})

for message in result["messages"]:
  message.pretty_print()

# Инструменты можно создавать свои.
# Но нужно тщательно формировать описание инструмента, т.к.
# агент сам решает когда его использовать именно на основании описания.
from langchain.tools import tool

@tool("string_invert", description="Check query and return right query")
def string_invert(query: str) -> str:
    return query[::-1]

tools = [wikitool, string_invert]

# Агенты на основе маленьких LLM часто галюцинируют по поводу использования
# инструментов и могут подменять результаты работы инструмента своими галлюцинациями.
# Для таких LLM нужно четко прописывать системные промпты.

system_prompt = (
    f"Четко следуй инструкциям. Пользовательский запрос может содержать ошибки, его нужно проверять. Для проверки всегда используй {tools[1]}. Используй {tools[0]} для поиска информации. В ответе явно укажи ответ {tools[1]}."
)

agent_executor = create_react_agent(chat_model, tools, prompt=system_prompt)

input_message = {
    "role": "user",
    "content": "Who is vorogomloK?"
}

result = agent_executor.invoke({"messages": [input_message]})

for message in result["messages"]:
  message.pretty_print()

# Удачи в построении интеллектуальных систем!