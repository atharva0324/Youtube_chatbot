from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import wikipediaapi

load_dotenv()


model = OpenAIEmbeddings(model='text-embedding-3-small')
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.2)
parser = StrOutputParser()

wiki_wiki = wikipediaapi.Wikipedia(
    user_agent='YouTubeRAGChatbot/1.0 (your_email@example.com)',
    language='en'
)

query_rewrite_prompt = ChatPromptTemplate.from_template(
    "Based on this question, output a short, specific Wikipedia search query "
    "(2-5 words, just the topic, no question format).\n\nQuestion: {question}\n\nSearch query:"
)

def generate_wiki_query(question):
    return (query_rewrite_prompt | llm | parser).invoke({"question": question}).strip()

def safe_wiki_retrieve(question):
    try:
        search_query = generate_wiki_query(question)
        print(f"Wikipedia search query: {search_query}")
        page = wiki_wiki.page(search_query)
        if page.exists():
            return page.summary[:2000]
        return ""
    except Exception as e:
        print(f"Wikipedia retrieval failed: {e}")
        return ""

class ContextSufficiency(BaseModel):
    sufficient: bool = Field(description="True if video context is enough to answer well, False if outside info is needed")
    reason: str = Field(description="Brief one-line reason for the decision")

sufficiency_parser = PydanticOutputParser(pydantic_object=ContextSufficiency)

sufficiency_prompt = ChatPromptTemplate.from_template(
    "Given this video context and question, decide if the context is sufficient to answer the question well.\n\n"
    "Context: {video_context}\nQuestion: {question}\n\n{format_instructions}"
).partial(format_instructions=sufficiency_parser.get_format_instructions())

sufficiency_chain = sufficiency_prompt | llm | sufficiency_parser

def is_sufficient(inputs):
    result = sufficiency_chain.invoke(inputs)
    print(f"Sufficiency check: {result.sufficient} — {result.reason}")
    return result.sufficient

def format_doc(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

chat_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful AI assistant answering questions about a YouTube video.\n\n"
     "VIDEO TRANSCRIPT CONTEXT (primary source):\n{video_context}\n\n"
     "WIKIPEDIA CONTEXT (secondary, may be empty):\n{wiki_context}\n\n"
     "STRICT RULES:\n"
     "1. Always answer using the video transcript context as your primary source.\n"
     "2. Only add Wikipedia information if it adds something the video does NOT cover.\n"
     "3. If you use Wikipedia content, prefix that part with: 'Additional info (not from the video):'\n"
     "4. If neither source helps, say so politely."
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])


retrievers_by_video = {}     # was: retriever
chat_histories_by_video = {} # was: chat_history


def ingest_video(video_id):
   
    transcript = None

    try:
        yt_api = YouTubeTranscriptApi()
        transcript_list = yt_api.fetch(video_id, languages=['en'])
        transcript = " ".join([snippet.text for snippet in transcript_list])
    except (TranscriptsDisabled, NoTranscriptFound):
        print("no transcript available")
        return {"status": "error", "message": "no transcript available"}

    vector_store = Chroma(
        embedding_function=model,
        persist_directory="./chroma_db",
        collection_name=video_id
    )

    if vector_store._collection.count() == 0:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript])
        for chunk in chunks:
            chunk.metadata['source'] = 'video_transcript'
        vector_store.add_documents(chunks)

    retriever = vector_store.as_retriever(search_kwargs={'k': 4}, search_type='mmr')

    # save this video's retriever and start its own chat history
    retrievers_by_video[video_id] = retriever
    chat_histories_by_video[video_id] = []

    return {"status": "success", "video_id": video_id}


def ask(video_id, question):
    """
    Same logic as your notebook's chain + ask() function.
    Only difference: pulls the correct retriever and chat_history
    for THIS video_id, instead of using one global retriever/chat_history.
    """
    retriever = retrievers_by_video[video_id]
    chat_history = chat_histories_by_video[video_id]

    base_chain = RunnableParallel({
        'video_context': retriever | RunnableLambda(format_doc),
        'question': RunnablePassthrough(),
    })

    wiki_branch = RunnableBranch(
        (lambda x: not is_sufficient(x), RunnableLambda(lambda x: safe_wiki_retrieve(x['question']))),
        RunnableLambda(lambda x: "")
    )

    chain = (
        base_chain
        | RunnableLambda(lambda x: {**x, 'wiki_context': wiki_branch.invoke(x), 'chat_history': chat_history[-8:]})
        | chat_prompt
        | llm
        | parser
    )

    answer = chain.invoke(question)
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))
    return answer