import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import PineconeHybridSearchRetriever
from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.integrations.langchain.llm_adapter import LangChainLLMAdapter
from nemoguardrails.rails.llm.options import RailStatus, RailType
import random
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langsmith import traceable
import wikipediaapi
 
load_dotenv()
 
 
# llm = the model that WRITES text (used in every prompt | llm | parser chain)
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.2)

# rails = the guardrails engine that CHECKS text (used only via rails.check(...) below)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
rails_config = RailsConfig.from_path(os.path.join(BASE_DIR, "config"))
rails = LLMRails(rails_config, llm=LangChainLLMAdapter(llm))

parser = StrOutputParser()
 
# --- Hybrid search setup (Pinecone) ---
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index("hybrid-search-rerank")
dense_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
 
wiki_wiki = wikipediaapi.Wikipedia(
    user_agent='YouTubeRAGChatbot/1.0 (shindeatharva0324@gamil.com)',
    language='en'
)
 
query_rewrite_prompt = ChatPromptTemplate.from_template(
    "Based on this question, output a short, specific Wikipedia search query "
    "(2-5 words, just the topic, no question format).\n\nQuestion: {question}\n\nSearch query:"
)
 
def generate_wiki_query(question):
    return (query_rewrite_prompt | llm | parser).invoke({"question": question}).strip()
 
@traceable(name="wikipedia_fetch", run_type="retriever")
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
     "4. Additionaly if no transcript is found for that particular video, say so and then use wikipedia and your paramteric knowledge to answer the questions.\n"
     "5. If neither source helps, say so politely."
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])
 
 
retrievers_by_video = {}     # was: retriever
chat_histories_by_video = {} # was: chat_history
 
 
def build_pinecone_hybrid_retriever(video_id, chunks, top_k=15, final_k=6, alpha=0.5, use_rerank=True):
    texts = [c.page_content for c in chunks]
    bm25_path = f"bm25_{video_id}.json"
    already_ingested = os.path.exists(bm25_path)
 
    if already_ingested:
        bm25 = BM25Encoder().load(bm25_path)
    else:
        bm25 = BM25Encoder()
        bm25.fit(texts)
 
    retriever = PineconeHybridSearchRetriever(
        embeddings=dense_model,
        sparse_encoder=bm25,
        index=pinecone_index,
        top_k=top_k,
        alpha=alpha,
        namespace=video_id,
    )
 
    if not already_ingested:
        retriever.add_texts(texts, namespace=video_id)
        bm25.dump(bm25_path)
 
    def retrieve(query):
        docs = retriever.invoke(query)
        if not docs or not use_rerank:
            return docs[:final_k]
        reranked = pc.inference.rerank(
            model="bge-reranker-v2-m3",
            query=query,
            top_n=final_k,
            documents=[{"id": str(i), "text": d.page_content} for i, d in enumerate(docs)],
            return_documents=True,
        )
        return [Document(page_content=item["document"]["text"]) for item in reranked.data]
 
    return RunnableLambda(retrieve).with_config(run_name="hybrid_retrieve")
 
 
def ingest_video(video_id):
 
    transcript = None
 
    try:
        yt_api = YouTubeTranscriptApi()
        transcript_list = yt_api.fetch(video_id, languages=['en'])
        transcript = " ".join([snippet.text for snippet in transcript_list])
    except (TranscriptsDisabled, NoTranscriptFound):
        print("no transcript available")
        return {"status": "error", "message": "no transcript available"}
 
    # chunks are needed every time (hybrid search fits BM25 on them)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
    for chunk in chunks:
        chunk.metadata['source'] = 'video_transcript'
 
    retriever = build_pinecone_hybrid_retriever(video_id, chunks)
 
    # saving this video's retriever and start its own chat history
    retrievers_by_video[video_id] = retriever
    chat_histories_by_video[video_id] = []
 
    return {"status": "success", "video_id": video_id}


# --- Guardrails: check the question before the chain, the answer after it ---
REFUSALS = [
    "Sorry, I can't help with that request.",
    "That's outside what I can help with. Feel free to ask me something about the video.",
    "I'm not able to assist with that one. Want to try a different question?",
    "I can't answer that, but I'm happy to help with anything about this video.",
    "That request isn't something I can help with. Let's get back to the video.",
    "I have to pass on that one. Is there something else about the video you'd like to know?",
]

def get_refusal():
    return random.choice(REFUSALS)

def input_allowed(question):
    result = rails.check(
        [{"role": "user", "content": question}],
        rail_types=[RailType.INPUT],
    )
    if result.status == RailStatus.BLOCKED:
        print(f"Input blocked by rail: {result.rail}")
        return False
    return True

def output_allowed(question, answer):
    result = rails.check(
        [{"role": "user", "content": question},
         {"role": "assistant", "content": answer}],
        rail_types=[RailType.OUTPUT],
    )
    if result.status == RailStatus.BLOCKED:
        print(f"Output blocked by rail: {result.rail}")
        return False
    return True
 
@traceable(name="youtube_rag_ask", run_type="chain")
def ask(video_id, question,return_context=False):
    # GUARDRAIL 1: check the question before anything runs
    if not input_allowed(question):
        return {"answer": get_refusal(), "context": []} if return_context else get_refusal()
    
    retriever = retrievers_by_video[video_id]
    chat_history = chat_histories_by_video[video_id]
    retrieved=[]
    def format_and_save(docs):
        retrieved.extend(d.page_content for d in docs)
        return format_doc(docs)
 
    base_chain = RunnableParallel({
        'video_context': retriever | RunnableLambda(format_and_save),
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

    # GUARDRAIL 2: check the answer before the user sees it
    if not output_allowed(question, answer):
        return {"answer": get_refusal(), "context": retrieved} if return_context else get_refusal()

    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))
    if return_context:
        return {"answer": answer, "context": retrieved}
    return answer