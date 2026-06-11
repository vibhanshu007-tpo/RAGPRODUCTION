import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from dotenv import load_dotenv
import uuid
import os
from groq import Groq
import datetime

from llama_index.core.evaluation import answer_relevancy
from openai.types.admin.organization import admin_api_key_delete_response

from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import RAGQueryResult,RAGSearchResult,RAGUpsertResult,RAGChunkAndSrc

load_dotenv()
groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

inngest_client = inngest.Inngest(
    app_id = "rag_app",
    logger = logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)
@inngest_client.create_function(
    fn_id = "RAG: ingest PDF",
    trigger =inngest.TriggerEvent(event="rag/ingest_PDF")
)

async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)


    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payload = [{"source_id": source_id, "text": chunks[i]} for i in range(len(chunks))]
        QdrantStorage().upsert(ids, vecs, payload)
        return RAGUpsertResult(ingested=len(chunks))


    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    return ingested.model_dump()

@inngest_client.create_function(
    fn_id = "RAG: Query PDF",
    trigger =inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_ingest_query(ctx: inngest.Context):
    def _search(question: str, top_k: int=5) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search(query_vec, top_k)
        return RAGSearchResult(context=found["contexts"], sources=found["sources"])

@inngest_client.create_function(
    fn_id = "RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int=5) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search(query_vec, top_k)
        return RAGSearchResult(context=found["contexts"], sources=found["sources"])

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAGSearchResult)

    context_block = "\n\n".join(f"- {c}" for c in found.context)
    user_content = (
        "use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question:\n{question}\n"
        "Answer Concisely using the context above."
    )

    # adapter = ai.groq.Adapter(
    #     auth_key=os.getenv("GROQ_API_KEY"),
    #     model="llama-3.3-70b-versatile"
    # )
    # adapter = ai.openai.Adapter(
    #     auth_key=os.getenv("OPENAI_API_KEY"),
    #     model="gpt-4o-mini"
    # )

    # res = await ctx.step.ai.infer(
    #     "llm-answer",
    #     adapter=adapter,
    #     body={
    #         "max_tokens": 1024,
    #         "temperature": 0.2,
    #         "messages":[
    #             {"role": "system", "content": "you answer question using only the provided context."},
    #             {"role": "user", "content": user_content}
    #         ]
    #     }
    # )
    #
    # answer = res["choices"][0]["message"]["content"].strip()

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "you answer question using only the provided context."
            },
            {
                "role": "user",
                "content": user_content
            }
        ],
        temperature=0.2,
        max_tokens=1024
    )

    answer = response.choices[0].message.content.strip()
    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.context)}

app = FastAPI()

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])