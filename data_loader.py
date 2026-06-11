from openai import OpenAI, responses
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
load_dotenv()

# client = OpenAI()
# client = Groq()
# EMBED_MODEL = "text-embedding-3-large"
# EMBED_DIM = 3072

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)
EMBED_DIM = 384




splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)


def load_and_chunk_pdf(path: str):
    doc = PDFReader().load_data(file=path)
    texts = [d.text for d in doc if getattr(d, "text", None)]
    chunks = []
    for text in texts:
        chunks.extend(splitter.split_text(text))
    return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    # responses = client.embeddings.create(
    #     model=EMBED_MODEL,
    #     input=texts,
    # )

    vectors = embedding_model.encode(texts)
    return vectors.tolist()






    # return [item.embedding for item in responses.data]



