import os
import re
import shutil
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

from rag.config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL


def _infer_city_from_source(source_path: str) -> str:
	source_lower = str(source_path).lower()
	if "swm_rules_2016" in source_lower or "solid waste management rules" in source_lower:
		return "national"
	if "bbmp" in source_lower or "bengaluru" in source_lower or "bangalore" in source_lower:
		return "bangalore"
	if "mcgm" in source_lower or "mumbai" in source_lower or "bmc" in source_lower:
		return "mumbai"
	if "ndmc" in source_lower or "delhi" in source_lower or "mcd" in source_lower:
		return "delhi"
	if "chennai" in source_lower or "gcc" in source_lower:
		return "chennai"
	return "unknown"


def _load_documents(pdfs_dir: Path):
	# Preferred path: pre-extracted text files from PDFs.
	txt_loader = DirectoryLoader(
		str(pdfs_dir),
		glob="*.txt",
		loader_cls=TextLoader,
		loader_kwargs={"encoding": "utf-8"},
	)
	txt_docs = txt_loader.load()
	if txt_docs:
		print(f"Loaded {len(txt_docs)} documents from .txt files")
		return txt_docs

	# Fallback path: raw PDFs if text files are not available yet.
	pdf_loader = PyPDFDirectoryLoader(str(pdfs_dir))
	pdf_docs = pdf_loader.load()
	print(f"Loaded {len(pdf_docs)} documents from .pdf files")
	return pdf_docs


def _normalize_regulatory_text(text: str) -> str:
	"""Insert breakpoints around rule/section markers to improve semantic chunk boundaries."""
	if not text:
		return ""

	normalized = text.replace("\r\n", "\n")
	normalized = re.sub(r"(?i)(?<!\n)(\b(rule|section|chapter|clause)\s+\d+(?:\.\d+)*)", r"\n\1", normalized)
	normalized = re.sub(r"\n{3,}", "\n\n", normalized)
	return normalized.strip()


def ingest_pdfs(reset_store: bool = False) -> int:
	project_root = Path(__file__).resolve().parent.parent
	pdf_dir = project_root / "pdfs"
	persist_dir = os.getenv("CHROMA_PERSIST_DIR")

	if not persist_dir:
		raise ValueError("CHROMA_PERSIST_DIR environment variable is required")

	persist_path = Path(persist_dir)
	if not persist_path.is_absolute():
		persist_path = project_root / persist_path

	if reset_store and persist_path.exists():
		for item in persist_path.iterdir():
			if item.is_dir():
				shutil.rmtree(item)
			else:
				item.unlink()
		print(f"Reset existing vector store contents at: {persist_path}")

	documents = _load_documents(pdf_dir)
	for doc in documents:
		doc.page_content = _normalize_regulatory_text(doc.page_content)

	splitter = RecursiveCharacterTextSplitter(
		chunk_size=CHUNK_SIZE,
		chunk_overlap=CHUNK_OVERLAP,
		separators=["\n\n", "\n", ". ", " ", ""],
	)
	chunks = splitter.split_documents(documents)

	# Attach normalized city metadata for strict city-aware filtering at query time.
	for chunk in chunks:
		source = chunk.metadata.get("source", "")
		chunk.metadata["city"] = _infer_city_from_source(source)

	embeddings = HuggingFaceEmbeddings(
		model_name=EMBEDDING_MODEL,
		encode_kwargs={"normalize_embeddings": True},
	)
	vector_store = Chroma(
		persist_directory=str(persist_path),
		embedding_function=embeddings,
	)
	vector_store.add_documents(chunks)

	print(f"Total number of chunks created: {len(chunks)}")
	return len(chunks)


if __name__ == "__main__":
	# Set RAG_REBUILD=1 to reset and rebuild the vector store from scratch.
	rebuild = os.getenv("RAG_REBUILD", "0") == "1"
	ingest_pdfs(reset_store=rebuild)
