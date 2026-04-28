"""저장소 - ChromaDB 벡터 저장소 + BM25 역색인"""

import json
import logging
import pickle
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

from app.config import settings
from app.models import Chunk, ChunkMetadata

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB 기반 벡터 저장소"""

    def __init__(self) -> None:
        persist_dir = Path(settings.CHROMA_PERSIST_DIR)
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB 초기화 완료: %s", persist_dir)

    @property
    def collection(self) -> chromadb.Collection:
        return self._collection

    def add_chunks(
        self, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> None:
        """청크와 임베딩을 ChromaDB에 저장"""
        if not chunks:
            return
        ids = [c.metadata.chunk_id for c in chunks]
        documents = [c.content for c in chunks]
        metadatas = [
            {
                "doc_id": c.metadata.doc_id,
                "page_num": c.metadata.page_num,
                "chunk_type": c.metadata.chunk_type,
                "image_path": c.metadata.image_path or "",
            }
            for c in chunks
        ]
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info("ChromaDB에 %d개 청크 저장 완료", len(chunks))

    def search(
        self,
        query_embedding: list[float],
        doc_id: str | None = None,
        top_k: int = 5,
    ) -> list[tuple[str, float, str, dict]]:
        """벡터 유사도 검색 → (chunk_id, distance, content, metadata) 리스트"""
        where = {"doc_id": doc_id} if doc_id else None
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        items: list[tuple[str, float, str, dict]] = []
        if not results["ids"] or not results["ids"][0]:
            return items
        for cid, dist, doc, meta in zip(
            results["ids"][0],
            results["distances"][0],
            results["documents"][0],
            results["metadatas"][0],
        ):
            items.append((cid, dist, doc, meta))
        return items

    def delete_document(self, doc_id: str) -> None:
        """특정 문서의 모든 청크 삭제"""
        self._collection.delete(where={"doc_id": doc_id})
        logger.info("ChromaDB에서 문서 %s 삭제 완료", doc_id)

    def get_document_chunk_count(self, doc_id: str) -> int:
        """문서의 청크 수 반환"""
        result = self._collection.get(where={"doc_id": doc_id})
        return len(result["ids"])


class BM25Store:
    """BM25 기반 스파스 검색용 역색인"""

    def __init__(self) -> None:
        self._index_dir = Path(settings.BM25_INDEX_DIR)
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._corpus_file = self._index_dir / "corpus.pkl"

        self._corpus: dict[str, dict[str, list[str]]] = {}
        self._bm25: BM25Okapi | None = None
        self._chunk_id_order: list[str] = []
        self._load()

    def _load(self) -> None:
        if self._corpus_file.exists():
            with open(self._corpus_file, "rb") as f:
                self._corpus = pickle.load(f)
            self._rebuild_bm25()
            logger.info("BM25 인덱스 로딩 완료: %d개 문서", len(self._corpus))

    def _save(self) -> None:
        with open(self._corpus_file, "wb") as f:
            pickle.dump(self._corpus, f)

    def _rebuild_bm25(self) -> None:
        self._chunk_id_order = []
        all_tokens: list[list[str]] = []
        for chunks in self._corpus.values():
            for chunk_id, tokens in chunks.items():
                self._chunk_id_order.append(chunk_id)
                all_tokens.append(tokens)
        if all_tokens:
            self._bm25 = BM25Okapi(all_tokens)
        else:
            self._bm25 = None

    def add_documents(
        self,
        doc_id: str,
        chunks: list[Chunk],
        tokenized_texts: list[list[str]],
    ) -> None:
        """토큰화된 청크를 BM25 인덱스에 추가"""
        if doc_id in self._corpus:
            self.delete_document(doc_id)
        self._corpus[doc_id] = {}
        for chunk, tokens in zip(chunks, tokenized_texts):
            self._corpus[doc_id][chunk.metadata.chunk_id] = tokens
        self._rebuild_bm25()
        self._save()
        logger.info("BM25 인덱스에 문서 %s 추가: %d개 청크", doc_id, len(chunks))

    def search(
        self,
        tokenized_query: list[str],
        doc_id: str | None = None,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """BM25 검색 → (chunk_id, score) 리스트"""
        if self._bm25 is None or not self._chunk_id_order:
            return []
        scores = self._bm25.get_scores(tokenized_query)

        valid_chunks: set[str] | None = None
        if doc_id and doc_id in self._corpus:
            valid_chunks = set(self._corpus[doc_id].keys())

        results: list[tuple[str, float]] = []
        for idx, score in enumerate(scores):
            cid = self._chunk_id_order[idx]
            if valid_chunks is not None and cid not in valid_chunks:
                continue
            results.append((cid, float(score)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def delete_document(self, doc_id: str) -> None:
        if doc_id in self._corpus:
            del self._corpus[doc_id]
            self._rebuild_bm25()
            self._save()
            logger.info("BM25 인덱스에서 문서 %s 삭제 완료", doc_id)
