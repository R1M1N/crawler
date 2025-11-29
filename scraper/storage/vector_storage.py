"""
Vector Storage for RAG Applications

Provides vector storage capabilities using various open-source vector databases
including ChromaDB, FAISS, and Qdrant for storing and retrieving embeddings.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import hashlib

import chromadb
from chromadb.config import Settings
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import qdrant_client
from qdrant_client.http import models

from ..utils.logger import get_logger
from ..utils.config import VectorStorageConfig

logger = get_logger(__name__)


class VectorStorage:
    """Vector storage with multiple backend support."""
    
    def __init__(self, config: Optional[VectorStorageConfig] = None):
        self.config = config or VectorStorageConfig()
        
        # Initialize embedding model
        self.embedding_model = None
        self.embedding_dim = None
        
        # Vector database clients
        self.chroma_client = None
        self.chroma_collection = None
        self.faiss_index = None
        self.faiss_metadata = []
        self.qdrant_client = None
        self.qdrant_collection = None
        
        # Storage backend
        self.backend = self.config.backend.lower()
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
        
    async def initialize(self):
        """Initialize the vector storage system."""
        try:
            # Load embedding model
            await self._load_embedding_model()
            
            # Initialize vector database based on backend
            if self.backend == 'chromadb':
                await self._initialize_chromadb()
            elif self.backend == 'faiss':
                await self._initialize_faiss()
            elif self.backend == 'qdrant':
                await self._initialize_qdrant()
            else:
                raise ValueError(f"Unsupported vector storage backend: {self.backend}")
                
            logger.info(f"Vector storage initialized with {self.backend} backend")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector storage: {e}")
            raise
            
    async def _load_embedding_model(self):
        """Load the sentence transformer model for embeddings."""
        try:
            model_name = self.config.embedding_model or 'all-MiniLM-L6-v2'
            logger.info(f"Loading embedding model: {model_name}")
            
            self.embedding_model = SentenceTransformer(model_name)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            
            logger.info(f"Embedding model loaded with dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
            
    async def _initialize_chromadb(self):
        """Initialize ChromaDB."""
        try:
            # Create ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.config.storage_path / 'chromadb')
            )
            
            # Create or get collection
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info("ChromaDB initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
            
    async def _initialize_faiss(self):
        """Initialize FAISS index."""
        try:
            # Create FAISS index
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
            
            # Normalize vectors for cosine similarity
            self.faiss_index = faiss.IndexIDMap(self.faiss_index)
            
            # Initialize metadata storage
            self.faiss_metadata = []
            
            logger.info("FAISS index initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize FAISS: {e}")
            raise
            
    async def _initialize_qdrant(self):
        """Initialize Qdrant client."""
        try:
            # Create Qdrant client
            if self.config.qdrant_url:
                self.qdrant_client = qdrant_client.QdrantClient(url=self.config.qdrant_url)
            else:
                # Use in-memory client
                self.qdrant_client = qdrant_client.QdrantClient(":memory:")
                
            # Create or get collection
            collection_info = self.qdrant_client.get_collection(self.config.collection_name)
            
            if collection_info is None:
                # Create new collection
                self.qdrant_client.create_collection(
                    collection_name=self.config.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.embedding_dim,
                        distance=models.Distance.COSINE
                    )
                )
                
            self.qdrant_collection = self.config.collection_name
            
            logger.info("Qdrant initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            raise
            
    async def add_documents(self, 
                          documents: List[Dict[str, Any]],
                          metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Add documents to the vector storage.
        
        Args:
            documents: List of document dictionaries with 'text' and optional 'id'
            metadata: Additional metadata to include with all documents
            
        Returns:
            List of document IDs
        """
        try:
            # Generate embeddings
            texts = [doc.get('text', '') for doc in documents]
            embeddings = await self._generate_embeddings(texts)
            
            # Prepare metadata
            doc_metadata = []
            doc_ids = []
            
            for i, doc in enumerate(documents):
                # Generate or use provided ID
                doc_id = doc.get('id') or self._generate_document_id(doc, i)
                doc_ids.append(doc_id)
                
                # Combine metadata
                combined_metadata = {
                    **doc.get('metadata', {}),
                    **(metadata or {}),
                    'added_at': datetime.now().isoformat(),
                    'text_length': len(doc.get('text', '')),
                    'chunk_index': i
                }
                doc_metadata.append(combined_metadata)
                
            # Add to vector database
            if self.backend == 'chromadb':
                await self._add_to_chromadb(doc_ids, texts, embeddings, doc_metadata)
            elif self.backend == 'faiss':
                await self._add_to_faiss(doc_ids, texts, embeddings, doc_metadata)
            elif self.backend == 'qdrant':
                await self._add_to_qdrant(doc_ids, texts, embeddings, doc_metadata)
                
            logger.info(f"Added {len(documents)} documents to vector storage")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
            
    async def search(self, 
                    query: str,
                    limit: int = 10,
                    filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            limit: Maximum number of results
            filters: Metadata filters
            
        Returns:
            List of search results with scores and metadata
        """
        try:
            # Generate query embedding
            query_embedding = await self._generate_embeddings([query])
            
            # Search based on backend
            if self.backend == 'chromadb':
                results = await self._search_chromadb(query_embedding[0], limit, filters)
            elif self.backend == 'faiss':
                results = await self._search_faiss(query_embedding[0], limit, filters)
            elif self.backend == 'qdrant':
                results = await self._search_qdrant(query_embedding[0], limit, filters)
            else:
                raise ValueError(f"Unsupported backend for search: {self.backend}")
                
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
            
    async def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific document by ID."""
        try:
            if self.backend == 'chromadb':
                return await self._get_chromadb_document(doc_id)
            elif self.backend == 'faiss':
                return await self._get_faiss_document(doc_id)
            elif self.backend == 'qdrant':
                return await self._get_qdrant_document(doc_id)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to get document {doc_id}: {e}")
            return None
            
    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document from storage."""
        try:
            if self.backend == 'chromadb':
                self.chroma_collection.delete(ids=[doc_id])
            elif self.backend == 'faiss':
                await self._delete_faiss_document(doc_id)
            elif self.backend == 'qdrant':
                self.qdrant_client.delete(
                    collection_name=self.qdrant_collection,
                    points_selector=models.PointIdsList(points=[doc_id])
                )
            else:
                return False
                
            logger.info(f"Deleted document: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False
            
    async def update_document(self, 
                            doc_id: str, 
                            text: str,
                            metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update a document in storage."""
        try:
            # Generate new embedding
            embedding = await self._generate_embeddings([text])
            
            # Update based on backend
            if self.backend == 'chromadb':
                self.chroma_collection.update(
                    ids=[doc_id],
                    documents=[text],
                    embeddings=[embedding[0]],
                    metadatas=[metadata or {}]
                )
            elif self.backend == 'faiss':
                await self._update_faiss_document(doc_id, text, embedding[0], metadata)
            elif self.backend == 'qdrant':
                self.qdrant_client.upsert(
                    collection_name=self.qdrant_collection,
                    points=[models.PointStruct(
                        id=doc_id,
                        vector=embedding[0],
                        payload={**metadata, 'text': text}
                    )]
                )
                
            logger.info(f"Updated document: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update document {doc_id}: {e}")
            return False
            
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            if self.backend == 'chromadb':
                collection_info = self.chroma_collection.count()
                return {
                    'document_count': collection_info,
                    'backend': 'chromadb',
                    'collection_name': self.config.collection_name
                }
            elif self.backend == 'faiss':
                return {
                    'document_count': len(self.faiss_metadata),
                    'backend': 'faiss',
                    'embedding_dimension': self.embedding_dim
                }
            elif self.backend == 'qdrant':
                collection_info = self.qdrant_client.get_collection(self.qdrant_collection)
                return {
                    'document_count': collection_info.vectors_count,
                    'backend': 'qdrant',
                    'collection_name': self.config.collection_name
                }
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}
            
    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        try:
            # Use the sentence transformer model
            embeddings = self.embedding_model.encode(
                texts,
                convert_to_tensor=False,
                show_progress_bar=False
            )
            
            # Convert to list of lists for compatibility
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
            
    def _generate_document_id(self, doc: Dict[str, Any], index: int) -> str:
        """Generate a unique document ID."""
        content = doc.get('text', '') + json.dumps(doc.get('metadata', {}), sort_keys=True)
        hash_obj = hashlib.md5(content.encode())
        return f"doc_{hash_obj.hexdigest()}_{index}"
        
    # ChromaDB specific methods
    async def _add_to_chromadb(self, doc_ids: List[str], texts: List[str], 
                              embeddings: List[List[float]], metadatas: List[Dict[str, Any]]):
        """Add documents to ChromaDB."""
        self.chroma_collection.add(
            ids=doc_ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
    async def _search_chromadb(self, query_embedding: List[float], 
                             limit: int, filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Search ChromaDB."""
        results = self.chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=filters
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'score': results['distances'][0][i]
            })
            
        return formatted_results
        
    async def _get_chromadb_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document from ChromaDB."""
        try:
            result = self.chroma_collection.get(
                ids=[doc_id],
                include=['documents', 'metadatas', 'embeddings']
            )
            
            if result['ids']:
                return {
                    'id': result['ids'][0],
                    'text': result['documents'][0] if result['documents'] else '',
                    'metadata': result['metadatas'][0] if result['metadatas'] else {},
                    'embedding': result['embeddings'][0] if result['embeddings'] else None
                }
            return None
            
        except Exception:
            return None
            
    # FAISS specific methods
    async def _add_to_faiss(self, doc_ids: List[str], texts: List[str],
                          embeddings: List[List[float]], metadatas: List[Dict[str, Any]]):
        """Add documents to FAISS."""
        # Convert IDs to integers for FAISS
        int_ids = [int(doc_id.replace('doc_', ''), 16) for doc_id in doc_ids]
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings_array)
        
        # Add to index
        self.faiss_index.add_with_ids(embeddings_array, np.array(int_ids))
        
        # Store metadata
        for i, (doc_id, text, metadata) in enumerate(zip(doc_ids, texts, metadatas)):
            self.faiss_metadata.append({
                'id': doc_id,
                'int_id': int_ids[i],
                'text': text,
                'metadata': metadata
            })
            
    async def _search_faiss(self, query_embedding: List[float], 
                          limit: int, filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Search FAISS."""
        if self.faiss_index.ntotal == 0:
            return []
            
        # Normalize query embedding
        query_array = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_array)
        
        # Search
        scores, indices = self.faiss_index.search(query_array, limit)
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid result
                # Find metadata by int_id
                metadata = next((m for m in self.faiss_metadata if m['int_id'] == idx), None)
                if metadata:
                    results.append({
                        'id': metadata['id'],
                        'text': metadata['text'],
                        'metadata': metadata['metadata'],
                        'score': float(score)
                    })
                    
        return results
        
    async def _get_faiss_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document from FAISS."""
        metadata = next((m for m in self.faiss_metadata if m['id'] == doc_id), None)
        return metadata
        
    async def _update_faiss_document(self, doc_id: str, text: str, 
                                   embedding: List[float], metadata: Optional[Dict[str, Any]]):
        """Update document in FAISS."""
        # Find and update metadata
        for i, doc_metadata in enumerate(self.faiss_metadata):
            if doc_metadata['id'] == doc_id:
                doc_metadata['text'] = text
                if metadata:
                    doc_metadata['metadata'].update(metadata)
                break
                
        # Note: FAISS doesn't support in-place updates easily
        # In a production system, you might want to rebuild the index
        
    async def _delete_faiss_document(self, doc_id: str):
        """Delete document from FAISS."""
        # Remove from metadata
        self.faiss_metadata = [m for m in self.faiss_metadata if m['id'] != doc_id]
        # Note: FAISS doesn't support deletion easily
        # In production, consider rebuilding the index periodically
        
    # Qdrant specific methods
    async def _add_to_qdrant(self, doc_ids: List[str], texts: List[str],
                           embeddings: List[List[float]], metadatas: List[Dict[str, Any]]):
        """Add documents to Qdrant."""
        points = []
        for i, (doc_id, text, embedding, metadata) in enumerate(zip(doc_ids, texts, embeddings, metadatas)):
            points.append(models.PointStruct(
                id=doc_id,
                vector=embedding,
                payload={**metadata, 'text': text}
            ))
            
        self.qdrant_client.upsert(
            collection_name=self.qdrant_collection,
            points=points
        )
        
    async def _search_qdrant(self, query_embedding: List[float], 
                           limit: int, filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Search Qdrant."""
        search_result = self.qdrant_client.search(
            collection_name=self.qdrant_collection,
            query_vector=query_embedding,
            limit=limit,
            query_filter=filters
        )
        
        return [
            {
                'id': str(result.id),
                'text': result.payload.get('text', ''),
                'metadata': {k: v for k, v in result.payload.items() if k != 'text'},
                'score': result.score
            }
            for result in search_result
        ]
        
    async def _get_qdrant_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document from Qdrant."""
        try:
            points = self.qdrant_client.retrieve(
                collection_name=self.qdrant_collection,
                ids=[doc_id]
            )
            
            if points:
                point = points[0]
                return {
                    'id': str(point.id),
                    'text': point.payload.get('text', ''),
                    'metadata': {k: v for k, v in point.payload.items() if k != 'text'},
                    'embedding': None  # Qdrant doesn't return embeddings by default
                }
            return None
            
        except Exception:
            return None
            
    async def cleanup(self):
        """Cleanup resources."""
        try:
            if self.backend == 'qdrant' and self.qdrant_client:
                # Qdrant client doesn't need explicit cleanup
                pass
                
            logger.info("Vector storage cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            
    async def export_collection(self, output_path: Path) -> bool:
        """Export collection to file."""
        try:
            export_data = {
                'backend': self.backend,
                'collection_name': self.config.collection_name,
                'embedding_model': self.config.embedding_model,
                'documents': []
            }
            
            if self.backend == 'chromadb':
                # Get all documents from ChromaDB
                result = self.chroma_collection.get(include=['documents', 'metadatas', 'ids'])
                for i, doc_id in enumerate(result['ids']):
                    export_data['documents'].append({
                        'id': doc_id,
                        'text': result['documents'][i] if result['documents'] else '',
                        'metadata': result['metadatas'][i] if result['metadatas'] else {}
                    })
                    
            elif self.backend == 'faiss':
                export_data['documents'] = self.faiss_metadata
                
            elif self.backend == 'qdrant':
                # Get all documents from Qdrant
                # This is a simplified export - in practice you'd want pagination
                scroll_result = self.qdrant_client.scroll(
                    collection_name=self.qdrant_collection,
                    limit=10000
                )
                
                for point in scroll_result[0]:
                    export_data['documents'].append({
                        'id': str(point.id),
                        'text': point.payload.get('text', ''),
                        'metadata': {k: v for k, v in point.payload.items() if k != 'text'}
                    })
                    
            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Collection exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False