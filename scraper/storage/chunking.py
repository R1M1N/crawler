"""
Text Chunking for RAG Applications

Implements various chunking strategies optimized for retrieval augmented generation:
- Fixed-size chunking
- Sentence-based chunking
- Semantic/semantic similarity chunking
- Markdown-aware chunking
- Custom chunking strategies
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from pathlib import Path

import nltk
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    id: str
    text: str
    start_position: int
    end_position: int
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


class TextChunker:
    """Advanced text chunking for RAG applications."""
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 chunking_strategy: str = "semantic",
                 preserve_sentences: bool = True,
                 language: str = "en"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy
        self.preserve_sentences = preserve_sentences
        self.language = language
        
        # Initialize NLP tools
        self.nlp = None
        self.embedding_model = None
        self.sentence_tokenizer = None
        
        # Initialize language-specific tools
        self._initialize_nlp_tools()
        
    def _initialize_nlp_tools(self):
        """Initialize NLP processing tools."""
        try:
            # Load spaCy model for the specified language
            if self.language == "en":
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    logger.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
                    
            # Download NLTK data if needed
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
                
            # Load sentence transformer for semantic chunking
            if self.chunking_strategy == "semantic":
                try:
                    self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                except Exception as e:
                    logger.warning(f"Failed to load embedding model: {e}")
                    self.chunking_strategy = "sentence"  # Fallback
                    
            logger.info(f"Text chunker initialized with {self.chunking_strategy} strategy")
            
        except Exception as e:
            logger.error(f"Failed to initialize NLP tools: {e}")
            
    def chunk_text(self, 
                   text: str,
                   document_id: str,
                   metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """
        Chunk text using the configured strategy.
        
        Args:
            text: Text to chunk
            document_id: Unique identifier for the document
            metadata: Additional metadata to include with chunks
            
        Returns:
            List of Chunk objects
        """
        if not text.strip():
            return []
            
        metadata = metadata or {}
        
        if self.chunking_strategy == "fixed_size":
            return self._chunk_fixed_size(text, document_id, metadata)
        elif self.chunking_strategy == "sentence":
            return self._chunk_by_sentences(text, document_id, metadata)
        elif self.chunking_strategy == "semantic":
            return self._chunk_semantically(text, document_id, metadata)
        elif self.chunking_strategy == "markdown":
            return self._chunk_markdown_aware(text, document_id, metadata)
        elif self.chunking_strategy == "recursive":
            return self._chunk_recursive(text, document_id, metadata)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.chunking_strategy}")
            
    def _chunk_fixed_size(self, text: str, document_id: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Fixed-size chunking with overlap."""
        chunks = []
        
        # Clean text
        text = self._clean_text(text)
        
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # Calculate end position
            end = min(start + self.chunk_size, len(text))
            
            # Find last sentence boundary if preserving sentences
            if self.preserve_sentences:
                end = self._find_sentence_boundary(text, start, end)
                
            # Extract chunk text
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk = Chunk(
                    id=f"{document_id}_chunk_{chunk_index:04d}",
                    text=chunk_text,
                    start_position=start,
                    end_position=end,
                    metadata={
                        **metadata,
                        'chunking_strategy': 'fixed_size',
                        'chunk_index': chunk_index,
                        'chunk_size': len(chunk_text)
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
                
            # Move start position with overlap
            start = max(start + self.chunk_size - self.chunk_overlap, end)
            
        return chunks
        
    def _chunk_by_sentences(self, text: str, document_id: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Chunk text by sentences."""
        chunks = []
        
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        current_chunk = ""
        current_start = 0
        chunk_index = 0
        
        for i, sentence in enumerate(sentences):
            sentence_start = text.find(sentence, current_start)
            
            # Check if adding this sentence would exceed chunk size
            if current_chunk and len(current_chunk + " " + sentence) > self.chunk_size:
                # Create chunk with accumulated sentences
                if current_chunk.strip():
                    chunk = Chunk(
                        id=f"{document_id}_chunk_{chunk_index:04d}",
                        text=current_chunk.strip(),
                        start_position=current_start,
                        end_position=sentence_start,
                        metadata={
                            **metadata,
                            'chunking_strategy': 'sentence',
                            'chunk_index': chunk_index,
                            'sentence_count': current_chunk.count('.') + current_chunk.count('!') + current_chunk.count('?'),
                            'chunk_size': len(current_chunk)
                        }
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)
                current_chunk = overlap_text + " " + sentence
                current_start = sentence_start - len(overlap_text)
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                    current_start = sentence_start
                    
        # Add final chunk
        if current_chunk.strip():
            chunk = Chunk(
                id=f"{document_id}_chunk_{chunk_index:04d}",
                text=current_chunk.strip(),
                start_position=current_start,
                end_position=len(text),
                metadata={
                    **metadata,
                    'chunking_strategy': 'sentence',
                    'chunk_index': chunk_index,
                    'sentence_count': current_chunk.count('.') + current_chunk.count('!') + current_chunk.count('?'),
                    'chunk_size': len(current_chunk)
                }
            )
            chunks.append(chunk)
            
        return chunks
        
    def _chunk_semantically(self, text: str, document_id: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Semantic chunking using embeddings and similarity."""
        if not self.embedding_model:
            logger.warning("Embedding model not available, falling back to sentence chunking")
            return self._chunk_by_sentences(text, document_id, metadata)
            
        chunks = []
        
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        if len(sentences) <= 1:
            return self._chunk_fixed_size(text, document_id, metadata)
            
        # Get embeddings for all sentences
        try:
            embeddings = self.embedding_model.encode(sentences)
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return self._chunk_by_sentences(text, document_id, metadata)
            
        # Find semantic breakpoints
        breakpoints = self._find_semantic_breakpoints(embeddings)
        
        # Create chunks based on breakpoints
        chunk_index = 0
        start_pos = 0
        
        for breakpoint in breakpoints:
            if breakpoint > start_pos:
                # Find sentences in this chunk
                chunk_sentences = sentences[start_pos:breakpoint]
                chunk_text = " ".join(chunk_sentences)
                
                # Find positions in original text
                start_position = text.find(chunk_sentences[0]) if chunk_sentences else start_pos
                end_position = text.rfind(chunk_sentences[-1]) + len(chunk_sentences[-1]) if chunk_sentences else start_position + len(chunk_text)
                
                # Calculate average embedding for chunk
                chunk_embedding = np.mean(embeddings[start_pos:breakpoint], axis=0).tolist()
                
                chunk = Chunk(
                    id=f"{document_id}_chunk_{chunk_index:04d}",
                    text=chunk_text,
                    start_position=start_position,
                    end_position=end_position,
                    metadata={
                        **metadata,
                        'chunking_strategy': 'semantic',
                        'chunk_index': chunk_index,
                        'sentence_count': len(chunk_sentences),
                        'chunk_size': len(chunk_text),
                        'semantic_score': self._calculate_chunk_coherence(embeddings[start_pos:breakpoint])
                    },
                    embedding=chunk_embedding
                )
                chunks.append(chunk)
                chunk_index += 1
                
            start_pos = breakpoint
            
        # Add final chunk if needed
        if start_pos < len(sentences):
            chunk_sentences = sentences[start_pos:]
            chunk_text = " ".join(chunk_sentences)
            
            start_position = text.find(chunk_sentences[0]) if chunk_sentences else start_pos
            end_position = text.rfind(chunk_sentences[-1]) + len(chunk_sentences[-1]) if chunk_sentences else start_position + len(chunk_text)
            
            chunk_embedding = np.mean(embeddings[start_pos:], axis=0).tolist() if len(embeddings) > start_pos else None
            
            chunk = Chunk(
                id=f"{document_id}_chunk_{chunk_index:04d}",
                text=chunk_text,
                start_position=start_position,
                end_position=end_position,
                metadata={
                    **metadata,
                    'chunking_strategy': 'semantic',
                    'chunk_index': chunk_index,
                    'sentence_count': len(chunk_sentences),
                    'chunk_size': len(chunk_text),
                    'semantic_score': self._calculate_chunk_coherence(embeddings[start_pos:]) if len(embeddings) > start_pos else 1.0
                },
                embedding=chunk_embedding
            )
            chunks.append(chunk)
            
        return chunks
        
    def _chunk_markdown_aware(self, text: str, document_id: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Markdown-aware chunking that respects document structure."""
        chunks = []
        
        # Parse markdown structure
        lines = text.split('\n')
        current_section = ""
        current_section_start = 0
        chunk_index = 0
        section_level = 0
        
        for i, line in enumerate(lines):
            # Check for heading
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
            
            if heading_match:
                # Save previous section if it has content
                if current_section.strip():
                    chunk = Chunk(
                        id=f"{document_id}_chunk_{chunk_index:04d}",
                        text=current_section.strip(),
                        start_position=current_section_start,
                        end_position=current_section_start + len(current_section),
                        metadata={
                            **metadata,
                            'chunking_strategy': 'markdown',
                            'chunk_index': chunk_index,
                            'section_level': section_level,
                            'chunk_size': len(current_section),
                            'structure': 'section'
                        }
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    
                # Start new section
                current_section = line + '\n'
                current_section_start = text.find(line, current_section_start)
                section_level = len(heading_match.group(1))
                
            else:
                # Add line to current section
                current_section += line + '\n'
                
                # Check if section is getting too large
                if len(current_section) > self.chunk_size * 1.5:
                    # Try to find a natural break point
                    break_pos = self._find_natural_break(current_section, self.chunk_size)
                    
                    if break_pos > 0:
                        chunk = Chunk(
                            id=f"{document_id}_chunk_{chunk_index:04d}",
                            text=current_section[:break_pos].strip(),
                            start_position=current_section_start,
                            end_position=current_section_start + break_pos,
                            metadata={
                                **metadata,
                                'chunking_strategy': 'markdown',
                                'chunk_index': chunk_index,
                                'section_level': section_level,
                                'chunk_size': break_pos,
                                'structure': 'partial_section'
                            }
                        )
                        chunks.append(chunk)
                        chunk_index += 1
                        
                        # Continue with remainder
                        current_section = current_section[break_pos:]
                        
        # Add final section
        if current_section.strip():
            chunk = Chunk(
                id=f"{document_id}_chunk_{chunk_index:04d}",
                text=current_section.strip(),
                start_position=current_section_start,
                end_position=current_section_start + len(current_section),
                metadata={
                    **metadata,
                    'chunking_strategy': 'markdown',
                    'chunk_index': chunk_index,
                    'section_level': section_level,
                    'chunk_size': len(current_section),
                    'structure': 'final_section'
                }
            )
            chunks.append(chunk)
            
        return chunks
        
    def _chunk_recursive(self, text: str, document_id: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Recursive chunking that tries multiple strategies."""
        # First try markdown-aware chunking
        if self._is_markdown_like(text):
            chunks = self._chunk_markdown_aware(text, document_id, metadata)
            if len(chunks) > 1:
                return chunks
                
        # Then try sentence-based chunking
        if self.preserve_sentences:
            chunks = self._chunk_by_sentences(text, document_id, metadata)
            if len(chunks) > 1:
                return chunks
                
        # Fall back to fixed-size chunking
        return self._chunk_fixed_size(text, document_id, metadata)
        
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        try:
            if self.nlp:
                # Use spaCy for sentence segmentation
                doc = self.nlp(text)
                return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            else:
                # Use NLTK as fallback
                import nltk
                tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
                return [sent.strip() for sent in tokenizer.tokenize(text) if sent.strip()]
        except Exception:
            # Fallback to regex-based sentence splitting
            sentence_endings = r'[.!?]+'
            sentences = re.split(sentence_endings, text)
            return [sent.strip() for sent in sentences if sent.strip()]
            
    def _clean_text(self, text: str) -> str:
        """Clean text for chunking."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove excessive newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()
        
    def _find_sentence_boundary(self, text: str, start: int, end: int) -> int:
        """Find the best sentence boundary within the range."""
        # Look for sentence endings in the range
        search_range = text[start:end]
        
        # Find the last sentence ending
        sentence_endings = ['.', '!', '?']
        best_pos = -1
        
        for ending in sentence_endings:
            pos = search_range.rfind(ending)
            if pos > best_pos:
                best_pos = pos
                
        if best_pos > 0:
            # Include some context after the sentence ending
            return start + best_pos + 1
            
        return end
        
    def _find_semantic_breakpoints(self, embeddings: np.ndarray) -> List[int]:
        """Find semantic breakpoints using similarity analysis."""
        if len(embeddings) < 3:
            return [len(embeddings)]
            
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Find breakpoints where similarity drops significantly
        breakpoints = []
        window_size = 3
        
        for i in range(window_size, len(embeddings) - window_size):
            # Calculate average similarity in windows
            before_similarity = np.mean(similarity_matrix[i-window_size:i, i-window_size:i])
            after_similarity = np.mean(similarity_matrix[i:i+window_size, i:i+window_size])
            
            # Calculate cross-window similarity
            cross_similarity = np.mean(similarity_matrix[i-window_size:i, i:i+window_size])
            
            # If cross-similarity is much lower than within-window similarity, it's a breakpoint
            if cross_similarity < before_similarity * 0.7:
                breakpoints.append(i)
                
        return breakpoints
        
    def _calculate_chunk_coherence(self, embeddings: np.ndarray) -> float:
        """Calculate coherence score for a chunk based on embedding similarity."""
        if len(embeddings) < 2:
            return 1.0
            
        # Calculate average pairwise similarity
        similarity_matrix = cosine_similarity(embeddings)
        n = len(similarity_matrix)
        
        # Get upper triangle (excluding diagonal)
        similarities = []
        for i in range(n):
            for j in range(i+1, n):
                similarities.append(similarity_matrix[i][j])
                
        return np.mean(similarities) if similarities else 1.0
        
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get overlap text from the end of a chunk."""
        if len(text) <= overlap_size:
            return text
            
        # Find a good break point for overlap
        sentences = self._split_into_sentences(text)
        overlap_text = ""
        
        for sentence in reversed(sentences):
            if len(overlap_text + sentence) <= overlap_size:
                overlap_text = sentence + " " + overlap_text
            else:
                break
                
        return overlap_text.strip()
        
    def _find_natural_break(self, text: str, max_size: int) -> int:
        """Find a natural break point in text."""
        if len(text) <= max_size:
            return len(text)
            
        # Try to break at sentence boundaries
        search_text = text[:max_size]
        sentences = self._split_into_sentences(search_text)
        
        if len(sentences) > 1:
            # Find the best sentence break
            cumulative_length = 0
            for i, sentence in enumerate(sentences):
                sentence_length = len(sentence) + 1  # +1 for space
                if cumulative_length + sentence_length > max_size:
                    break
                cumulative_length += sentence_length
            return cumulative_length
            
        # Fall back to word boundary
        words = search_text.split()
        length = 0
        for word in words:
            word_length = len(word) + 1  # +1 for space
            if length + word_length > max_size:
                break
            length += word_length
            
        return min(length, max_size)
        
    def _is_markdown_like(self, text: str) -> bool:
        """Check if text appears to be markdown formatted."""
        markdown_indicators = [
            r'^#{1,6}\s+',  # Headings
            r'\*\*.*?\*\*',  # Bold text
            r'\*.*?\*',      # Italic text
            r'`.*?`',        # Inline code
            r'```',          # Code blocks
            r'^\s*[-*+]\s',  # List items
            r'^\s*\d+\.\s',  # Numbered lists
        ]
        
        lines = text.split('\n')
        markdown_lines = 0
        
        for line in lines[:10]:  # Check first 10 lines
            for pattern in markdown_indicators:
                if re.match(pattern, line.strip()):
                    markdown_lines += 1
                    break
                    
        return markdown_lines / min(len(lines), 10) > 0.3  # 30% markdown indicators
        
    def optimize_chunking_for_query(self, 
                                   chunks: List[Chunk], 
                                   query: str,
                                   top_k: int = 5) -> List[Chunk]:
        """
        Optimize chunking for a specific query by merging similar chunks.
        
        Args:
            chunks: List of chunks
            query: Query text
            top_k: Number of top chunks to consider
            
        Returns:
            List of optimized chunks
        """
        if not self.embedding_model or len(chunks) < 2:
            return chunks
            
        try:
            # Get query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Calculate similarities
            similarities = []
            for chunk in chunks:
                if chunk.embedding:
                    similarity = cosine_similarity([query_embedding[0]], [chunk.embedding])[0][0]
                    similarities.append((similarity, chunk))
                else:
                    similarities.append((0.0, chunk))
                    
            # Sort by similarity
            similarities.sort(reverse=True, key=lambda x: x[0])
            
            # Take top chunks and try to merge nearby similar chunks
            top_chunks = [item[1] for item in similarities[:top_k]]
            
            # Sort top chunks by original position
            top_chunks.sort(key=lambda x: x.start_position)
            
            # Merge adjacent chunks if they're similar and not too large
            merged_chunks = []
            current_chunk = None
            
            for chunk in top_chunks:
                if current_chunk is None:
                    current_chunk = chunk
                else:
                    # Check if chunks should be merged
                    should_merge = (
                        len(current_chunk.text) + len(chunk.text) <= self.chunk_size * 1.5 and
                        abs(chunk.start_position - current_chunk.end_position) < 100  # Close positions
                    )
                    
                    if should_merge:
                        # Merge chunks
                        current_chunk.text += " " + chunk.text
                        current_chunk.end_position = chunk.end_position
                        current_chunk.metadata['merged_with'] = chunk.id
                    else:
                        merged_chunks.append(current_chunk)
                        current_chunk = chunk
                        
            if current_chunk:
                merged_chunks.append(current_chunk)
                
            return merged_chunks
            
        except Exception as e:
            logger.error(f"Failed to optimize chunking: {e}")
            return chunks
            
    def get_chunking_stats(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Get statistics about the chunking results."""
        if not chunks:
            return {}
            
        chunk_sizes = [len(chunk.text) for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'total_characters': sum(chunk_sizes),
            'average_chunk_size': np.mean(chunk_sizes),
            'median_chunk_size': np.median(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'std_chunk_size': np.std(chunk_sizes),
            'chunking_strategy': self.chunking_strategy,
            'chunk_overlap': self.chunk_overlap,
            'target_chunk_size': self.chunk_size
        }