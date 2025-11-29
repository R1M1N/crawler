"""
OCR Processor

Uses Tesseract and other open-source OCR engines to extract text from images.
Supports multiple languages and image preprocessing.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import tempfile
import subprocess

import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np

from ..utils.logger import get_logger
from ..utils.config import OCRConfig

logger = get_logger(__name__)


class OCRProcessor:
    """OCR processor using Tesseract with image preprocessing."""
    
    def __init__(self, config: Optional[OCRConfig] = None):
        self.config = config or OCRConfig()
        
        # Tesseract languages to load
        self.languages = self.config.languages or ['eng']
        self.language_string = '+'.join(self.languages)
        
        # OCR engines and configurations
        self.engines = {
            'tesseract': self._tesseract_ocr,
            'opencv_tesseract': self._opencv_tesseract_ocr
        }
        
    async def extract_text(self, 
                          image_path: str,
                          language: Optional[str] = None,
                          engine: str = 'tesseract',
                          preprocess: bool = True) -> Dict[str, Any]:
        """
        Extract text from an image using OCR.
        
        Args:
            image_path: Path to image file
            language: Language code (overrides config default)
            engine: OCR engine to use
            preprocess: Whether to apply image preprocessing
            
        Returns:
            Dictionary with extracted text and metadata
        """
        start_time = datetime.now()
        
        try:
            # Load image
            image = Image.open(image_path)
            
            # Preprocess image if requested
            if preprocess:
                image = await self._preprocess_image(image)
                
            # Extract text using specified engine
            if engine not in self.engines:
                raise ValueError(f"Unknown OCR engine: {engine}")
                
            result = await self.engines[engine](image, language or self.language_string)
            
            # Add metadata
            result.update({
                'image_path': image_path,
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'engine': engine,
                'language': language or self.language_string,
                'processed_at': start_time.isoformat()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"OCR extraction failed for {image_path}: {e}")
            return {
                'text': '',
                'confidence': 0.0,
                'error': str(e),
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'processed_at': start_time.isoformat()
            }
            
    async def extract_text_batch(self, 
                               image_paths: List[str],
                               language: Optional[str] = None,
                               engine: str = 'tesseract',
                               preprocess: bool = True) -> List[Dict[str, Any]]:
        """
        Extract text from multiple images concurrently.
        
        Args:
            image_paths: List of image file paths
            language: Language code
            engine: OCR engine to use
            preprocess: Whether to apply image preprocessing
            
        Returns:
            List of OCR results
        """
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        async def process_single_image(image_path: str):
            async with semaphore:
                return await self.extract_text(image_path, language, engine, preprocess)
                
        # Process all images concurrently
        tasks = [process_single_image(path) for path in image_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log them
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process {image_paths[i]}: {result}")
            else:
                valid_results.append(result)
                
        return valid_results
        
    async def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image to improve OCR accuracy.
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed PIL Image
        """
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)
            
            # Apply slight sharpening
            image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=2))
            
            # Convert to OpenCV format for advanced preprocessing
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply bilateral filter to reduce noise while preserving edges
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Apply adaptive thresholding for better text recognition
            thresh = cv2.adaptiveThreshold(
                filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Convert back to PIL Image
            processed_image = Image.fromarray(cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB))
            
            return processed_image
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            return image  # Return original image if preprocessing fails
            
    async def _tesseract_ocr(self, image: Image.Image, language: str) -> Dict[str, Any]:
        """Standard Tesseract OCR."""
        try:
            # Configure Tesseract
            config = f'--oem 3 --psm 6 -l {language}'
            
            # Extract text
            text = pytesseract.image_to_string(image, config=config)
            
            # Get confidence data
            data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
            
            # Calculate average confidence
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Extract words with positions
            words = []
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 0:
                    words.append({
                        'text': data['text'][i],
                        'confidence': int(data['conf'][i]),
                        'left': data['left'][i],
                        'top': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i]
                    })
                    
            return {
                'text': text.strip(),
                'confidence': avg_confidence,
                'words': words,
                'word_count': len(words),
                'character_count': len(text.strip())
            }
            
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return {
                'text': '',
                'confidence': 0.0,
                'error': str(e),
                'words': []
            }
            
    async def _opencv_tesseract_ocr(self, image: Image.Image, language: str) -> Dict[str, Any]:
        """Advanced OCR with OpenCV preprocessing."""
        try:
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply multiple preprocessing techniques
            processed_images = []
            
            # 1. Original grayscale
            processed_images.append(('original', gray))
            
            # 2. Gaussian blur + threshold
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            _, thresh1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_images.append(('gaussian', thresh1))
            
            # 3. Morphological operations
            kernel = np.ones((2, 2), np.uint8)
            morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            processed_images.append(('morphological', morph))
            
            # 4. Canny edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            processed_images.append(('edges', edges))
            
            # Run OCR on each processed image and choose best result
            best_result = None
            best_score = 0
            
            for method_name, processed_img in processed_images:
                try:
                    # Convert back to PIL Image
                    pil_image = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB))
                    
                    # Run Tesseract
                    config = f'--oem 3 --psm 6 -l {language}'
                    text = pytesseract.image_to_string(pil_image, config=config)
                    
                    # Get confidence
                    data = pytesseract.image_to_data(pil_image, config=config, output_type=pytesseract.Output.DICT)
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                    
                    # Calculate score based on text length and confidence
                    score = len(text.strip()) * (avg_confidence / 100.0)
                    
                    if score > best_score:
                        best_score = score
                        best_result = {
                            'text': text.strip(),
                            'confidence': avg_confidence,
                            'method': method_name,
                            'words': [word for word in [
                                {
                                    'text': data['text'][i],
                                    'confidence': int(data['conf'][i]),
                                    'left': data['left'][i],
                                    'top': data['top'][i],
                                    'width': data['width'][i],
                                    'height': data['height'][i]
                                }
                                for i in range(len(data['text']))
                                if int(data['conf'][i]) > 0
                            ] if word['text'].strip()]
                        }
                        
                except Exception as e:
                    logger.debug(f"OCR failed for method {method_name}: {e}")
                    continue
                    
            if best_result:
                best_result.update({
                    'word_count': len(best_result['words']),
                    'character_count': len(best_result['text'])
                })
                return best_result
            else:
                return {
                    'text': '',
                    'confidence': 0.0,
                    'error': 'All OCR methods failed',
                    'words': []
                }
                
        except Exception as e:
            logger.error(f"OpenCV Tesseract OCR failed: {e}")
            return {
                'text': '',
                'confidence': 0.0,
                'error': str(e),
                'words': []
            }
            
    async def extract_text_with_layout(self, 
                                     image_path: str,
                                     language: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract text with layout information (paragraphs, lines, blocks).
        
        Args:
            image_path: Path to image file
            language: Language code
            
        Returns:
            Dictionary with structured text layout
        """
        try:
            image = Image.open(image_path)
            lang = language or self.language_string
            
            # Get detailed layout data
            config = f'--oem 3 --psm 6 -l {lang}'
            data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
            
            # Structure the data into paragraphs, lines, and blocks
            layout = {
                'paragraphs': [],
                'lines': [],
                'blocks': [],
                'words': []
            }
            
            current_block = None
            current_line = None
            
            for i in range(len(data['text'])):
                if int(data['conf'][i]) <= 0:
                    continue
                    
                word_data = {
                    'text': data['text'][i],
                    'confidence': int(data['conf'][i]),
                    'left': data['left'][i],
                    'top': data['top'][i],
                    'width': data['width'][i],
                    'height': data['height'][i],
                    'block_num': data['block_num'][i],
                    'par_num': data['par_num'][i],
                    'line_num': data['line_num'][i],
                    'word_num': data['word_num'][i]
                }
                
                layout['words'].append(word_data)
                
                # Group words into lines
                if current_line is None or word_data['line_num'] != current_line['line_num']:
                    if current_line:
                        layout['lines'].append(current_line)
                    current_line = {
                        'line_num': word_data['line_num'],
                        'words': [word_data],
                        'text': word_data['text'],
                        'confidence': word_data['confidence']
                    }
                else:
                    current_line['words'].append(word_data)
                    current_line['text'] += ' ' + word_data['text']
                    current_line['confidence'] = (current_line['confidence'] + word_data['confidence']) / 2
                    
                # Group lines into paragraphs
                if current_block is None or word_data['par_num'] != current_block['par_num']:
                    if current_block:
                        layout['blocks'].append(current_block)
                    current_block = {
                        'par_num': word_data['par_num'],
                        'lines': [current_line] if current_line else [],
                        'text': current_line['text'] if current_line else '',
                        'confidence': current_line['confidence'] if current_line else 0
                    }
                else:
                    if current_line:
                        current_block['lines'].append(current_line)
                        current_block['text'] += '\\n' + current_line['text']
                        current_block['confidence'] = (current_block['confidence'] + current_line['confidence']) / 2
                        
            # Add last line and block
            if current_line:
                layout['lines'].append(current_line)
            if current_block:
                layout['blocks'].append(current_block)
                
            # Generate paragraphs from blocks
            for block in layout['blocks']:
                layout['paragraphs'].append({
                    'text': block['text'].strip(),
                    'confidence': block['confidence'],
                    'line_count': len(block['lines'])
                })
                
            return {
                'text': ' '.join([p['text'] for p in layout['paragraphs']]),
                'layout': layout,
                'word_count': len(layout['words']),
                'paragraph_count': len(layout['paragraphs']),
                'line_count': len(layout['lines']),
                'block_count': len(layout['blocks'])
            }
            
        except Exception as e:
            logger.error(f"Layout extraction failed for {image_path}: {e}")
            return {
                'text': '',
                'layout': {'paragraphs': [], 'lines': [], 'blocks': [], 'words': []},
                'error': str(e)
            }
            
    async def detect_language(self, image_path: str) -> Dict[str, Any]:
        """
        Detect the language of text in an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with detected language and confidence
        """
        try:
            image = Image.open(image_path)
            
            # Use Tesseract's language detection
            config = '--oem 3 --psm 6'
            text = pytesseract.image_to_string(image, config=config)
            
            if not text.strip():
                return {
                    'language': 'unknown',
                    'confidence': 0.0,
                    'text_sample': ''
                }
                
            # Try to detect language using Tesseract
            # This is a simplified approach - in practice, you might want to use
            # a dedicated language detection library
            
            # Check for common language patterns
            language_indicators = {
                'eng': ['the', 'and', 'you', 'that', 'was', 'for', 'are'],
                'spa': ['el', 'la', 'de', 'que', 'y', 'a', 'en'],
                'fra': ['le', 'de', 'et', 'à', 'un', 'il', 'être'],
                'deu': ['der', 'die', 'und', 'in', 'den', 'von', 'zu'],
                'ita': ['il', 'di', 'che', 'e', 'la', 'per', 'un'],
                'por': ['o', 'a', 'de', 'e', 'do', 'da', 'em']
            }
            
            text_lower = text.lower()
            scores = {}
            
            for lang, indicators in language_indicators.items():
                score = sum(text_lower.count(word) for word in indicators)
                scores[lang] = score
                
            if scores:
                best_lang = max(scores, key=scores.get)
                max_score = scores[best_lang]
                total_score = sum(scores.values())
                
                confidence = (max_score / total_score * 100) if total_score > 0 else 0
                
                return {
                    'language': best_lang if max_score > 0 else 'unknown',
                    'confidence': confidence,
                    'text_sample': text[:200]  # First 200 characters
                }
            else:
                return {
                    'language': 'unknown',
                    'confidence': 0.0,
                    'text_sample': text[:200]
                }
                
        except Exception as e:
            logger.error(f"Language detection failed for {image_path}: {e}")
            return {
                'language': 'unknown',
                'confidence': 0.0,
                'error': str(e)
            }
            
    async def validate_tesseract(self) -> Dict[str, Any]:
        """Validate Tesseract installation and available languages."""
        try:
            # Check Tesseract version
            result = subprocess.run(['tesseract', '--version'], 
                                  capture_output=True, text=True)
            version_output = result.stdout if result.returncode == 0 else "Unknown"
            
            # Check available languages
            lang_result = subprocess.run(['tesseract', '--list-langs'], 
                                       capture_output=True, text=True)
            
            available_languages = []
            if lang_result.returncode == 0:
                available_languages = lang_result.stdout.strip().split('\\n')[1:]  # Skip first line
            
            # Check configured languages
            configured_languages = []
            for lang in self.languages:
                if lang in available_languages:
                    configured_languages.append(lang)
                else:
                    logger.warning(f"Configured language '{lang}' not available in Tesseract")
                    
            return {
                'available': True,
                'version': version_output.split('\\n')[0] if version_output else 'Unknown',
                'available_languages': available_languages,
                'configured_languages': configured_languages,
                'missing_languages': [lang for lang in self.languages if lang not in available_languages]
            }
            
        except Exception as e:
            logger.error(f"Tesseract validation failed: {e}")
            return {
                'available': False,
                'error': str(e)
            }