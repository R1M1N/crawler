"""
Advanced Content Extractor

Handles extraction of various content types including:
- Text content from HTML, Markdown, PDFs
- Structured data from JSON, XML
- Tables and lists
- Code blocks
- Meta information
"""

import re
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from bs4 import BeautifulSoup, Tag
import markdown
from markdown.extensions import codehilite, toc, tables, fenced_code
import PyPDF2
import pandas as pd
from urllib.parse import urljoin

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ContentExtractor:
    """Advanced content extraction with multiple strategies."""
    
    def __init__(self):
        # Initialize markdown parser with useful extensions
        self.md_parser = markdown.Markdown(extensions=[
            'codehilite',
            'toc',
            'tables', 
            'fenced_code'
        ])
        
        # Common content selectors for different site types
        self.content_selectors = [
            'main', 'article', '[role="main"]', '.content', '.post-content',
            '.entry-content', '.article-content', '.post-body', '.article-body',
            '#content', '#main-content', '.main-content'
        ]
        
        # Unwanted content selectors
        self.unwanted_selectors = [
            'nav', 'header', 'footer', 'aside', '.sidebar', '.navigation',
            '.menu', '.breadcrumb', '.social-share', '.comments', '.related-posts',
            '.advertisement', '.ads', '.banner', '.popup', '.modal'
        ]
        
    def extract_from_html(self, html: str, base_url: str = "") -> Dict[str, Any]:
        """
        Extract structured content from HTML.
        
        Args:
            html: HTML content to parse
            base_url: Base URL for resolving relative links
            
        Returns:
            Dictionary with extracted content
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove unwanted elements
        for selector in self.unwanted_selectors:
            for element in soup.select(selector):
                element.decompose()
                
        # Extract main content
        main_content = self._extract_main_content(soup)
        
        # Extract metadata
        metadata = self._extract_metadata_from_html(soup)
        
        # Extract structured data
        structured_data = self._extract_structured_data(soup)
        
        # Extract tables
        tables = self._extract_tables(soup)
        
        # Extract lists
        lists = self._extract_lists(soup)
        
        # Extract code blocks
        code_blocks = self._extract_code_blocks(soup)
        
        # Extract images with descriptions
        images = self._extract_images_with_context(soup, base_url)
        
        # Extract links with context
        links = self._extract_links_with_context(soup, base_url)
        
        # Generate summary
        summary = self._generate_summary(main_content)
        
        return {
            'main_content': main_content,
            'metadata': metadata,
            'structured_data': structured_data,
            'tables': tables,
            'lists': lists,
            'code_blocks': code_blocks,
            'images': images,
            'links': links,
            'summary': summary,
            'word_count': len(main_content.split()),
            'extraction_timestamp': datetime.now().isoformat()
        }
        
    def extract_from_markdown(self, markdown_text: str) -> Dict[str, Any]:
        """
        Extract structured content from Markdown.
        
        Args:
            markdown_text: Markdown content to parse
            
        Returns:
            Dictionary with extracted content
        """
        # Parse markdown
        html = self.md_parser.convert(markdown_text)
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract structure
        structure = self._extract_markdown_structure(markdown_text)
        
        # Convert to HTML-based extraction
        html_result = self.extract_from_html(html)
        
        # Add markdown-specific data
        html_result.update({
            'markdown_structure': structure,
            'markdown_text': markdown_text,
            'sections': self._extract_sections(markdown_text)
        })
        
        return html_result
        
    def extract_from_pdf(self, pdf_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract content from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with extracted content
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                content = {
                    'pages': len(pdf_reader.pages),
                    'text_content': '',
                    'metadata': {},
                    'tables': [],
                    'images': []
                }
                
                # Extract text from all pages
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    content['text_content'] += f"\n--- Page {page_num + 1} ---\n{text}"
                    
                # Extract metadata
                if pdf_reader.metadata:
                    content['metadata'] = {
                        'title': pdf_reader.metadata.get('/Title', ''),
                        'author': pdf_reader.metadata.get('/Author', ''),
                        'subject': pdf_reader.metadata.get('/Subject', ''),
                        'creator': pdf_reader.metadata.get('/Creator', ''),
                        'producer': pdf_reader.metadata.get('/Producer', ''),
                        'creation_date': str(pdf_reader.metadata.get('/CreationDate', '')),
                        'modification_date': str(pdf_reader.metadata.get('/ModDate', ''))
                    }
                    
                # Extract tables (basic implementation)
                # Note: Full table extraction from PDF requires additional libraries like pdfplumber
                content['word_count'] = len(content['text_content'].split())
                content['extraction_timestamp'] = datetime.now().isoformat()
                
                return content
                
        except Exception as e:
            logger.error(f"Failed to extract content from PDF {pdf_path}: {e}")
            return {
                'error': str(e),
                'extraction_timestamp': datetime.now().isoformat()
            }
            
    def extract_from_json(self, json_data: Union[str, Dict]) -> Dict[str, Any]:
        """
        Extract structured content from JSON data.
        
        Args:
            json_data: JSON string or dictionary
            
        Returns:
            Dictionary with extracted content
        """
        if isinstance(json_data, str):
            try:
                data = json.loads(json_data)
            except json.JSONDecodeError as e:
                return {'error': f'Invalid JSON: {e}'}
        else:
            data = json_data
            
        # Analyze JSON structure
        analysis = self._analyze_json_structure(data)
        
        # Extract text content from JSON
        text_content = self._extract_text_from_json(data)
        
        # Extract tables/arrays that could be tabular data
        tables = self._extract_tables_from_json(data)
        
        return {
            'original_data': data,
            'structure_analysis': analysis,
            'text_content': text_content,
            'tables': tables,
            'data_types': self._analyze_data_types(data),
            'extraction_timestamp': datetime.now().isoformat()
        }
        
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract the main content from HTML."""
        # Try content selectors first
        for selector in self.content_selectors:
            content_element = soup.select_one(selector)
            if content_element:
                return self._clean_text(content_element.get_text())
                
        # Fallback to body content
        body = soup.find('body')
        if body:
            return self._clean_text(body.get_text())
            
        # Last resort: return all text
        return self._clean_text(soup.get_text())
        
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text
        
    def _extract_metadata_from_html(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract metadata from HTML head and other sources."""
        metadata = {}
        
        # Standard meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property')
            content = meta.get('content')
            if name and content:
                metadata[name] = content
                
        # Open Graph tags
        for meta in soup.find_all('meta', property=True):
            prop = meta.get('property')
            content = meta.get('content')
            if prop and content:
                metadata[f'og:{prop}'] = content
                
        # Twitter Card tags
        for meta in soup.find_all('meta', attrs={'name': lambda x: x and x.startswith('twitter:')}):
            name = meta.get('name')
            content = meta.get('content')
            if name and content:
                metadata[name] = content
                
        # Title and description from head
        title = soup.find('title')
        if title:
            metadata['page_title'] = title.get_text().strip()
            
        # Canonical URL
        canonical = soup.find('link', rel='canonical')
        if canonical:
            metadata['canonical_url'] = canonical.get('href')
            
        # Schema.org JSON-LD
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        if json_ld_scripts:
            try:
                import json
                structured_data = []
                for script in json_ld_scripts:
                    data = json.loads(script.string)
                    structured_data.append(data)
                metadata['schema_org'] = structured_data
            except Exception:
                pass
                
        return metadata
        
    def _extract_structured_data(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract structured data like microdata and RDFa."""
        structured_data = []
        
        # Extract microdata
        for element in soup.find_all(attrs={'itemtype': True}):
            item_data = {'type': element.get('itemtype'), 'properties': {}}
            
            for prop_element in element.find_all(attrs={'itemprop': True}):
                prop_name = prop_element.get('itemprop')
                prop_value = prop_element.get('content') or prop_element.get_text()
                item_data['properties'][prop_name] = prop_value
                
            structured_data.append(item_data)
            
        return structured_data
        
    def _extract_tables(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract tables with their content."""
        tables = []
        
        for table in soup.find_all('table'):
            table_data = {
                'headers': [],
                'rows': [],
                'caption': '',
                'summary': table.get('summary', '')
            }
            
            # Extract caption
            caption = table.find('caption')
            if caption:
                table_data['caption'] = caption.get_text().strip()
                
            # Extract headers
            headers_row = table.find('thead')
            if headers_row:
                headers = headers_row.find_all(['th', 'td'])
                table_data['headers'] = [th.get_text().strip() for th in headers]
                
            # Extract rows
            for row in table.find_all('tr'):
                cells = row.find_all(['td', 'th'])
                if cells:
                    row_data = [cell.get_text().strip() for cell in cells]
                    table_data['rows'].append(row_data)
                    
            if table_data['rows']:  # Only add non-empty tables
                tables.append(table_data)
                
        return tables
        
    def _extract_lists(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract ordered and unordered lists."""
        lists = []
        
        for list_elem in soup.find_all(['ul', 'ol']):
            list_data = {
                'type': list_elem.name,
                'items': [],
                'class': list_elem.get('class', []),
                'id': list_elem.get('id', '')
            }
            
            for item in list_elem.find_all('li'):
                list_data['items'].append(item.get_text().strip())
                
            lists.append(list_data)
            
        return lists
        
    def _extract_code_blocks(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract code blocks and preformatted text."""
        code_blocks = []
        
        # Find pre blocks with code
        for pre in soup.find_all('pre'):
            code_elem = pre.find('code')
            if code_elem:
                language = code_elem.get('class', [])
                # Remove 'language-' prefix if present
                language = [lang.replace('language-', '') for lang in language]
                
                code_blocks.append({
                    'language': language[0] if language else '',
                    'code': code_elem.get_text(),
                    'classes': code_elem.get('class', []),
                    'parent_classes': pre.get('class', [])
                })
                
        return code_blocks
        
    def _extract_images_with_context(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, Any]]:
        """Extract images with surrounding context."""
        images = []
        
        for img in soup.find_all('img'):
            src = img.get('src', '')
            if src:
                # Convert to absolute URL
                if base_url and src.startswith('/'):
                    src = urljoin(base_url, src)
                    
                # Find context (previous and next siblings)
                context = self._get_image_context(img)
                
                images.append({
                    'src': src,
                    'alt': img.get('alt', ''),
                    'title': img.get('title', ''),
                    'width': img.get('width'),
                    'height': img.get('height'),
                    'classes': img.get('class', []),
                    'context_before': context['before'],
                    'context_after': context['after'],
                    'caption': self._find_image_caption(img)
                })
                
        return images
        
    def _extract_links_with_context(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, Any]]:
        """Extract links with context and metadata."""
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link.get('href', '')
            if href:
                # Convert to absolute URL
                if base_url and href.startswith('/'):
                    href = urljoin(base_url, href)
                    
                # Find context
                context = self._get_link_context(link)
                
                links.append({
                    'href': href,
                    'text': link.get_text().strip(),
                    'title': link.get('title', ''),
                    'classes': link.get('class', []),
                    'context_before': context['before'],
                    'context_after': context['after'],
                    'external': self._is_external_link(href, base_url)
                })
                
        return links
        
    def _get_image_context(self, img: Tag) -> Dict[str, str]:
        """Get text context around an image."""
        context = {'before': '', 'after': ''}
        
        # Get text before image
        before_siblings = []
        for sibling in img.previous_siblings:
            if hasattr(sibling, 'get_text'):
                before_siblings.append(sibling.get_text())
        context['before'] = ' '.join(before_siblings[-2:])  # Last 2 siblings
        
        # Get text after image  
        after_siblings = []
        for sibling in img.next_siblings:
            if hasattr(sibling, 'get_text'):
                after_siblings.append(sibling.get_text())
        context['after'] = ' '.join(after_siblings[:2])  # First 2 siblings
        
        return context
        
    def _get_link_context(self, link: Tag) -> Dict[str, str]:
        """Get text context around a link."""
        context = {'before': '', 'after': ''}
        
        # Simple implementation - could be enhanced
        parent = link.parent
        if parent:
            parent_text = parent.get_text()
            link_text = link.get_text()
            
            # Find position of link in parent text
            link_pos = parent_text.find(link_text)
            if link_pos >= 0:
                # Get context before and after
                before_start = max(0, link_pos - 50)
                after_end = min(len(parent_text), link_pos + len(link_text) + 50)
                
                context['before'] = parent_text[before_start:link_pos].strip()
                context['after'] = parent_text[link_pos + len(link_text):after_end].strip()
                
        return context
        
    def _find_image_caption(self, img: Tag) -> str:
        """Find caption for an image."""
        # Check for figcaption
        figure = img.find_parent('figure')
        if figure:
            caption = figure.find('figcaption')
            if caption:
                return caption.get_text().strip()
                
        # Check for title attribute or nearby text
        if img.get('title'):
            return img.get('title')
            
        return ''
        
    def _is_external_link(self, href: str, base_url: str) -> bool:
        """Check if link is external."""
        if not base_url:
            return href.startswith('http')
            
        base_domain = urlparse(base_url).netloc
        link_domain = urlparse(href).netloc
        
        return link_domain and link_domain != base_domain
        
    def _generate_summary(self, content: str, max_length: int = 500) -> str:
        """Generate a summary of the content."""
        if len(content) <= max_length:
            return content
            
        # Simple sentence-based summary
        sentences = re.split(r'[.!?]+', content)
        summary_sentences = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                if current_length + len(sentence) <= max_length:
                    summary_sentences.append(sentence)
                    current_length += len(sentence) + 1
                else:
                    break
                    
        return '. '.join(summary_sentences) + ('.' if summary_sentences else '')
        
    def _extract_markdown_structure(self, markdown_text: str) -> Dict[str, Any]:
        """Extract structure from markdown text."""
        lines = markdown_text.split('\n')
        structure = {
            'headings': [],
            'code_blocks': 0,
            'tables': 0,
            'lists': 0,
            'images': 0,
            'links': 0
        }
        
        in_code_block = False
        in_list = False
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('```'):
                in_code_block = not in_code_block
                if in_code_block:
                    structure['code_blocks'] += 1
                continue
                
            if in_code_block:
                continue
                
            # Headings
            if line.startswith('#'):
                level = len(line) - len(line.lstrip('#'))
                title = line.lstrip('#').strip()
                structure['headings'].append({'level': level, 'title': title})
                
            # Lists
            if re.match(r'^[\s]*[-*+]\s', line) or re.match(r'^[\s]*\d+\.\s', line):
                if not in_list:
                    structure['lists'] += 1
                    in_list = True
            else:
                in_list = False
                
            # Tables
            if '|' in line and '---' in line:
                structure['tables'] += 1
                
            # Images
            if '![' in line:
                structure['images'] += 1
                
            # Links
            if '[' in line and '](' in line:
                structure['links'] += 1
                
        return structure
        
    def _extract_sections(self, markdown_text: str) -> List[Dict[str, Any]]:
        """Extract sections from markdown based on headings."""
        lines = markdown_text.split('\n')
        sections = []
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            # Check for heading
            if line.startswith('#'):
                # Save previous section
                if current_section:
                    sections.append(current_section)
                    
                # Start new section
                level = len(line) - len(line.lstrip('#'))
                title = line.lstrip('#').strip()
                current_section = {
                    'level': level,
                    'title': title,
                    'content': [],
                    'line_start': 0  # This would need actual line tracking
                }
            elif current_section:
                current_section['content'].append(line)
                
        # Add last section
        if current_section:
            sections.append(current_section)
            
        return sections
        
    def _analyze_json_structure(self, data: Any, depth: int = 0) -> Dict[str, Any]:
        """Analyze JSON structure recursively."""
        if depth > 5:  # Prevent infinite recursion
            return {'type': 'complex', 'depth': depth}
            
        if isinstance(data, dict):
            return {
                'type': 'object',
                'keys': list(data.keys()),
                'nested_analysis': {
                    key: self._analyze_json_structure(value, depth + 1)
                    for key, value in list(data.items())[:5]  # Limit analysis
                }
            }
        elif isinstance(data, list):
            return {
                'type': 'array',
                'length': len(data),
                'sample_analysis': [
                    self._analyze_json_structure(item, depth + 1)
                    for item in data[:3]  # Analyze first 3 items
                ]
            }
        else:
            return {
                'type': type(data).__name__,
                'value': str(data)[:100]  # Limit value length
            }
            
    def _extract_text_from_json(self, data: Any, text_content: str = '') -> str:
        """Recursively extract text content from JSON."""
        if isinstance(data, dict):
            for value in data.values():
                text_content = self._extract_text_from_json(value, text_content)
        elif isinstance(data, list):
            for item in data:
                text_content = self._extract_text_from_json(item, text_content)
        elif isinstance(data, str):
            text_content += ' ' + data
        elif data is not None:
            text_content += ' ' + str(data)
            
        return text_content.strip()
        
    def _extract_tables_from_json(self, data: Any) -> List[Dict[str, Any]]:
        """Extract tabular data from JSON structures."""
        tables = []
        
        def find_tables(obj, path=''):
            if isinstance(obj, list) and obj and isinstance(obj[0], dict):
                # Potential table
                all_keys = set()
                for item in obj:
                    if isinstance(item, dict):
                        all_keys.update(item.keys())
                        
                if len(all_keys) > 1:  # Has multiple columns
                    tables.append({
                        'data': obj,
                        'columns': list(all_keys),
                        'path': path
                    })
                    
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    find_tables(value, f"{path}.{key}" if path else key)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    find_tables(item, f"{path}[{i}]" if path else f"[{i}]")
                    
        find_tables(data)
        return tables
        
    def _analyze_data_types(self, data: Any) -> Dict[str, Any]:
        """Analyze data types in JSON structure."""
        types = {}
        
        def count_types(obj):
            if isinstance(obj, dict):
                for value in obj.values():
                    count_types(value)
            elif isinstance(obj, list):
                for item in obj:
                    count_types(item)
            else:
                type_name = type(obj).__name__
                types[type_name] = types.get(type_name, 0) + 1
                
        count_types(data)
        return types