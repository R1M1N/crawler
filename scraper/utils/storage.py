"""
Storage management utilities.

Provides storage management for scraped content, results, and media files.
"""

import json
import pickle
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
import hashlib
from dataclasses import asdict

from ..types import ScrapingResult, ScrapedPage


class StorageManager:
    """Manages storage of scraped data in various formats."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize storage manager.
        
        Args:
            storage_path: Base directory for storing data
        """
        self.storage_path = storage_path or Path("scraped_data")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.raw_path = self.storage_path / "raw"
        self.processed_path = self.storage_path / "processed"
        self.media_path = self.storage_path / "media"
        self.metadata_path = self.storage_path / "metadata"
        
        for path in [self.raw_path, self.processed_path, self.media_path, self.metadata_path]:
            path.mkdir(exist_ok=True)
            
        # Initialize database
        self.db_path = self.storage_path / "scraper.db"
        self._init_database()
        
        self.logger = logging.getLogger(__name__)
        
    def _init_database(self):
        """Initialize SQLite database for metadata storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scraping_results (
                id TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                title TEXT,
                content TEXT,
                html TEXT,
                markdown TEXT,
                metadata TEXT,
                timestamp TEXT,
                processing_time REAL,
                success BOOLEAN,
                error_message TEXT,
                file_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS crawl_jobs (
                id TEXT PRIMARY KEY,
                start_url TEXT NOT NULL,
                domain TEXT,
                status TEXT,
                urls_discovered INTEGER,
                urls_crawled INTEGER,
                urls_successful INTEGER,
                urls_failed INTEGER,
                started_at TEXT,
                completed_at TEXT,
                config TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS media_files (
                id TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                local_path TEXT,
                file_type TEXT,
                file_size INTEGER,
                content_type TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vector_chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT,
                chunk_index INTEGER,
                text TEXT,
                embedding TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def store_scraping_result(self, 
                            result: ScrapingResult,
                            options: Optional[Dict[str, Any]] = None) -> str:
        """
        Store scraping result to disk and database.
        
        Args:
            result: ScrapingResult object
            options: Storage options
            
        Returns:
            Storage ID for the result
        """
        options = options or {}
        storage_id = self._generate_storage_id(result.url)
        
        try:
            # Store in database
            self._store_in_database(storage_id, result)
            
            # Store raw data if requested
            if options.get('store_raw', True):
                self._store_raw_data(storage_id, result, options)
                
            # Store processed data
            if options.get('store_processed', True):
                self._store_processed_data(storage_id, result, options)
                
            self.logger.info(f"Stored scraping result: {storage_id}")
            return storage_id
            
        except Exception as e:
            self.logger.error(f"Failed to store scraping result: {e}")
            raise
            
    def _generate_storage_id(self, url: str) -> str:
        """Generate unique storage ID from URL."""
        url_hash = hashlib.md5(url.encode()).hexdigest()[:16]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_{url_hash}"
        
    def _store_in_database(self, storage_id: str, result: ScrapingResult):
        """Store result metadata in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO scraping_results 
            (id, url, title, content, html, markdown, metadata, 
             timestamp, processing_time, success, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            storage_id,
            result.url,
            result.title,
            result.content,
            result.html,
            result.markdown,
            json.dumps(result.metadata) if result.metadata else None,
            result.timestamp.isoformat(),
            result.processing_time,
            result.success,
            result.error_message
        ))
        
        conn.commit()
        conn.close()
        
    def _store_raw_data(self, storage_id: str, result: ScrapingResult, options: Dict[str, Any]):
        """Store raw HTML and other raw data."""
        raw_dir = self.raw_path / storage_id
        raw_dir.mkdir(exist_ok=True)
        
        # Store HTML
        if result.html and options.get('store_html', True):
            html_path = raw_dir / "page.html"
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(result.html)
                
        # Store screenshot
        if result.screenshot and options.get('store_screenshot', True):
            screenshot_path = raw_dir / "screenshot.png"
            with open(screenshot_path, 'wb') as f:
                f.write(result.screenshot)
                
        # Store JSON with all data
        if options.get('store_json', True):
            json_path = raw_dir / "data.json"
            result_dict = {
                'url': result.url,
                'title': result.title,
                'content': result.content,
                'html': result.html,
                'markdown': result.markdown,
                'metadata': result.metadata,
                'links': result.links,
                'media': result.media,
                'timestamp': result.timestamp.isoformat(),
                'processing_time': result.processing_time,
                'success': result.success,
                'error_message': result.error_message,
                'headers': result.headers,
                'status_code': result.status_code
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)
                
    def _store_processed_data(self, storage_id: str, result: ScrapingResult, options: Dict[str, Any]):
        """Store processed data in various formats."""
        processed_dir = self.processed_path / storage_id
        processed_dir.mkdir(exist_ok=True)
        
        # Store clean markdown
        if result.markdown and options.get('store_markdown', True):
            md_path = processed_dir / "content.md"
            with open(md_path, 'w', encoding='utf-8') as f:
                # Add metadata header
                f.write(f"# {result.title}\\n\\n")
                f.write(f"**Source:** {result.url}\\n")
                f.write(f"**Scraped:** {result.timestamp.isoformat()}\\n")
                if result.metadata:
                    f.write(f"**Metadata:** {json.dumps(result.metadata, indent=2)}\\n")
                f.write("\\n---\\n\\n")
                f.write(result.markdown)
                
        # Store clean text
        if result.content and options.get('store_text', True):
            text_path = processed_dir / "content.txt"
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(f"{result.title}\\n")
                f.write("=" * len(result.title) + "\\n\\n")
                f.write(f"Source: {result.url}\\n")
                f.write(f"Scraped: {result.timestamp.isoformat()}\\n\\n")
                f.write(result.content)
                
        # Store structured data
        if options.get('store_structured', True):
            structured_path = processed_dir / "structured.json"
            structured_data = {
                'title': result.title,
                'url': result.url,
                'content': result.content,
                'metadata': result.metadata,
                'links': result.links,
                'media': result.media,
                'timestamp': result.timestamp.isoformat(),
                'success': result.success
            }
            
            with open(structured_path, 'w', encoding='utf-8') as f:
                json.dump(structured_data, f, indent=2, ensure_ascii=False)
                
    def get_scraping_result(self, storage_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve scraping result from storage."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT url, title, content, html, markdown, metadata,
                       timestamp, processing_time, success, error_message
                FROM scraping_results WHERE id = ?
            ''', (storage_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return None
                
            return {
                'storage_id': storage_id,
                'url': row[0],
                'title': row[1],
                'content': row[2],
                'html': row[3],
                'markdown': row[4],
                'metadata': json.loads(row[5]) if row[5] else None,
                'timestamp': row[6],
                'processing_time': row[7],
                'success': bool(row[8]),
                'error_message': row[9]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve scraping result {storage_id}: {e}")
            return None
            
    def list_scraping_results(self, 
                            limit: int = 100,
                            offset: int = 0,
                            success_only: bool = False) -> List[Dict[str, Any]]:
        """List scraping results from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = '''
                SELECT id, url, title, timestamp, processing_time, success
                FROM scraping_results
            '''
            params = []
            
            if success_only:
                query += ' WHERE success = 1'
                
            query += ' ORDER BY created_at DESC LIMIT ? OFFSET ?'
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            return [
                {
                    'storage_id': row[0],
                    'url': row[1],
                    'title': row[2],
                    'timestamp': row[3],
                    'processing_time': row[4],
                    'success': bool(row[5])
                }
                for row in rows
            ]
            
        except Exception as e:
            self.logger.error(f"Failed to list scraping results: {e}")
            return []
            
    def delete_scraping_result(self, storage_id: str) -> bool:
        """Delete scraping result from storage."""
        try:
            # Delete from database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM scraping_results WHERE id = ?', (storage_id,))
            conn.commit()
            conn.close()
            
            # Delete files
            raw_dir = self.raw_path / storage_id
            processed_dir = self.processed_path / storage_id
            
            for directory in [raw_dir, processed_dir]:
                if directory.exists():
                    import shutil
                    shutil.rmtree(directory)
                    
            self.logger.info(f"Deleted scraping result: {storage_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete scraping result {storage_id}: {e}")
            return False
            
    def store_crawl_job(self, job_data: Dict[str, Any]) -> str:
        """Store crawl job information."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO crawl_jobs 
                (id, start_url, domain, status, urls_discovered, urls_crawled,
                 urls_successful, urls_failed, started_at, completed_at, config)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                job_data['id'],
                job_data['start_url'],
                job_data['domain'],
                job_data['status'],
                job_data.get('urls_discovered', 0),
                job_data.get('urls_crawled', 0),
                job_data.get('urls_successful', 0),
                job_data.get('urls_failed', 0),
                job_data.get('started_at'),
                job_data.get('completed_at'),
                json.dumps(job_data.get('config', {}))
            ))
            
            conn.commit()
            conn.close()
            
            return job_data['id']
            
        except Exception as e:
            self.logger.error(f"Failed to store crawl job: {e}")
            raise
            
    def get_crawl_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve crawl job information."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT start_url, domain, status, urls_discovered, urls_crawled,
                       urls_successful, urls_failed, started_at, completed_at, config
                FROM crawl_jobs WHERE id = ?
            ''', (job_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return None
                
            return {
                'id': job_id,
                'start_url': row[0],
                'domain': row[1],
                'status': row[2],
                'urls_discovered': row[3],
                'urls_crawled': row[4],
                'urls_successful': row[5],
                'urls_failed': row[6],
                'started_at': row[7],
                'completed_at': row[8],
                'config': json.loads(row[9]) if row[9] else {}
            }
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve crawl job {job_id}: {e}")
            return None
            
    def list_crawl_jobs(self, 
                       limit: int = 50,
                       offset: int = 0,
                       status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List crawl jobs from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = '''
                SELECT id, start_url, domain, status, urls_discovered, urls_crawled,
                       urls_successful, urls_failed, started_at, completed_at
                FROM crawl_jobs
            '''
            params = []
            
            if status:
                query += ' WHERE status = ?'
                params.append(status)
                
            query += ' ORDER BY created_at DESC LIMIT ? OFFSET ?'
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            return [
                {
                    'id': row[0],
                    'start_url': row[1],
                    'domain': row[2],
                    'status': row[3],
                    'urls_discovered': row[4],
                    'urls_crawled': row[5],
                    'urls_successful': row[6],
                    'urls_failed': row[7],
                    'started_at': row[8],
                    'completed_at': row[9]
                }
                for row in rows
            ]
            
        except Exception as e:
            self.logger.error(f"Failed to list crawl jobs: {e}")
            return []
            
    def store_media_file(self, 
                        url: str,
                        local_path: Path,
                        file_type: str,
                        metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store media file information."""
        try:
            media_id = hashlib.md5(url.encode()).hexdigest()[:16]
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO media_files 
                (id, url, local_path, file_type, file_size, content_type, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                media_id,
                url,
                str(local_path),
                file_type,
                local_path.stat().st_size,
                metadata.get('content_type') if metadata else None,
                json.dumps(metadata) if metadata else None
            ))
            
            conn.commit()
            conn.close()
            
            return media_id
            
        except Exception as e:
            self.logger.error(f"Failed to store media file: {e}")
            raise
            
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Count records
            cursor.execute('SELECT COUNT(*) FROM scraping_results')
            scraping_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM crawl_jobs')
            crawl_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM media_files')
            media_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM vector_chunks')
            chunk_count = cursor.fetchone()[0]
            
            # Calculate storage size
            total_size = 0
            for directory in [self.raw_path, self.processed_path, self.media_path]:
                for file_path in directory.rglob('*'):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
                        
            conn.close()
            
            return {
                'scraping_results': scraping_count,
                'crawl_jobs': crawl_count,
                'media_files': media_count,
                'vector_chunks': chunk_count,
                'storage_size_mb': total_size / (1024 * 1024),
                'storage_path': str(self.storage_path)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get storage stats: {e}")
            return {}
            
    def cleanup_old_files(self, days_old: int = 30) -> int:
        """Clean up old files and database records."""
        try:
            from datetime import timedelta
            
            cutoff_date = datetime.now() - timedelta(days=days_old)
            cleaned_count = 0
            
            # Clean up old database records
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete old scraping results
            cursor.execute('''
                DELETE FROM scraping_results 
                WHERE created_at < ?
            ''', (cutoff_date.isoformat(),))
            
            cleaned_count += cursor.rowcount
            
            # Delete old crawl jobs
            cursor.execute('''
                DELETE FROM crawl_jobs 
                WHERE created_at < ?
            ''', (cutoff_date.isoformat(),))
            
            cleaned_count += cursor.rowcount
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Cleaned up {cleaned_count} old records")
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old files: {e}")
            return 0
            
    def export_data(self, export_path: Path, format: str = 'json') -> bool:
        """Export all data to a file."""
        try:
            if format == 'json':
                return self._export_to_json(export_path)
            elif format == 'pickle':
                return self._export_to_pickle(export_path)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            self.logger.error(f"Failed to export data: {e}")
            return False
            
    def _export_to_json(self, export_path: Path) -> bool:
        """Export data to JSON format."""
        try:
            export_data = {
                'scraping_results': [],
                'crawl_jobs': [],
                'media_files': [],
                'vector_chunks': [],
                'export_timestamp': datetime.now().isoformat()
            }
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Export scraping results
            cursor.execute('SELECT * FROM scraping_results')
            for row in cursor.fetchall():
                export_data['scraping_results'].append({
                    'id': row[0],
                    'url': row[1],
                    'title': row[2],
                    'content': row[3],
                    'html': row[4],
                    'markdown': row[5],
                    'metadata': json.loads(row[6]) if row[6] else None,
                    'timestamp': row[7],
                    'processing_time': row[8],
                    'success': bool(row[9]),
                    'error_message': row[10]
                })
                
            # Export crawl jobs
            cursor.execute('SELECT * FROM crawl_jobs')
            for row in cursor.fetchall():
                export_data['crawl_jobs'].append({
                    'id': row[0],
                    'start_url': row[1],
                    'domain': row[2],
                    'status': row[3],
                    'urls_discovered': row[4],
                    'urls_crawled': row[5],
                    'urls_successful': row[6],
                    'urls_failed': row[7],
                    'started_at': row[8],
                    'completed_at': row[9],
                    'config': json.loads(row[10]) if row[10] else None
                })
                
            conn.close()
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"Exported data to {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export to JSON: {e}")
            return False
            
    def _export_to_pickle(self, export_path: Path) -> bool:
        """Export data to Pickle format."""
        try:
            export_data = {}
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Export all tables
            for table in ['scraping_results', 'crawl_jobs', 'media_files', 'vector_chunks']:
                cursor.execute(f'SELECT * FROM {table}')
                export_data[table] = cursor.fetchall()
                
            conn.close()
            
            with open(export_path, 'wb') as f:
                pickle.dump(export_data, f)
                
            self.logger.info(f"Exported data to {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export to Pickle: {e}")
            return False