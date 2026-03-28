"""Backend: Analytics and Query Logging"""

import os
import sqlite3
import time
from datetime import datetime
from typing import Optional
from contextlib import contextmanager

ANALYTICS_DB_PATH = os.path.join("data", "analytics.db")


def get_db_connection():
    """Create a connection to the analytics SQLite database."""
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(ANALYTICS_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_cursor():
    """Context manager for database operations."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        yield cursor
        conn.commit()
    finally:
        conn.close()


class QueryLogger:
    """Handles logging and retrieval of RAG query analytics."""

    _initialized = False

    @classmethod
    def initialize(cls):
        """Initialize the analytics database schema."""
        if cls._initialized:
            return
        
        with get_cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    response_time_ms REAL,
                    top_k INTEGER,
                    embedding_model TEXT,
                    llm_model TEXT,
                    response_mode TEXT,
                    chunks_retrieved INTEGER,
                    avg_chunk_distance REAL,
                    min_chunk_distance REAL,
                    max_chunk_distance REAL,
                    response_length INTEGER
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON queries(timestamp)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_query_text ON queries(query_text)
            """)
        
        cls._initialized = True

    @classmethod
    def log_query(
        cls,
        query_text: str,
        response_time_ms: float,
        top_k: int,
        embedding_model: str,
        llm_model: str,
        response_mode: str,
        chunks: list,
        response_length: int,
    ):
        """Log a query execution to the database."""
        cls.initialize()
        
        distances = [
            c.get("distance") 
            for c in chunks 
            if c and c.get("distance") is not None
        ]
        
        avg_distance = sum(distances) / len(distances) if distances else None
        min_distance = min(distances) if distances else None
        max_distance = max(distances) if distances else None
        
        with get_cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO queries (
                    timestamp, query_text, response_time_ms, top_k,
                    embedding_model, llm_model, response_mode,
                    chunks_retrieved, avg_chunk_distance,
                    min_chunk_distance, max_chunk_distance, response_length
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(),
                    query_text,
                    response_time_ms,
                    top_k,
                    embedding_model,
                    llm_model,
                    response_mode,
                    len(chunks),
                    avg_distance,
                    min_distance,
                    max_distance,
                    response_length,
                )
            )

    @classmethod
    def get_total_queries(cls) -> int:
        """Get total number of logged queries."""
        cls.initialize()
        with get_cursor() as cursor:
            cursor.execute("SELECT COUNT(*) as count FROM queries")
            return cursor.fetchone()["count"]

    @classmethod
    def get_queries_over_time(cls, days: int = 30) -> list:
        """Get query counts grouped by day for the last N days."""
        cls.initialize()
        with get_cursor() as cursor:
            cursor.execute(
                """
                SELECT DATE(timestamp) as date, COUNT(*) as count
                FROM queries
                WHERE timestamp >= datetime('now', ? || ' days')
                GROUP BY DATE(timestamp)
                ORDER BY date
                """,
                (-days,),
            )
            return [dict(row) for row in cursor.fetchall()]

    @classmethod
    def get_avg_response_time(cls) -> float:
        """Get average response time in milliseconds."""
        cls.initialize()
        with get_cursor() as cursor:
            cursor.execute(
                "SELECT AVG(response_time_ms) as avg_time FROM queries"
            )
            result = cursor.fetchone()["avg_time"]
            return result if result else 0.0

    @classmethod
    def get_response_time_distribution(cls) -> list:
        """Get response time distribution buckets."""
        cls.initialize()
        with get_cursor() as cursor:
            cursor.execute(
                """
                SELECT 
                    CASE
                        WHEN response_time_ms < 1000 THEN '< 1s'
                        WHEN response_time_ms < 3000 THEN '1-3s'
                        WHEN response_time_ms < 5000 THEN '3-5s'
                        WHEN response_time_ms < 10000 THEN '5-10s'
                        ELSE '> 10s'
                    END as bucket,
                    COUNT(*) as count
                FROM queries
                GROUP BY bucket
                ORDER BY 
                    CASE bucket
                        WHEN '< 1s' THEN 1
                        WHEN '1-3s' THEN 2
                        WHEN '3-5s' THEN 3
                        WHEN '5-10s' THEN 4
                        ELSE 5
                    END
                """
            )
            return [dict(row) for row in cursor.fetchall()]

    @classmethod
    def get_top_k_usage(cls) -> list:
        """Get distribution of top_k values used."""
        cls.initialize()
        with get_cursor() as cursor:
            cursor.execute(
                """
                SELECT top_k, COUNT(*) as count
                FROM queries
                GROUP BY top_k
                ORDER BY count DESC
                """
            )
            return [dict(row) for row in cursor.fetchall()]

    @classmethod
    def get_response_mode_usage(cls) -> list:
        """Get distribution of response modes."""
        cls.initialize()
        with get_cursor() as cursor:
            cursor.execute(
                """
                SELECT response_mode, COUNT(*) as count
                FROM queries
                GROUP BY response_mode
                ORDER BY count DESC
                """
            )
            return [dict(row) for row in cursor.fetchall()]

    @classmethod
    def get_avg_chunk_distances(cls) -> list:
        """Get average chunk distances over time (last 20 queries)."""
        cls.initialize()
        with get_cursor() as cursor:
            cursor.execute(
                """
                SELECT timestamp, avg_chunk_distance
                FROM queries
                WHERE avg_chunk_distance IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT 20
                """
            )
            return [dict(row) for row in reversed(cursor.fetchall())]

    @classmethod
    def get_avg_chunks_retrieved(cls) -> float:
        """Get average number of chunks retrieved per query."""
        cls.initialize()
        with get_cursor() as cursor:
            cursor.execute(
                "SELECT AVG(chunks_retrieved) as avg_chunks FROM queries"
            )
            result = cursor.fetchone()["avg_chunks"]
            return result if result else 0.0

    @classmethod
    def get_popular_words(cls, limit: int = 20) -> list:
        """Get most frequent words from queries (excluding stopwords)."""
        cls.initialize()
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "need", "dare",
            "what", "which", "who", "whom", "this", "that", "these", "those",
            "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
            "us", "them", "my", "your", "his", "its", "our", "their",
            "and", "or", "but", "if", "then", "else", "when", "where", "how",
            "why", "all", "each", "every", "both", "few", "more", "most",
            "other", "some", "such", "no", "not", "only", "same", "so",
            "than", "too", "very", "just", "also", "now", "here", "there",
            "about", "into", "through", "during", "before", "after", "above",
            "below", "to", "from", "up", "down", "in", "out", "on", "off",
            "over", "under", "again", "further", "once", "explain", "describe"
        }
        
        word_counts = {}
        with get_cursor() as cursor:
            cursor.execute("SELECT query_text FROM queries")
            for row in cursor.fetchall():
                words = row["query_text"].lower().split()
                for word in words:
                    word = ''.join(c for c in word if c.isalnum())
                    if word and len(word) > 2 and word not in stopwords:
                        word_counts[word] = word_counts.get(word, 0) + 1
        
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [{"word": w, "count": c} for w, c in sorted_words[:limit]]

    @classmethod
    def get_recent_queries(cls, limit: int = 10) -> list:
        """Get the most recent queries."""
        cls.initialize()
        with get_cursor() as cursor:
            cursor.execute(
                """
                SELECT timestamp, query_text, response_time_ms, 
                       chunks_retrieved, avg_chunk_distance
                FROM queries
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]

    @classmethod
    def clear_all_data(cls):
        """Clear all analytics data (for testing/reset)."""
        cls.initialize()
        with get_cursor() as cursor:
            cursor.execute("DELETE FROM queries")
