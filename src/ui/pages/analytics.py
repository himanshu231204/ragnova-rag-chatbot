"""Analytics Dashboard Page"""

import streamlit as st
import pandas as pd
from datetime import datetime
from src.backend.analytics import QueryLogger


def render_analytics_page():
    """Render the analytics dashboard."""
    st.markdown('<div class="app-shell">', unsafe_allow_html=True)
    
    QueryLogger.initialize()
    
    st.markdown('<h1 style="text-align: center; margin-bottom: 30px;">📊 Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    total_queries = QueryLogger.get_total_queries()
    
    if total_queries == 0:
        st.info(
            "📭 **No analytics data yet.**\n\n"
            "Start asking questions in the **Chat** tab, and your query statistics will appear here."
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Queries", total_queries)
    
    with col2:
        avg_time = QueryLogger.get_avg_response_time()
        st.metric("Avg Response Time", f"{avg_time:.0f}ms")
    
    with col3:
        avg_chunks = QueryLogger.get_avg_chunks_retrieved()
        st.metric("Avg Chunks Retrieved", f"{avg_chunks:.1f}")
    
    with col4:
        recent = QueryLogger.get_recent_queries(limit=1)
        if recent:
            last_query_time = datetime.fromisoformat(recent[0]["timestamp"])
            time_ago = datetime.now() - last_query_time
            if time_ago.total_seconds() < 60:
                last_query_str = "Just now"
            elif time_ago.total_seconds() < 3600:
                last_query_str = f"{int(time_ago.total_seconds() / 60)}m ago"
            else:
                last_query_str = f"{int(time_ago.total_seconds() / 3600)}h ago"
        else:
            last_query_str = "N/A"
        st.metric("Last Query", last_query_str)
    
    st.divider()
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📈 Queries Over Time")
        queries_over_time = QueryLogger.get_queries_over_time(days=30)
        if queries_over_time:
            df_time = pd.DataFrame(queries_over_time)
            df_time["date"] = pd.to_datetime(df_time["date"])
            st.line_chart(df_time.set_index("date"), height=250)
        else:
            st.write("No data available")
    
    with col_right:
        st.subheader("⏱️ Response Time Distribution")
        time_dist = QueryLogger.get_response_time_distribution()
        if time_dist:
            df_dist = pd.DataFrame(time_dist)
            st.bar_chart(df_dist.set_index("bucket"), height=250)
        else:
            st.write("No data available")
    
    st.divider()
    
    col_left2, col_right2 = st.columns(2)
    
    with col_left2:
        st.subheader("🔍 Top-K Usage")
        top_k_usage = QueryLogger.get_top_k_usage()
        if top_k_usage:
            df_k = pd.DataFrame(top_k_usage)
            st.bar_chart(df_k.set_index("top_k"), height=200)
            st.caption(f"Most used: top_k={top_k_usage[0]['top_k']}")
        else:
            st.write("No data available")
    
    with col_right2:
        st.subheader("📝 Response Mode Usage")
        mode_usage = QueryLogger.get_response_mode_usage()
        if mode_usage:
            df_mode = pd.DataFrame(mode_usage)
            st.bar_chart(df_mode.set_index("response_mode"), height=200)
            st.caption(f"Most used: {mode_usage[0]['response_mode']}")
        else:
            st.write("No data available")
    
    st.divider()
    
    st.subheader("💬 Average Chunk Distance Over Time")
    chunk_distances = QueryLogger.get_avg_chunk_distances()
    if chunk_distances:
        df_chunks = pd.DataFrame(chunk_distances)
        df_chunks["timestamp"] = pd.to_datetime(df_chunks["timestamp"])
        st.line_chart(df_chunks.set_index("timestamp"), height=200)
        st.caption("Lower distance = better semantic match")
    else:
        st.write("No data available")
    
    st.divider()
    
    st.subheader("🔥 Popular Query Terms")
    popular_words = QueryLogger.get_popular_words(limit=15)
    if popular_words:
        cols = st.columns(5)
        for i, item in enumerate(popular_words):
            with cols[i % 5]:
                st.metric(item["word"], f"{item['count']}x")
    else:
        st.write("No data available")
    
    st.divider()
    
    st.subheader("📜 Recent Queries")
    recent_queries = QueryLogger.get_recent_queries(limit=10)
    if recent_queries:
        df_recent = pd.DataFrame(recent_queries)
        df_recent["timestamp"] = df_recent["timestamp"].apply(
            lambda x: datetime.fromisoformat(x).strftime("%m/%d %H:%M")
        )
        df_recent = df_recent.rename(columns={
            "timestamp": "Time",
            "query_text": "Query",
            "response_time_ms": "Time (ms)",
            "chunks_retrieved": "Chunks",
            "avg_chunk_distance": "Avg Distance"
        })
        df_recent["Avg Distance"] = df_recent["Avg Distance"].apply(
            lambda x: f"{x:.3f}" if x else "N/A"
        )
        st.dataframe(df_recent, use_container_width=True, hide_index=True)
    else:
        st.write("No recent queries")
    
    st.divider()
    
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🗑️ Clear Analytics Data", type="secondary"):
            QueryLogger.clear_all_data()
            st.success("Analytics data cleared!")
            st.rerun()
    
    with col2:
        st.caption("Analytics data is stored locally in `data/analytics.db`")
    
    st.markdown("</div>", unsafe_allow_html=True)
