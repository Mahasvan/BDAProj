import streamlit as st
import requests
import json

st.set_page_config(page_title="SparkGPT", page_icon="⚡")
st.title("Welcome to SparkGPT")

# Sidebar settings
st.sidebar.header("Settings")
host = st.sidebar.text_input("API Host", "http://localhost:8000")

# Main UI
st.subheader("Ask questions about Big Data and related topics!")

query = st.text_input("Enter your query:", placeholder="e.g., climate change")

if st.button("Send"):
    if not query.strip():
        st.warning("Please enter a query first.")
    else:
        st.write("### 💬 Response:")
        response_placeholder = st.empty()
        full_response = ""

        try:
            # Stream response from backend
            url = f"{host}/chat/wiki?query={query.replace(' ', '%20')}"
            with requests.get(url, stream=True) as r:
                for line in r.iter_lines(decode_unicode=True):
                    if line:
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                full_response += data["response"]
                                response_placeholder.markdown(full_response)
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            st.error(f"Error: {e}")
