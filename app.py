import streamlit as st

st.set_page_config(page_title="Medical Chatbot", page_icon="🩺")

st.title("🩺 Medical Chatbot")
st.write("Welcome! Ask me medical-related questions (for educational purposes only).")

# Example input/output UI
user_input = st.text_input("Enter your question:")

if user_input:
    st.write(f"🤖 Bot response for: {user_input}")
