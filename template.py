import os

# Create directories
os.makedirs("src", exist_ok=True)
os.makedirs("research", exist_ok=True)

# Files to create
files = {
    "src/__init__.py": "",
    "src/helper.py": "# helper functions for data processing, etc.\n",
    "src/prompt.py": "# define your prompt templates here\n",
    ".env": "# store your secrets here (DO NOT commit to GitHub)\n# Example:\n# OPENAI_API_KEY=your_api_key_here\n",
    "setup.py": """from setuptools import setup, find_packages

setup(
    name="medical_chatbot",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
""",
    "app.py": """import streamlit as st

st.set_page_config(page_title="Medical Chatbot", page_icon="🩺")

st.title("🩺 Medical Chatbot")
st.write("Welcome! Ask me medical-related questions (for educational purposes only).")

# Example input/output UI
user_input = st.text_input("Enter your question:")

if user_input:
    st.write(f"🤖 Bot response for: {user_input}")
""",
    "research/trials.ipynb": "",
    "requirements.txt": """streamlit
openai
python-dotenv
langchain
pandas
numpy
"""
}

# Create files with starter content
for filepath, content in files.items():
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

print("✅ Project structure created successfully for Streamlit app!")
