import os
import logging
import glob
import shutil
import PyPDF2
from typing import List, Dict, Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


class MultiCollegeRAGChatbot:
    def __init__(self, pdf_directory: str):
        """
        Initialize Multi-College RAG Chatbot

        Args:
            pdf_directory (str): Directory containing college PDF documents
        """
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Set paths and initialize attributes
        self.pdf_directory = pdf_directory
        self.vectorstore = None
        self.rag_chain = None
        self.extracted_texts = []

        # Categorization keywords
        self.category_keywords = {
            "placements": ["placement", "job", "career", "recruit", "hire"],
            "admission": ["admission", "entry", "apply", "eligibility", "criteria"],
            "scholarship": ["scholarship", "fund", "financial", "support", "aid"],
            "hostel": ["hostel", "accommodation", "residence", "dorm", "room"],
            "academic": ["course", "program", "study", "curriculum", "academic"]
        }

        # Initialize LLM
        self.initialize_llm()

    def initialize_llm(self):
        """
        Initialize local language model
        """
        model_name = "google/flan-t5-small"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.local_llm = HuggingFacePipeline(
            pipeline=pipeline(
                 "text2text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_length=1024,  # Increase max tokens in output
                    truncation=True
            )
        )

    # [All other methods remain the same as in the original file]

    def interactive_chat(self):
        """
        Run an interactive chat loop with personalized greetings.
        """
        print("🤖 Welcome to the Student Helper Chatbot!")
        print("Type 'exit' to quit")

        # Ask for user's name and personalize greetings
        user_name = input("\n👤 May I know your name?: ").strip()
        if not user_name:
            user_name = "Guest"
        print(f"\n👋 Hello, {user_name}! How can I assist you today?:")

        while True:
            query = input(f"\n{user_name}, your question: ").strip()

            if query.lower() == 'exit':
                print(f"\n👋 Goodbye, {user_name}! Have a great day!")
                break

            response = self.get_response(query)
            print("\n📄 Bot Response:")
            print(response)  # This line is fixed - prints the response instead of returning


def main():
    # Specify directory containing college PDFs
    pdf_dir = 'pdfs'
    
    # Ensure the PDF directory exists
    os.makedirs(pdf_dir, exist_ok=True)
    
    chatbot = MultiCollegeRAGChatbot(pdf_dir)

    # Prepare vector store
    try:
        chatbot.create_vector_store()
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return

    # Start interactive chat
    chatbot.interactive_chat()


if __name__ == "__main__":
    main()