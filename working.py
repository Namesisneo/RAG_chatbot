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

    def extract_text_from_pdfs(self) -> List[str]:
        """
        Extract text from all PDF files in the specified directory

        Returns:
            List of extracted text from PDFs
        """
        self.logger.info(f"Extracting text from PDFs in {self.pdf_directory}")
        texts = []

        # Validate PDF directory
        if not os.path.exists(self.pdf_directory):
            raise ValueError(f"Directory not found: {self.pdf_directory}")

        # Iterate through PDF files
        for filename in os.listdir(self.pdf_directory):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(self.pdf_directory, filename)

                try:
                    with open(pdf_path, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        pdf_text = ""

                        # Extract text from all pages
                        for page in pdf_reader.pages:
                            pdf_text += page.extract_text() + "\n"

                        # Add college name from filename
                        college_name = filename.replace('.pdf', '').strip()
                        pdf_text = f"[COLLEGE: {college_name}]\n{pdf_text}"

                        texts.append(pdf_text)
                        self.logger.info(f"Processed: {filename}")

                except Exception as e:
                    self.logger.error(f"Error processing {filename}: {e}")

        self.extracted_texts = texts
        return texts

    def create_vector_store(self, chunk_size: int = 500, chunk_overlap: int = 100, force_retrain: bool = False):
        """
        Create or load a vector store from extracted PDF texts.

        Args:
            chunk_size (int): Size of text chunks for embedding
            chunk_overlap (int): Overlap between text chunks
            force_retrain (bool): If True, retrain and overwrite the vector store.
        """
        vectorstore_path = "vectorstore.faiss"

        # Comprehensive file deletion approach
        def delete_vector_store_files():
            """Helper function to delete vector store files"""
            import glob
            import shutil

            # List of potential file patterns to delete
            file_patterns = [
                vectorstore_path,  # The main file
                f"{vectorstore_path}.*",  # Any additional files with extensions
                f"{vectorstore_path}_*"  # Any additional files with prefixes
            ]

            for pattern in file_patterns:
                try:
                    # Find all matching files
                    matching_files = glob.glob(pattern)

                    for file_path in matching_files:
                        try:
                            # Try different methods to remove the file
                            if os.path.isdir(file_path):
                                shutil.rmtree(file_path)
                            else:
                                os.remove(file_path)
                            self.logger.info(f"Deleted: {file_path}")
                        except PermissionError:
                            self.logger.warning(f"Permission denied when deleting: {file_path}")
                        except Exception as e:
                            self.logger.error(f"Error deleting {file_path}: {e}")
                except Exception as e:
                    self.logger.error(f"Error processing pattern {pattern}: {e}")

        # If force_retrain is True, delete existing vector store files
        if force_retrain:
            delete_vector_store_files()

        # Rest of the method remains the same...
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        except Exception as e:
            self.logger.error(f"Error initializing embeddings: {e}")
            raise

        # Try to load existing vector store
        try:
            if os.path.exists(vectorstore_path) and not force_retrain:
                self.vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=False)
                self.logger.info("Loaded vector store from file.")
                return
        except Exception as e:
            self.logger.warning(f"Could not load existing vector store: {e}")

        # If no existing vector store or force_retrain is True, create new vector store
        if not self.extracted_texts:
            self.extract_text_from_pdfs()

        # Text splitter for semantic chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

        # Split texts into chunks
        all_chunks = []
        for text in self.extracted_texts:
            chunks = text_splitter.split_text(text)
            all_chunks.extend(chunks)

        # Create vector store and save to file
        try:
            self.vectorstore = FAISS.from_texts(all_chunks, embeddings)
            self.vectorstore.save_local(vectorstore_path)
            self.logger.info(f"Vector store created with {len(all_chunks)} chunks and saved to file.")
        except Exception as e:
            self.logger.error(f"Error creating and saving vector store: {e}")
            raise
    def classify_query(self, query: str) -> str:
        """
        Classify query into a specific category
        """
        query_lower = query.lower()

        for category, keywords in self.category_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return category

        return 'general'

    def create_rag_chain(self, category: str):
        """
        Create RAG chain for specific query category
        """
        if self.vectorstore is None:
            self.create_vector_store()

        # Customized prompt templates for different categories
        category_templates = {
            "placements": """
            Focus on placement-related information. 
            Provide details about job opportunities, recruitment process, 
            and placement statistics for the college.

            Context: {context}
            Question: {question}
            Detailed Answer:""",

            "admission": """
            Provide comprehensive admission guidelines. 
            Include eligibility criteria, application process, 
            important dates, and required documents.

            Context: {context}
            Question: {question}
            Detailed Answer:""",

            "general": """
            Provide a comprehensive overview of the college 
            based on the available information.

            Context: {context}
            Question: {question}
            Informative Answer:"""
        }

        # Select appropriate template
        template = category_templates.get(category, category_templates['general'])

        qa_prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        # Create RAG chain
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=self.local_llm,
            retriever=self.vectorstore.as_retriever(
                search_kwargs={
                    "k": 5,  # Retrieve top 5 most relevant chunks
                    "search_type": "similarity"
                }
            ),
            chain_type="stuff",
            chain_type_kwargs={"prompt": qa_prompt}
        )

    def get_response(self, query: str) -> str:
        """
        Get response for the given query
        """
        # Classify query
        category = self.classify_query(query)

        # Create appropriate RAG chain
        self.create_rag_chain(category)

        try:
            response = self.rag_chain.invoke({"query": query})
            return response.get('result', "I couldn't find specific information.")
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return "An error occurred while processing your query."

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
            return response
            print(response)


def main():
    # Specify directory containing college PDFs
    pdf_dir = 'pdfs'
    chatbot = MultiCollegeRAGChatbot(pdf_dir)

    # Prepare vector store
    # chatbot.create_vector_store(chunk_size=500, chunk_overlap=100, force_retrain=True)
    chatbot.create_vector_store()

    # Start interactive chat
    chatbot.interactive_chat()


if __name__ == "__main__":
    main()