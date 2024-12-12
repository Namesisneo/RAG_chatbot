import os
import dotenv
import PyPDF2
import logging
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

model_name = "google/flan-t5-small"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
# Or specify a specific cache directory
# os.environ['TRANSFORMERS_CACHE'] = 'C:\\path\\to\\your\\cache\\directory'

local_llm = HuggingFacePipeline(
    pipeline=pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer
    )
)


class PDFRAGChatbot:
    def __init__(self, pdf_directory: str):
        """
        Initialize the PDF-based RAG Chatbot.

        Args:
            pdf_directory (str): Directory containing PDF files.
        """
        # Configure logging
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)

        # Load environment variables
        dotenv.load_dotenv()

        # Set paths and initialize attributes
        self.pdf_directory = pdf_directory
        self.vectorstore = None
        self.rag_chain = None
        self.extracted_texts = []

    def extract_text_from_pdfs(self) -> List[str]:
        """
        Extract text from all PDF files in the specified directory.

        Returns:
            List[str]: Extracted text from PDFs
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

                        texts.append(pdf_text)
                        self.logger.info(f"Processed: {filename}")

                except Exception as e:
                    self.logger.error(f"Error processing {filename}: {e}")

        self.extracted_texts = texts
        return texts

    def create_vector_store(self, chunk_size: int = 500, chunk_overlap: int = 100, force_retrain: bool = False):
        vectorstore_path = "vectorstore.faiss"

        # Initialize embeddings
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        except Exception as e:
            self.logger.error(f"Error initializing embeddings: {e}")
            raise

        # If retraining is forced, delete existing vector store
        if force_retrain and os.path.exists(vectorstore_path):
            try:
                os.remove(vectorstore_path)
                self.logger.info("Existing vector store deleted. Retraining...")
            except PermissionError:
                self.logger.error("Could not delete existing vector store. Check file permissions.")
                raise

        # Load existing vector store if not forcing retraining
        try:
            if os.path.exists(vectorstore_path) and not force_retrain:
                self.vectorstore = FAISS.load_local(vectorstore_path, embeddings)
                self.logger.info("Loaded vector store from file.")
                return
        except Exception as e:
            self.logger.error(f"Error loading vector store: {e}")
            # If loading fails, we'll create a new vector store

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

    def create_rag_chain(self):
        """
        Create the RAG retrieval chain.
        """
        if self.vectorstore is None:
            self.create_vector_store()

        # Prompt template
        review_template_str = """You are a helpful assistant analyzing PDF documents. 
        Provide detailed and accurate information based on the following context. 
        If the information is not in the context, say you don't have specific details.

        Context:
        {context}

        Question: {question}
        Helpful Answer:"""

        qa_prompt = PromptTemplate(
            template=review_template_str,
            input_variables=["context", "question"]
        )

        # Create RAG chain using local LLM
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=local_llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type="stuff",
            chain_type_kwargs={"prompt": qa_prompt}
        )

    def get_response(self, question: str) -> Optional[str]:
        """
        Get a response from the RAG chain.

        Args:
            question (str): User's query

        Returns:
            Optional[str]: Bot's response
        """
        if self.rag_chain is None:
            self.create_rag_chain()

        try:
            response = self.rag_chain.invoke({"query": question})
            return response.get('result', "I couldn't find a relevant answer.")
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return f"An error occurred: {str(e)}"

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
            print(response)


def main():
    # Example usage
    pdf_dir = 'pdfs'  # Directory containing your PDF files
    chatbot = PDFRAGChatbot(pdf_dir)

    # Optional: Prepare vector store beforehand
    chatbot.create_vector_store()

    # Start interactive chat
    chatbot.interactive_chat()


if __name__ == "__main__":
    main()
