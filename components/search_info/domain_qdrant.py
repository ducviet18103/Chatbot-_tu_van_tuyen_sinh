# from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance
from langchain_qdrant import QdrantVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
import pandas as pd
# from qdrant_client.http.models import VectorParams
from langchain_core.documents import Document
from langchain_qdrant import FastEmbedSparse, RetrievalMode
from models.settings import settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from typing import List, Optional, Tuple
from langchain_core.runnables import chain
from components.search_info.fusion_rag import FusionRAG
import random
from langchain_community.document_loaders import DataFrameLoader


class QdrantSetup():
    def __init__(self, path=settings.qdrant_config.qdrant_path):
        self.path = path
        self.fusion_rag = FusionRAG()
        self.raw_data_path = './raw_data'
        self.vector_size = int(settings.qdrant_config.qdrant_vector_size)
        self.distance_metric = Distance.COSINE
        self.embedding = HuggingFaceEmbeddings(model_name=settings.qdrant_config.qdrant_embedding_model)
        self.sparse_embeddings = FastEmbedSparse(model_name=settings.qdrant_config.qdrant_sparse_vector_name)
        # self.client = QdrantClient(path=self.path)

    def read_text_files(self) -> List[Document]:
        documents = []
        for filename in os.listdir(self.raw_data_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(self.raw_data_path, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    # Read content and remove empty lines
                    lines = file.readlines()
                    cleaned_content = "\n".join(line.strip() for line in lines if line.strip())
                    documents.append(Document(
                        page_content=cleaned_content,
                        metadata={'source': filename}
                    ))
        return documents
    
    def insert_text_file(self, file_obj, file_name) -> List[Document]:
        # Đọc nội dung từ file
        lines = file_obj.readlines()
        
        # Làm sạch nội dung, loại bỏ dòng trống
        cleaned_content = "\n".join(line.decode('utf-8').strip() for line in lines if line.strip())
        
        # Tạo document với metadata là tên file
        document = Document(
            page_content=cleaned_content,
            metadata={'source': file_name}
        )
        return [document]
    
    def insert_dataframe(self, df) -> List[Document]:
        df['combined_content'] = df.apply(lambda row: ' '.join(row.astype(str)), axis=1)
        loader = DataFrameLoader(
            df,
            page_content_column="combined_content"  # Use the combined col # Include original columns as metadata
        )

        # Load the documents
        docs = loader.load()
        return docs
    def dataframe_documents(self, df) -> List[Document]:
        """
        Insert documents from DataFrame into Qdrant collection
        
        :param df: pandas DataFrame with the documents
        :param collection_name: Name of the collection
        """
        # df = pd.read_excel('D:/ait_chatbot/Diem_Chuan_Tuyen_Sinh (2).xlsx')
        
        # Create combined content
        df["combined_content"] = (
            "Điểm chuẩn trúng tuyển đại học bách khoa hà nội năm " + df["Năm"].astype(str) + 
            ' ngành ' + df["Mã xét tuyển"].astype(str) + " " +
            df["Tên ngành/chương trình đào tạo"].astype(str) + " là " + 
            df["Điểm chuẩn"].astype(str) + " với hình thức " + 
            df["Hình thức"].astype(str)
        )
        
        # Convert DataFrame rows to Document objects
        loader = DataFrameLoader(
            df,
            page_content_column="combined_content"  # Use the combined col # Include original columns as metadata
        )

        # Load the documents
        docs = loader.load()
        return docs
    
    def chunk_documents(self, 
                        documents: List[Document], 
                        chunk_size: int = 1000, 
                        chunk_overlap: int = 300) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return text_splitter.split_documents(documents)

    def insert_documents(self, 
                         documents: List[Document], 
                         collection_name: str):
       
        vector_store = QdrantVectorStore.from_documents(
            documents=documents,
            embedding=self.embedding,
            sparse_embedding=self.sparse_embeddings,
            sparse_vector_name="sparse-vector",
            path=self.path,
            collection_name=collection_name,
            retrieval_mode=RetrievalMode.HYBRID,
        )
        
        
    def insert_documents_if_exist(self, 
                         documents, 
                         collection_name):
        """
        Insert chunked documents into Qdrant collection
        
        :param documents: List of chunked documents
        :param collection_name: Name of the collection
        """
        vector_store = QdrantVectorStore.from_existing_collection(
            collection_name=collection_name,
            path=self.path,
            retrieval_mode=RetrievalMode.HYBRID,
            embedding=self.embedding, 
            sparse_embedding=self.sparse_embeddings,
            sparse_vector_name="sparse-vector"
        )
        vector_store.add_documents(documents=documents)
        print(f"Inserted {len(documents)} documents into collection '{collection_name}'")

    def process_and_index(self, collection_name: str):
        """
        Comprehensive method to process files and index in Qdrant
        
        :param collection_name: Name of the collection to create
        """
        documents = self.read_text_files()
        dataframe = self.dataframe_documents()
        chunked_documents = self.chunk_documents(documents)
        all_documents = chunked_documents + dataframe
        self.insert_documents(all_documents, collection_name)
    def hybrid_search(self, 
                      query: str, 
                      collection_name: str, 
                      k: int = 2, 
                      score_threshold: float = 0.9) -> List[Tuple[Document, float]]:
       
        qdrant = QdrantVectorStore.from_existing_collection(
            collection_name=collection_name,
            path=self.path,
            retrieval_mode=RetrievalMode.HYBRID,
            embedding=self.embedding, 
            sparse_embedding=self.sparse_embeddings,
            sparse_vector_name="sparse-vector"
        )
        
        return qdrant.similarity_search_with_score(
            query=query, 
            score_threshold=score_threshold
        )
     
    def vector_search(self, query, all_documents):
        available_docs = list(all_documents.keys())
        random.shuffle(available_docs)
        selected_docs = available_docs[:random.randint(2, 5)]
        scores = {doc: round(random.uniform(0.7, 0.9), 2) for doc in selected_docs}
        return {doc: score for doc, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)}

    def convert_to_document_dict(self, results):
        all_documents = {}
        for idx, (doc, _) in enumerate(results):
            doc_key = f"doc{idx + 1}"
            all_documents[doc_key] = doc.page_content
        return all_documents

    def perform_search(self, query):
        # generated_queries = self.fusion_rag.generate_queries_chatgpt(query)
        # print("Generated queries:", generated_queries)
        all_results = {}
        results = self.hybrid_search(query, collection_name='domain_chatbot')
        #print results content and score
        for doc, score in results:
            print(f"Document: {doc.page_content[:100]}...")
            print(f"Score: {score}\n")
        all_documents = self.convert_to_document_dict(results)
        # print("All documents:", all_documents)
        # for query in generated_queries:
        search_results = self.vector_search(query, all_documents)
        all_results[query] = search_results
        return all_results, all_documents
    # Reciprocal Rank Fusion algorithm
    def reciprocal_rank_fusion(self, search_results_dict, k=60):
        fused_scores = {}
        print("Initial individual search result ranks:")
        for query, doc_scores in search_results_dict.items():
            for rank, (doc, score) in enumerate(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)):
                if doc not in fused_scores:
                    fused_scores[doc] = 0
                previous_score = fused_scores[doc]
                fused_scores[doc] += 1 / (rank + k)
               
        reranked_results = {doc: score for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)}
        print("Final reranked results:", reranked_results)
        return reranked_results
    
    def generate_output(self, reranked_results, queries):
        #Document tương ứng với key doc1, doc2, doc
        
        return f"Final output based on {queries} and reranked documents: {list(reranked_results.keys())}"

   

    
if __name__ == "__main__":
    setup = QdrantSetup(
        raw_data_path='./raw_data', 
        qdrant_path='./qdrant_storage'
    )
    
    # Process and index documents
    setup.process_and_index('my_document_collection')
    
    # Perform hybrid search
    results = setup.hybrid_search(
        query="Your search query here",
        collection_name='my_document_collection'
    )
    
    for doc, score in results:
        print(f"Document: {doc.page_content[:100]}...")
        print(f"Score: {score}\n")
        
        

        
        
