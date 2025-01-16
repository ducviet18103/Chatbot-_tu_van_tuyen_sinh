import os
import openai
import random
from openai import OpenAI
from models.settings import settings
# from models.config.logger

# Initialize OpenAI API
class FusionRAG:
    def __init__(self):
        self.messages = None
        self.client = OpenAI(
                api_key=settings.openai.openai_key
            )
        # self.logger = Logger(self.__class__.__name__)

    # Function to generate queries using OpenAI's ChatGPT
    def generate_queries_chatgpt(self, original_query):

        message_obj = [
                {"role": "system", "content": "You are a helpful assistant that generates multiple search queries based on a single input query."},
                {"role": "user", "content": f"Generate multiple search queries related to: {original_query}"},
                {"role": "user", "content": "OUTPUT (4 queries):"}
            ]

        response = self.client.chat.completions.create(
            model=settings.openai.openai_chat,
            messages= message_obj,
            temperature = 0.00000001,
            max_tokens=3000
        )
        print(response.choices[0].message.content.strip().split("\n"))
        return response.choices[0].message.content.strip().split("\n")

    # Mock function to simulate vector search, returning random scores
    def vector_search(self, query, all_documents):
        available_docs = list(all_documents.keys())
        random.shuffle(available_docs)
        selected_docs = available_docs[:random.randint(2, 5)]
        scores = {doc: round(random.uniform(0.7, 0.9), 2) for doc in selected_docs}
        return {doc: score for doc, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)}
    
    # def retriever(query: str, ) -> List[Document]:
    #     docs, scores = zip(*vector_store.similarity_search_with_score(query, k=3))
    #     for doc, score in zip(docs, scores):
    #         doc.metadata["score"] = score

    #     return docs

    # Reciprocal Rank Fusion algorithm
    def reciprocal_rank_fusion(self, search_results_dict, k=60):
        fused_scores = {}
        print("Initial individual search result ranks:")
        for query, doc_scores in search_results_dict.items():
            print(f"For query '{query}': {doc_scores}")
            
        for query, doc_scores in search_results_dict.items():
            for rank, (doc, score) in enumerate(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)):
                if doc not in fused_scores:
                    fused_scores[doc] = 0
                previous_score = fused_scores[doc]
                fused_scores[doc] += 1 / (rank + k)
                print(f"Updating score for {doc} from {previous_score} to {fused_scores[doc]} based on rank {rank} in query '{query}'")

        reranked_results = {doc: score for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)}
        print("Final reranked results:", reranked_results)
        return reranked_results

    # Dummy function to simulate generative output
    def generate_output(self, reranked_results, queries):
        return f"Final output based on {queries} and reranked documents: {list(reranked_results.keys())}"



