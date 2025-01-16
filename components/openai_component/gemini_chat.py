import google.generativeai as genai
from dotenv import load_dotenv
import os


class GeminiChat:
    def __init__(self):
        # genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        # self.qdrant = QdrantSetup()
        # self.fusion_rag = FusionRAG()
        
    def generate_message_obj(self):
        prompt = f"""Act as an admissions consultant for Hanoi University of Science and Technology. 
        Your task is to answer questions from users based on the information in the documents below. The priority for using documents will follow their order in the list.
        If the document is not relevant or is empty, respond with: "I don't know. Could you provide more documents for me?"
        """
        return prompt

    def get_response(self, user_input):
        # generated_queries = self.fusion_rag.generate_queries_chatgpt(user_input)
        all_results, all_documents=self.qdrant.perform_search(user_input)
        reranked_results = self.qdrant.reciprocal_rank_fusion(all_results)
            
        # final_output = setup.generate_output(reranked_results, generated_queries)
        all_documents_sorted = [all_documents[i] for i in reranked_results.keys()]
        prompt = self.generate_message_obj(user_input = user_input, search_info = all_documents_sorted)
        
        # Create a chat session
        chat = self.model.start_chat(
            history=[],
            generation_config={
                "temperature": 0.0,
                "max_output_tokens": 1000,
            }
        )
        
        # Get the response
        response = chat.send_message(prompt)
        return response.text

# Example usage:
if __name__ == "__main__":
    load_dotenv()
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    
    chat = GeminiChat(api_key=GOOGLE_API_KEY)
    response = chat.get_response("What is the configuration for X?", "Sample search info")
    print(response)