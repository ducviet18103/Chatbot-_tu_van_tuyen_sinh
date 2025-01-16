import openai
import google.generativeai as genai
from dotenv import load_dotenv
from components.search_info.domain_qdrant import QdrantSetup
from components.search_info.fusion_rag import FusionRAG
import os
from openai import OpenAI
from models.settings import settings

# GOOGLE_API_KEY=os.environ.get('GOOGLE_API_KEY')
class OpenAIChat:
    def __init__(self):
        self.client = OpenAI(
                api_key=settings.openai.openai_key
            )
        self.qdrant = QdrantSetup()
        self.fusion_rag = FusionRAG()

    def generate_message_obj(self, user_input, search_info: str = ""):
        message_obj = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f""""Act as an admissions consultant for Hanoi University of Science and Technology. 
                        Your task is to answer questions from users based on the information in the documents below. The priority for using documents will follow their order in the list.
                        If the document is not relevant or is empty, respond with: "I don't know. Could you provide more documents for me?"
                        
                        Here is User input:
                        {user_input}
                        
                        ## Related documents: {search_info}
                        """
                    }
                ],
            }
        ]
        return message_obj

    def get_response(self, user_input):
        all_results, all_documents=self.qdrant.perform_search(user_input)
        reranked_results = self.qdrant.reciprocal_rank_fusion(all_results)
            
        # final_output = setup.generate_output(reranked_results, generated_queries)
        all_documents_sorted = [all_documents[i] for i in reranked_results.keys()]
        message_obj = self.generate_message_obj(user_input, search_info=all_documents_sorted)
        response = self.client.beta.chat.completions.parse(
          model=settings.openai.openai_chat,
          messages= message_obj,
          response_format= ListTestCase,
          temperature = 0.00000001,
          max_tokens=10000
        )
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=message_obj,
            temperature=0.0,
            max_tokens=1000,
            stream=True,
            stream_options={"include_usage": True}
        )
        return response.choices[0].message['content']
