import time
import os
import pandas as pd
import joblib
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx
import google.generativeai as genai
from components.openai_component.gemini_chat import GeminiChat
from components.search_info.domain_qdrant import QdrantSetup
from dotenv import load_dotenv
from datetime import datetime
load_dotenv()
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Disable the default Streamlit rerun animation
st.set_page_config(
    page_title="HUST",
    page_icon="👋",
    initial_sidebar_state="expanded"
)

hide_streamlit_style = """
<style>
    #root > div:nth-child(1) > div > div > div > div > section > div {opacity: 100% !important}
    .stDeployButton {display:none;}
    .loader {display: none;}
    .stMarkdown {opacity: 100% !important}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Initialize QdrantSetup once and store in session state
@st.cache_resource
def init_qdrant():
    return QdrantSetup()

if 'setup' not in st.session_state:
    st.session_state.setup = init_qdrant()

# Initialize other components
gemini_chat = GeminiChat()
def generate_chat_id():
    # Lấy thời gian hiện tại
    current_time = datetime.now()
    # Định dạng thời gian thành chuỗi ngày, tháng, năm, giờ, phút, giây
    chat_id = current_time.strftime('%Y%m%d_%H%M%S')  # Ví dụ: "20241014_123045"
    return chat_id

# Sử dụng hàm
new_chat_id = generate_chat_id()
# new_chat_id = f'{time.time()}'
# st.write(new_chat_id)
MODEL_ROLE = 'ai'
AI_AVATAR_ICON = '✨'

# Create system prompt
SYSTEM_PROMPT = """Tôi là Admin của trường Đại học Bách khoa Hà Nội (HUST).
Nhiệm vụ của tôi là cung cấp thông tin chính xác và hữu ích về tuyển sinh, các chương trình đào tạo, cũng như cuộc sống sinh viên tại HUST.
Tôi luôn giữ thái độ chuyên nghiệp, thân thiện và cung cấp câu trả lời chi tiết dựa trên thông tin được cung cấp.
Nếu có thông tin tôi không chắc chắn, tôi sẽ thẳng thắn thừa nhận và gợi ý nơi bạn có thể tìm thấy thông tin chính xác hơn.
Tôi sẽ trả lời bằng tiếng Việt trừ khi bạn yêu cầu sử dụng ngôn ngữ khác.
Hãy cho tôi biết bạn cần hỗ trợ gì nhé! """

# Create a data/ folder if it doesn't exist
os.makedirs('data/', exist_ok=True)

# Load past chats
@st.cache_data
def load_past_chats():
    try:
        return joblib.load('data/past_chats_list')
    except:
        return {}

past_chats = load_past_chats()

# Sidebar for chat history
with st.sidebar:
    st.write('# Lịch sử')
    if st.session_state.get('chat_id') is None:
        st.session_state.chat_id = st.selectbox(
            label='Chọn lịch sử chat',
            options=[new_chat_id] + list(past_chats.keys()),
            format_func=lambda x: past_chats.get(x, 'Đoạn chat mới'),
            placeholder='_',
        )
    else:
        st.session_state.chat_id = st.selectbox(
            label='Chọn lịch sử chat',
            options=[new_chat_id, st.session_state.chat_id] + list(past_chats.keys()),
            index=1,
            format_func=lambda x: past_chats.get(x, 'Đoạn chat mới' if x != st.session_state.chat_id else st.session_state.chat_title),
            placeholder='_',
        )
    st.session_state.chat_title = f'Đoạn chat-{st.session_state.chat_id}'
    
    #Upload file to domain knowledge
    uploaded_file = st.file_uploader("Tải tài liệu lên", type=['xlsx', 'csv' ,'txt'])
    if uploaded_file is not None:
        file_path = f"data_domain/{uploaded_file.name}"
         # Check if the file already exists
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        docs_total = []
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(file_path)
            docs = st.session_state.setup.insert_dataframe(df)
            docs_total += docs
            # st.session_state.setup.insert_documents_if_exist(docs, collection_name='domain_chatbot')
            st.write("File loaded successfully as an Excel file:")
        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(file_path)
            docs = st.session_state.setup.insert_dataframe(df)
            docs_total += docs
            # st.session_state.setup.insert_documents_if_exist(docs, collection_name='domain_chatbot')
            st.write("File loaded successfully as a CSV file:")
        elif uploaded_file.name.endswith('.txt'):
            documents = st.session_state.setup.insert_text_file(uploaded_file, uploaded_file.name)
            docs = st.session_state.setup.chunk_documents(documents)
            docs_total += docs
            st.write("File loaded successfully as a Text file:")
        else:
            st.error("Unsupported file format.")
            
        if st.button("Insert documents"):
            if docs_total:
                st.session_state.setup.insert_documents_if_exist(docs_total, collection_name='domain_chatbot')
                st.success("Insert file successfully.")
            else:
                st.error("No documents to insert.")
        
st.write('# Chatbot tư vấn tuyển sinh :violet[HUST] :sunglasses:')

# Initialize or load chat history
@st.cache_data
def load_chat_history():
    try:
        messages = joblib.load(f'data/{st.session_state.chat_id}-st_messages')
        gemini_history = joblib.load(f'data/{st.session_state.chat_id}-gemini_messages')
        return messages, gemini_history
    except:
        return [], []

if 'messages' not in st.session_state:
    st.session_state.messages, st.session_state.gemini_history = load_chat_history()

# Initialize Gemini model with system prompt
@st.cache_resource
def initialize_chat():
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")
    chat = model.start_chat(history=[])
    combined_prompt = f"{gemini_chat.generate_message_obj()}\n\n{SYSTEM_PROMPT}"
    chat.send_message(combined_prompt, stream=False)
    return chat

if 'chat' not in st.session_state:
    st.session_state.chat = initialize_chat()

# Create a container for chat messages
chat_container = st.container()

# Display chat history
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(name=message['role'], avatar=message.get('avatar')):
            st.markdown(message['content'])

# Function to process user input and generate response
def process_message(prompt, message_placeholder):
    try:
        # Use the cached QdrantSetup instance
        all_results, all_documents = st.session_state.setup.perform_search(prompt)
        reranked_results = st.session_state.setup.reciprocal_rank_fusion(all_results)
        all_documents_sorted = [all_documents[i] for i in reranked_results.keys()]
        print("All documents sorted:", all_documents_sorted)

        prompt_with_context = f"""
        Câu hỏi của người dùng: {prompt}
        
        Tài liệu liên quan: {all_documents_sorted}
        
        Tài liệu liên quan có thể là bộ câu hỏi và trả lời tôi đã chuẩn bị trước đó. Bạn hãy chọn ra những thông tin chính xác và cần thiết nhất để có thể trả lời tốt nhất.
        Nếu tài liệu liên quan có nhiều năm khác nhau, nhớ lưu ý để trả lời ghi chú rõ ràng năm nào.
        Nếu bộ câu hỏi không liên quan, hãy trả lời tôi không biết, không trả lời bịa thông tin hoặc có thể gợi ý nơi bạn có thể tìm thêm thông tin.       
        Hãy trả lời bằng tiếng Việt dựa trên ngữ cảnh được cung cấp và tuân theo hướng dẫn hệ thống.
        """

        full_response = ''
        response = st.session_state.chat.send_message(prompt_with_context, stream=True)
        text_buffer = ''
        
        for chunk in response:
            chunk_text = chunk.text
            
            # Xử lý từng ký tự thay vì split theo space
            for char in chunk_text:
                text_buffer += char
                
                # Chỉ hiển thị khi gặp dấu câu hoặc khoảng trắng
                if char in [' ', '.', '!', '?', '\n', ',', ':', ';']:
                    full_response += text_buffer
                    message_placeholder.markdown(full_response + '▌')
                    text_buffer = ''
                    time.sleep(0.05)
        
        # Thêm phần text còn lại trong buffer (nếu có)
        if text_buffer:
            full_response += text_buffer
            message_placeholder.markdown(full_response)
        # for chunk in response:
        #     for ch in chunk.text.split(' '):
        #         full_response += ch + ' '
        #         message_placeholder.markdown(full_response + '▌')
        #         time.sleep(0.05)
        
        # message_placeholder.markdown(full_response)
        # print("Full response:", full_response)
        # Save response to session state
        st.session_state.messages.append({
            'role': MODEL_ROLE,
            'content': full_response,
            'avatar': AI_AVATAR_ICON,
        })
        
        st.session_state.gemini_history = st.session_state.chat.history
        
        # Save chat history
        joblib.dump(st.session_state.messages, f'data/{st.session_state.chat_id}-st_messages')
        joblib.dump(st.session_state.gemini_history, f'data/{st.session_state.chat_id}-gemini_messages')
        
    except Exception as e:
        st.error(f"Có lỗi xảy ra: {str(e)}")
        st.session_state.chat = initialize_chat()

# Handle user input
if prompt := st.chat_input('Nhập câu hỏi...'):
    # Save chat to history
    if st.session_state.chat_id not in past_chats.keys():
        past_chats[st.session_state.chat_id] = st.session_state.chat_title
        joblib.dump(past_chats, 'data/past_chats_list')

    # Display user message
    with chat_container:
        with st.chat_message('user'):
            st.markdown(prompt)
        st.session_state.messages.append({
            'role': 'user',
            'content': prompt,
        })
        
        # Create a message placeholder for the assistant's response
        with st.chat_message(name=MODEL_ROLE, avatar=AI_AVATAR_ICON):
            message_placeholder = st.empty()
            process_message(prompt, message_placeholder)