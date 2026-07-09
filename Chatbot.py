import streamlit as st
from openai import OpenAI
import anthropic
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain_openai import ChatOpenAI, OpenAI as LangChainOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from streamlit_feedback import streamlit_feedback

# ==========================================
# ⚙️ CONFIGURATION & GLOBAL SIDEBAR
# ==========================================
st.set_page_config(page_title="AI Super Hub", page_icon="🚀", layout="wide")

# Manajemen API Key Terpusat di Sidebar agar semua halaman bisa mengaksesnya
with st.sidebar:
    st.title("⚙️ Global Settings")
    openai_api_key = st.text_input("OpenAI API Key", type="password", key="global_openai")
    anthropic_api_key = st.text_input("Anthropic API Key", type="password", key="global_anthropic")
    
    st.markdown("---")
    st.markdown("### 🔗 Quick Links")
    st.markdown("- [Get OpenAI API Key](https://platform.openai.com/account/api-keys)")
    st.markdown("- [Get Anthropic API Key](https://console.anthropic.com/)")

# ==========================================
# 📑 DEFINISI HALAMAN-HALAMAN APLIKASI
# ==========================================

def simple_chatbot():
    st.title("💬 Simple Chatbot")
    st.caption("🚀 A clean, basic chatbot powered by OpenAI GPT-4o-mini")
    
    if "messages_simple" not in st.session_state:
        st.session_state.messages_simple = [{"role": "assistant", "content": "How can I help you today?"}]

    for msg in st.session_state.messages_simple:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        if not openai_api_key:
            st.info("Please add your OpenAI API key in the sidebar to continue.")
            st.stop()

        st.session_state.messages_simple.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        client = OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(model="gpt-4o-mini", messages=st.session_state.messages_simple)
        msg = response.choices[0].message.content
        st.session_state.messages_simple.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)


def file_qa():
    st.title("📝 File Q&A with Anthropic")
    st.caption("Upload a document and ask questions about its content.")
    
    uploaded_file = st.file_uploader("Upload an article", type=("txt", "md"))
    question = st.text_input("Ask something about the article", placeholder="Can you give me a short summary?", disabled=not uploaded_file)

    if uploaded_file and question:
        if not anthropic_api_key:
            st.info("Please add your Anthropic API key in the sidebar to continue.")
            st.stop()
            
        article = uploaded_file.read().decode()
        full_prompt = f"{anthropic.HUMAN_PROMPT} Here's an article:\n\n<article>\n{article}\n</article>\n\n{question}{anthropic.AI_PROMPT}"

        client = anthropic.Client(api_key=anthropic_api_key)
        response = client.completions.create(
            prompt=full_prompt,
            stop_sequences=[anthropic.HUMAN_PROMPT],
            model="claude-2.1",
            max_tokens_to_sample=500,
        )
        st.write("### Answer")
        st.write(response.completion)


def chat_with_search():
    st.title("🔎 LangChain - Chat with Search")
    st.caption("An advanced agent that can search the live web using DuckDuckGo.")

    if "messages_search" not in st.session_state:
        st.session_state.messages_search = [{"role": "assistant", "content": "Hi, I can search the web. What's on your mind?"}]

    for msg in st.session_state.messages_search:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input(placeholder="Who won the latest Super Bowl?"):
        st.session_state.messages_search.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        if not openai_api_key:
            st.info("Please add your OpenAI API key in the sidebar to continue.")
            st.stop()

        llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=openai_api_key, streaming=True)
        search = DuckDuckGoSearchRun(name="Search")
        search_agent = initialize_agent([search], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)
        
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            chat_history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages_search])
            response = search_agent.run(chat_history_str, callbacks=[st_cb])
            st.session_state.messages_search.append({"role": "assistant", "content": response})
            st.write(response)


def langchain_quickstart():
    st.title("🦜🔗 LangChain Quickstart")
    st.caption("Simple LLM execution using LangChain's standard invoke.")

    def generate_response(input_text):
        llm = LangChainOpenAI(temperature=0.7, openai_api_key=openai_api_key, model="gpt-3.5-turbo-instruct")
        st.info(llm.invoke(input_text))

    with st.form("quickstart_form"):
        text = st.text_area("Enter text:", "What are 3 key advice for learning how to code?")
        submitted = st.form_submit_button("Submit")
        if submitted:
            if not openai_api_key:
                st.info("Please add your OpenAI API key to continue.")
            else:
                generate_response(text)


def blog_generator():
    st.title("✍️ Blog Outline Generator")
    st.caption("Generate structured blog outlines with LangChain Prompt Templates.")

    def generate_outline(topic):
        llm = LangChainOpenAI(temperature=0.7, openai_api_key=openai_api_key, model="gpt-3.5-turbo-instruct")
        template = "As an experienced technical writer, generate an outline for a blog about {topic}."
        prompt = PromptTemplate(input_variables=["topic"], template=template)
        response = llm.invoke(prompt.format(topic=topic))
        st.info(response)

    with st.form("blog_form"):
        topic_text = st.text_input("Enter blog topic:", "")
        submitted = st.form_submit_button("Generate Outline")
        if submitted:
            if not openai_api_key:
                st.info("Please add your OpenAI API key to continue.")
            else:
                generate_outline(topic_text)


def chat_with_feedback():
    st.title("⭐ Chat with Feedback (Trubrics)")
    st.caption("Talk to the AI and submit user evaluation metrics.")

    if "messages_fb" not in st.session_state:
        st.session_state.messages_fb = [{"role": "assistant", "content": "How can I help you? Leave a thumbs up or down!"}]
    if "last_fb_response" not in st.session_state:
        st.session_state.last_fb_response = None

    for msg in st.session_state.messages_fb:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input(placeholder="Tell me a joke..."):
        st.session_state.messages_fb.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
            
        client = OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(model="gpt-4o-mini", messages=st.session_state.messages_fb)
        st.session_state.last_fb_response = response.choices[0].message.content
        
        with st.chat_message("assistant"):
            st.session_state.messages_fb.append({"role": "assistant", "content": st.session_state.last_fb_response})
            st.write(st.session_state.last_fb_response)

    if st.session_state.last_fb_response:
        feedback = streamlit_feedback(feedback_type="thumbs", optional_text_label="[Optional] Explanation", key=f"fb_{len(st.session_state.messages_fb)}")
        
        trubrics_email = st.secrets.get("TRUBRICS_EMAIL") if "TRUBRICS_EMAIL" in st.secrets else None
        trubrics_password = st.secrets.get("TRUBRICS_PASSWORD") if "TRUBRICS_PASSWORD" in st.secrets else None
        
        if feedback and trubrics_email and trubrics_password:
            import trubrics
            config = trubrics.init(email=trubrics_email, password=trubrics_password)
            collection = trubrics.collect(component_name="default", model="gpt-4o-mini", response=feedback, metadata={"chat": st.session_state.messages_fb})
            trubrics.save(config, collection)
            st.toast("Feedback recorded!", icon="📝")

# ==========================================
# 🚀 STREAMLIT MODERN NAVIGATION SETUP
# ==========================================
pg = st.navigation({
    "Core Chat Applications": [
        st.Page(simple_chatbot, title="Simple Chatbot", icon="💬"),
        st.Page(chat_with_search, title="Chat with Search", icon="🔎"),
        st.Page(chat_with_feedback, title="Chat with Feedback", icon="⭐"),
    ],
    "Document & Content Tools": [
        st.Page(file_qa, title="File Q&A (Anthropic)", icon="📝"),
        st.Page(blog_generator, title="Blog Outline Generator", icon="✍️"),
        st.Page(langchain_quickstart, title="LangChain Quickstart", icon="🦜"),
    ]
})

# Jalankan Router Navigasi
pg.run()
