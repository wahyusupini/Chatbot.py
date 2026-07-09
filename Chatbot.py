import streamlit as st
from openai import OpenAI
import anthropic
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain_openai import ChatOpenAI, OpenAI as LangChainOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from streamlit_feedback import streamlit_feedback

# ==========================================
# CONFIGURATION & SIDEBAR NAVIGATION
# ==========================================
st.set_page_config(page_title="AI Multi-Tool Hub", page_icon="🚀", layout="wide")

st.sidebar.title("🛠️ AI Multi-Tool Hub")
menu = st.sidebar.selectbox(
    "Pilih Fitur Aplikasi:",
    [
        "💬 Simple Chatbot (OpenAI)",
        "📝 File Q&A (Anthropic)",
        "🔎 Chat with Search (LangChain)",
        "🦜 LangChain Quickstart",
        "✍️ Blog Outline Generator",
        "⭐ Chat with Feedback (Trubrics)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔑 API Key Configuration")

# Manajemen API Key Terpusat di Sidebar
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password", key="global_openai_key")
anthropic_api_key = st.sidebar.text_input("Anthropic API Key", type="password", key="global_anthropic_key")

st.sidebar.markdown("---")
st.sidebar.markdown("[📄 Dapatkan OpenAI API key](https://platform.openai.com/account/api-keys)")
st.sidebar.markdown("[📄 Dapatkan Anthropic API key](https://console.anthropic.com/)")

# ==========================================
# 1. SIMPLE CHATBOT (OPENAI)
# ==========================================
if menu == "💬 Simple Chatbot (OpenAI)":
    st.title("💬 Simple Chatbot")
    st.caption("🚀 A Streamlit chatbot powered by OpenAI GPT-4o-mini")
    
    if "messages_simple" not in st.session_state:
        st.session_state["messages_simple"] = [{"role": "assistant", "content": "How can I help you today?"}]

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

# ==========================================
# 2. FILE Q&A (ANTHROPIC)
# ==========================================
elif menu == "📝 File Q&A (Anthropic)":
    st.title("📝 File Q&A with Anthropic")
    uploaded_file = st.file_uploader("Upload an article", type=("txt", "md"))
    question = st.text_input(
        "Ask something about the article",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )

    if uploaded_file and question and not anthropic_api_key:
        st.info("Please add your Anthropic API key in the sidebar to continue.")

    if uploaded_file and question and anthropic_api_key:
        article = uploaded_file.read().decode()
        # Menggunakan format Anthropic modern jika diperlukan, atau format prompt lama
        full_prompt = f"{anthropic.HUMAN_PROMPT} Here's an article:\n\n<article>\n{article}\n</article>\n\n{question}{anthropic.AI_PROMPT}"

        client = anthropic.Client(api_key=anthropic_api_key)
        # Menggunakan model claude terbaru yang stabil untuk teks
        response = client.completions.create(
            prompt=full_prompt,
            stop_sequences=[anthropic.HUMAN_PROMPT],
            model="claude-2.1",
            max_tokens_to_sample=500,
        )
        st.write("### Answer")
        st.write(response.completion)

# ==========================================
# 3. CHAT WITH SEARCH (LANGCHAIN)
# ==========================================
elif menu == "🔎 Chat with Search (LangChain)":
    st.title("🔎 LangChain - Chat with search")
    st.caption("Using StreamlitCallbackHandler to display agent thoughts in real-time.")

    if "messages_search" not in st.session_state:
        st.session_state["messages_search"] = [
            {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
        ]

    for msg in st.session_state.messages_search:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input(placeholder="Who won the latest Formula 1 championship?"):
        st.session_state.messages_search.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        if not openai_api_key:
            st.info("Please add your OpenAI API key in the sidebar to continue.")
            st.stop()

        llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=openai_api_key, streaming=True)
        search = DuckDuckGoSearchRun(name="Search")
        search_agent = initialize_agent(
            [search], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True
        )
        
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            # Konversi history ke format string agar kompatibel dengan basic Zero-Shot Agent
            chat_history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages_search])
            response = search_agent.run(chat_history_str, callbacks=[st_cb])
            st.session_state.messages_search.append({"role": "assistant", "content": response})
            st.write(response)

# ==========================================
# 4. LANGCHAIN QUICKSTART
# ==========================================
elif menu == "🦜 LangChain Quickstart":
    st.title("🦜🔗 Langchain Quickstart App")

    def generate_response(input_text):
        llm = LangChainOpenAI(temperature=0.7, openai_api_key=openai_api_key, model="gpt-3.5-turbo-instruct")
        st.info(llm.invoke(input_text))

    with st.form("my_form"):
        text = st.text_area("Enter text:", "What are 3 key advice for learning how to code?")
        submitted = st.form_submit_button("Submit")
        if not openai_api_key:
            st.info("Please add your OpenAI API key in the sidebar to continue.")
        elif submitted:
            generate_response(text)

# ==========================================
# 5. BLOG OUTLINE GENERATOR
# ==========================================
elif menu == "✍️ Blog Outline Generator":
    st.title("🦜🔗 Langchain - Blog Outline Generator App")

    def blog_outline(topic):
        llm = LangChainOpenAI(temperature=0.7, openai_api_key=openai_api_key, model="gpt-3.5-turbo-instruct")
        template = "As an experienced data scientist and technical writer, generate an outline for a blog about {topic}."
        prompt = PromptTemplate(input_variables=["topic"], template=template)
        prompt_query = prompt.format(topic=topic)
        response = llm.invoke(prompt_query)
        return st.info(response)

    with st.form("myform"):
        topic_text = st.text_input("Enter prompt (e.g. Quantum Computing):", "")
        submitted = st.form_submit_button("Submit")
        if not openai_api_key:
            st.info("Please add your OpenAI API key in the sidebar to continue.")
        elif submitted:
            blog_outline(topic_text)

# ==========================================
# 6. CHAT WITH FEEDBACK (TRUBRICS)
# ==========================================
elif menu == "⭐ Chat with Feedback (Trubrics)":
    st.title("📝 Chat with feedback (Trubrics)")
    st.caption("Collect user feedback (thumbs up/down) and send it to Trubrics.")

    if "messages_feedback" not in st.session_state:
        st.session_state["messages_feedback"] = [
            {"role": "assistant", "content": "How can I help you? Leave feedback to help me improve!"}
        ]
    if "last_response" not in st.session_state:
        st.session_state["last_response"] = None

    for msg in st.session_state.messages_feedback:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input(placeholder="Tell me a joke about programming"):
        st.session_state.messages_feedback.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        if not openai_api_key:
            st.info("Please add your OpenAI API key in the sidebar to continue.")
            st.stop()
            
        client = OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(model="gpt-4o-mini", messages=st.session_state.messages_feedback)
        st.session_state["last_response"] = response.choices[0].message.content
        
        with st.chat_message("assistant"):
            st.session_state.messages_feedback.append({"role": "assistant", "content": st.session_state["last_response"]})
            st.write(st.session_state["last_response"])

    if st.session_state["last_response"]:
        feedback = streamlit_feedback(
            feedback_type="thumbs",
            optional_text_label="[Optional] Please provide an explanation",
            key=f"fb_{len(st.session_state.messages_feedback)}",
        )
        
        # Pengiriman ke Trubrics jika setup secret tersedia
        import os
        # Mendukung pembacaan dari st.secrets (Streamlit Cloud)
        trubrics_email = st.secrets.get("TRUBRICS_EMAIL") if "TRUBRICS_EMAIL" in st.secrets else None
        trubrics_password = st.secrets.get("TRUBRICS_PASSWORD") if "TRUBRICS_PASSWORD" in st.secrets else None
        
        if feedback and trubrics_email and trubrics_password:
            import trubrics
            config = trubrics.init(email=trubrics_email, password=trubrics_password)
            collection = trubrics.collect(
                component_name="default",
                model="gpt-4o-mini",
                response=feedback,
                metadata={"chat": st.session_state.messages_feedback},
            )
            trubrics.save(config, collection)
            st.toast("Feedback recorded successfully!", icon="📝")
