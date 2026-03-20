import streamlit as st
from crewai import LLM, Agent, Task, Crew
from crewai_tools import SerperDevTool
import os
from dotenv import load_dotenv
from datetime import datetime
import time

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Homework Helper AI",
    page_icon="📚",
    layout="wide"
)

# Keep session alive
if 'session_start' not in st.session_state:
    st.session_state.session_start = time.time()

# ═══════════════════════════════════════
# HISTORY FUNCTIONS
# ═══════════════════════════════════════
def load_history(name):
    key = f"history_{name.lower().strip()}"
    if key not in st.session_state:
        st.session_state[key] = []
    return st.session_state[key]

def save_history(name, history):
    key = f"history_{name.lower().strip()}"
    st.session_state[key] = history

# ═══════════════════════════════════════
# INITIALIZE LLM
# ═══════════════════════════════════════
@st.cache_resource
def get_llm():
    return LLM(
        model="groq/llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=3000,
        temperature=0.2
    )

# ═══════════════════════════════════════
# INITIALIZE SEARCH TOOL
# ═══════════════════════════════════════
@st.cache_resource
def get_search_tool():
    return SerperDevTool()

# ═══════════════════════════════════════
# HERO SECTION
# ═══════════════════════════════════════
st.title("📚 Homework Helper AI")
st.subheader("Your intelligent academic assistant — get accurate answers for any subject instantly")
st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Subjects", "10+")
with col2:
    st.metric("Powered By", "AI")
with col3:
    st.metric("Available", "24/7")

st.divider()

# ═══════════════════════════════════════
# NOTICE
# ═══════════════════════════════════════
st.info("💡 **Tip:** Always use the **same name** every time to access your previous question history!")

# ═══════════════════════════════════════
# MAIN LAYOUT
# ═══════════════════════════════════════
col_main, col_history = st.columns([3, 2])

# ═══════════════════════════════════════
# LEFT COLUMN - ASK QUESTION
# ═══════════════════════════════════════
with col_main:
    st.subheader("🎯 Ask Your Question")

    with st.form("homework_form"):
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input(
                "Your Name",
                placeholder="e.g. M.Uzair",
            )

        with col2:
            subject = st.selectbox(
                "Subject",
                [
                    "📐 Mathematics",
                    "🔬 Science",
                    "📜 History",
                    "📝 English",
                    "🌍 Geography",
                    "⚡ Physics",
                    "🧪 Chemistry",
                    "🧬 Biology",
                    "💻 Computer Science",
                    "📚 Other"
                ],
            )

        question = st.text_area(
            "Your Question",
            placeholder="Type your homework question here in detail for the most accurate answer...",
            height=150,
        )

        submitted = st.form_submit_button(
            "🚀 Get Answer Now",
            use_container_width=True,
            type="primary"
        )

# ═══════════════════════════════════════
# RIGHT COLUMN - HISTORY
# ═══════════════════════════════════════
with col_history:
    st.subheader("📜 Your History")

    history_name = st.text_input(
        "Enter your name to view history",
        placeholder="Same name you used to ask questions"
    )

    if history_name:
        history = load_history(history_name)
        if history:
            st.success(f"Found **{len(history)}** questions for **{history_name}**")
            for item in reversed(history):
                clean_subject = item['subject'].replace("📐 ", "").replace("🔬 ", "").replace("📜 ", "").replace("📝 ", "").replace("🌍 ", "").replace("⚡ ", "").replace("🧪 ", "").replace("🧬 ", "").replace("💻 ", "").replace("📚 ", "")
                with st.expander(f"📚 {clean_subject} — {item['question'][:50]}..."):
                    st.caption(f"🕐 {item['date']}")
                    st.write(item['answer'])
        else:
            st.warning(f"No history found for **{history_name}**. Ask your first question!")
    else:
        st.caption("Enter your name above to view your question history")

# ═══════════════════════════════════════
# PROCESS QUESTION
# ═══════════════════════════════════════
if submitted:
    if not name:
        st.error("⚠️ Please enter your name")
    elif not question:
        st.error("⚠️ Please enter your homework question")
    elif not os.getenv("GROQ_API_KEY"):
        st.error("⚠️ API not configured. Please contact admin.")
    else:
        st.divider()

        progress = st.progress(0, text="🔍 Starting research...")
        time.sleep(0.5)
        progress.progress(20, text="🌐 Searching the web...")

        try:
            llm = get_llm()
            search_tool = get_search_tool()

            progress.progress(40, text="🤖 Researcher agent working...")

            researcher = Agent(
                role='Expert Academic Researcher',
                goal='Find the most accurate and comprehensive information',
                backstory='''You are a world-class academic researcher with expertise in all subjects.
                You always find the most accurate, detailed, and up-to-date information.
                You use web search to verify facts and provide comprehensive answers.
                You never guess — you always research before answering.''',
                llm=llm,
                tools=[search_tool],
                verbose=False
            )

            teacher = Agent(
                role='Expert Teacher & Academic Tutor',
                goal='Explain complex topics clearly and completely',
                backstory='''You are an award-winning teacher with 20 years of experience.
                You explain topics step by step with examples and analogies.
                You always give complete, well-structured explanations.
                You make sure students fully understand the topic.''',
                llm=llm,
                verbose=False
            )

            research_task = Task(
                description=f'''Research this {subject} question thoroughly: {question}
                - Find accurate and detailed information from reliable sources
                - Include all important facts, dates, examples, and context
                - Make sure information is correct, complete and well-organized
                - Search web for most accurate and up to date information''',
                expected_output='A comprehensive, accurate, detailed answer with all relevant facts',
                agent=researcher
            )

            explain_task = Task(
                description=f'''Create a complete educational explanation for {name} about: {question}
                Structure your answer like this:
                1. Quick Summary (2-3 sentences)
                2. Detailed Explanation (step by step)
                3. Key Points to Remember
                4. Real World Examples
                5. Conclusion
                Make it complete, accurate and easy to understand for a student.''',
                expected_output='A well-structured complete educational answer with all 5 sections',
                agent=teacher,
                context=[research_task]
            )

            progress.progress(60, text="👨‍🏫 Teacher agent explaining...")

            crew = Crew(
                agents=[researcher, teacher],
                tasks=[research_task, explain_task],
                verbose=False,
                max_rpm=2
            )

            result = crew.kickoff()
            result_str = str(result)

            progress.progress(90, text="💾 Saving to history...")

            # Save to history
            history = load_history(name)
            history.append({
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "subject": subject,
                "question": question,
                "answer": result_str
            })
            save_history(name, history)

            progress.progress(100, text="✅ Done!")
            time.sleep(0.5)
            progress.empty()

            # Display result
            st.success("✅ Answer ready!")
            st.divider()

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Student", name)
            with col_b:
                st.metric("Subject", subject.split(" ", 1)[1] if " " in subject else subject)
            with col_c:
                st.metric("Date", datetime.now().strftime("%d %b %Y"))

            st.divider()
            st.subheader("💡 Answer")
            st.write(result_str)
            st.divider()

            result_text = f"Homework Helper AI\nStudent: {name}\nSubject: {subject}\nQuestion: {question}\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n{result_str}"

            st.download_button(
                label="📥 Download Answer",
                data=result_text,
                file_name=f"{name}_{subject}_answer.txt",
                mime="text/plain",
                use_container_width=True
            )

        except Exception as e:
            progress.empty()
            st.error(f"❌ An error occurred: {str(e)}")
            st.info("💡 Please try again later")

# ═══════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════
st.divider()
st.caption("Powered by CrewAI & Groq (Llama 3.3 70B) | Homework Helper AI | Built by M.Uzair")