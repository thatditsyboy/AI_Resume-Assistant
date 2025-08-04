import streamlit as st
from PyPDF2 import PdfReader
from typing import List
import google.generativeai as genai
import requests
import json

# ---- PAGE CONFIG ----
st.set_page_config(page_title="Resume Assistant", layout="wide")

# ---- PROMPT TEMPLATES FOR ROLES ----
PROMPTS = {
    "Product Manager": """
Act as an experienced hiring manager. Analyze the following resume and provide an insightful evaluation of the candidate's potential fit for a Product Manager role.

Your answer should:
- Highlight the most relevant experiences, skills, and achievements.
- Comment on product sense, leadership, impact, and clarity of communication.
- Point out any potential gaps or red flags.
- Conclude with a short summary of whether the candidate appears to be a strong, moderate, or weak fit for the role, and why.

Resume Context:
{}
Question:
{}
""",
    "Contact Centre": """
Act as a seasoned recruiter evaluating a resume for a Contact Centre position (agent or team lead).

Your answer should:
- Highlight relevant customer-facing or contact centre experience, communication skills, and problem-solving ability.
- Comment on handling high call volumes, empathy, patience, teamwork, and adaptability.
- Identify any possible suitability issues or gaps for the Contact Centre role.
- Conclude with a brief statement on whether the candidate seems a strong, moderate, or weak fit, and why.

Resume Context:
{}
Question:
{}
""",
    "Content & Social Media": """
You are a recruiter assessing candidates for the Content & Social Media Intern position at Ketto Foundation, focused on supporting high-impact CSR campaigns.

Role snapshot:
- Write and edit content for social media
- Conceptualize and script original videos, including field stories
- Support video production and manage social content on Instagram, LinkedIn, YouTube
- Keep up with social trends; bring fresh ideas

Looking for candidates who:
- Have strong writing AND storytelling skills (must!)
- Demonstrate creativity, self-drive, and willingness to experiment
- Communicate clearly and proactively
- Are interested in (or experienced with) digital/video media and social platforms
- Can juggle multiple projects/timelines
- Take initiative, share ideas, and are comfortable with tools like Canva/CapCut (bonus)

Your answer should:
- Highlight the most relevant content creation, writing, or social media experience
- Comment on storytelling, creativity, and communication style
- Note evidence of initiative, teamwork, adaptability, or video interest/skills
- Call out any missing essentials (especially writing/storytelling and communication)
- Conclude with a summary: Does this candidate appear a strong, moderate, or weak fit for this specific intern role? Why?

Resume Context:
{}
Question:
{}
"""
}

# ---- API KEYS ----
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
PERPLEXITY_API_KEY = st.secrets["PERPLEXITY_API_KEY"]

if not GEMINI_API_KEY:
    st.error("Please set your GEMINI_API_KEY in .streamlit/secrets.toml.")
    st.stop()
if not GOOGLE_API_KEY:
    st.error("Please set your GOOGLE_API_KEY in .streamlit/secrets.toml.")
    st.stop()
if not PERPLEXITY_API_KEY:
    st.error("Please set your PERPLEXITY_API_KEY in .streamlit/secrets.toml.")
    st.stop()

# ---- INITIALIZE GEMINI CLIENT ----
genai.configure(api_key=GEMINI_API_KEY)

# ---- PDF TO TEXT ----
@st.cache_data
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    reader = PdfReader(pdf_bytes)
    text = []
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return "\n".join(text)

# ---- LLM QUERIES ----
def ask_llm_gemini(question: str, context: str, prompt_template: str) -> str:
    prompt = prompt_template.format(context, question).strip()
    model = genai.GenerativeModel("models/gemini-2.5-pro")
    response = model.generate_content(prompt)
    return response.text.strip()

def ask_llm_perplexity(question: str, context: str, prompt_template: str) -> str:
    url = "https://api.perplexity.ai/chat/completions"
    prompt = prompt_template.format(context, question).strip()
    payload = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": "Reply concisely as an experienced hiring manager/recruiter for the specified role."},
            {"role": "user", "content": prompt}
        ]
    }
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=40)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Perplexity API error: {str(e)}"

# ---- UI ----
st.title("📄 Resume Chat Assistant: Gemini | Perplexity Sonar")

role = st.selectbox(
    "Which role are you evaluating resumes for?",
    [
        "Product Manager",
        "Contact Centre",
        "Content & Social Media"
    ],
    index=0
)
prompt_template = PROMPTS[role]

pdf_files = st.file_uploader(
    "Upload your resume PDF(s)", type=["pdf"], accept_multiple_files=True)

if pdf_files:
    text = ""
    # Combine all files' text into one!
    for pdf_file in pdf_files:
        with st.spinner(f"Extracting text from {pdf_file.name}..."):
            text += extract_text_from_pdf(pdf_file) + "\n"

    with st.expander("Preview extracted text"):
        st.write(text[:2000] + ("..." if len(text) > 2000 else ""))

    st.subheader("Ask questions about your resume/candidate:")
    if "history" not in st.session_state:
        st.session_state.history = []

    user_question = st.text_input(
        "Your question (e.g., 'Evaluate this candidate.'):") or "Evaluate this candidate."

    if st.button("Send"):
        st.session_state.history.append({"role": "user", "content": user_question})
        with st.spinner("Generating answers from both models..."):
            gemini_answer = ask_llm_gemini(user_question, text, prompt_template)
            sonar_answer = ask_llm_perplexity(user_question, text, prompt_template)
        st.session_state.history.append({
            "role": "assistant",
            "gemini": gemini_answer,
            "sonar": sonar_answer,
        })

    # Render chat with 2 sections: Gemini | Perplexity Sonar
    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            cols = st.columns(2)
            with cols[0]:
                st.markdown(f"#### Gemini 2.5 Pro\n")
                st.write(msg["gemini"])
            with cols[1]:
                st.markdown(f"#### Perplexity Sonar\n")
                st.write(msg["sonar"])
