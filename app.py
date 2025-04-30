import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util

# Load models once and cache
@st.cache_resource
def load_models():
    t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    t5_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return t5_tokenizer, t5_model, embed_model

t5_tokenizer, t5_model, embedding_model = load_models()

# Question prompt generator
def generate_question_prompt(job_role, level):
    return f"Generate a technical interview question for a {level} {job_role}."

# Answer prompt generator
def generate_answer_prompt(question, job_role, level):
    return f"""
You are a technical hiring manager for {job_role} positions.

Create a model answer to this {level}-level interview question:
"{question}"

Format: Provide a detailed, well-organized answer.
"""

# Generate question
def generate_question(job_role, level):
    prompt = generate_question_prompt(job_role, level)
    input_ids = t5_tokenizer(prompt, return_tensors="pt").input_ids
    output = t5_model.generate(input_ids, max_length=150, top_p=0.9, temperature=0.8, do_sample=True)
    return t5_tokenizer.decode(output[0], skip_special_tokens=True).strip()

# Generate model answer
def generate_ideal_answer(question, job_role, level):
    prompt = generate_answer_prompt(question, job_role, level)
    input_ids = t5_tokenizer(prompt, return_tensors="pt").input_ids
    output = t5_model.generate(input_ids, max_length=512, top_p=0.95, temperature=0.7, do_sample=True)
    return t5_tokenizer.decode(output[0], skip_special_tokens=True).strip()

# Technical feedback
def generate_technical_feedback(user_answer, ideal_answer):
    prompt = f"""
Compare these two answers to the same technical interview question:

USER ANSWER:
{user_answer}

IDEAL ANSWER:
{ideal_answer}

Provide feedback:
1. Concepts missed or incorrect
2. Strengths
3. Suggestions for improvement
"""
    input_ids = t5_tokenizer(prompt, return_tensors="pt").input_ids
    output = t5_model.generate(input_ids, max_length=200)
    return t5_tokenizer.decode(output[0], skip_special_tokens=True).strip()

# Similarity scoring
def calculate_similarity(ans1, ans2):
    emb1 = embedding_model.encode(ans1, convert_to_tensor=True)
    emb2 = embedding_model.encode(ans2, convert_to_tensor=True)
    return util.pytorch_cos_sim(emb1, emb2).item()

# Score + feedback
def evaluate_answer(user_answer, ideal_answer):
    similarity = calculate_similarity(user_answer, ideal_answer)
    if similarity >= 0.8:
        score, feedback = 1.0, "Excellent answer that covers all the key points!"
    elif similarity >= 0.7:
        score, feedback = 0.9, "Very good answer with most key points covered."
    elif similarity >= 0.6:
        score, feedback = 0.8, "Good answer with solid understanding."
    elif similarity >= 0.5:
        score, feedback = 0.7, "Acceptable answer, but needs more detail."
    elif similarity >= 0.4:
        score, feedback = 0.5, "Answer misses some important aspects."
    elif similarity >= 0.3:
        score, feedback = 0.3, "Answer has significant gaps."
    else:
        score, feedback = 0.1, "Answer doesn't effectively address the question."

    tech_feedback = generate_technical_feedback(user_answer, ideal_answer)
    return {"score": score, "similarity": similarity, "general_feedback": feedback, "technical_feedback": tech_feedback}

# Streamlit UI
st.set_page_config(page_title="AI Interview Simulator", layout="wide")
st.title("🤖 Technical Interview Simulator")

# Initialize session state
if "step" not in st.session_state:
    st.session_state.step = "setup"
    st.session_state.job_role = ""
    st.session_state.level = "Junior"
    st.session_state.q_index = 0
    st.session_state.total_score = 0
    st.session_state.question = ""
    st.session_state.ideal_answer = ""

# Setup step
if st.session_state.step == "setup":
    st.subheader("Interview Setup")
    st.session_state.job_role = st.text_input("Enter your job role:")
    st.session_state.level = st.selectbox("Choose experience level:", ["Junior", "Mid", "Senior"])
    if st.button("🎬 Begin Interview", disabled=not st.session_state.job_role.strip()):
        st.session_state.q_index = 1
        st.session_state.step = "question"
        st.session_state.question = generate_question(st.session_state.job_role, st.session_state.level)

# Question step
elif st.session_state.step == "question":
    st.subheader(f"Question {st.session_state.q_index}")
    st.write(st.session_state.question)

    user_answer = st.text_area("Your Answer", key=f"answer_{st.session_state.q_index}")
    if st.button("Submit Answer"):
        with st.spinner("Evaluating your answer..."):
            st.session_state.ideal_answer = generate_ideal_answer(
                st.session_state.question, st.session_state.job_role, st.session_state.level)
            result = evaluate_answer(user_answer, st.session_state.ideal_answer)
            st.session_state.total_score += result['score']

        st.session_state.step = "feedback"
        st.session_state.feedback_result = result
        st.rerun()

# Feedback step
elif st.session_state.step == "feedback":
    result = st.session_state.feedback_result
    st.subheader("✅ Feedback")
    st.write(f"**Score:** {result['score']:.2f}")
    st.write(f"**Similarity:** {result['similarity']:.2f}")
    st.write(f"**General Feedback:** {result['general_feedback']}")
    st.markdown("**Technical Feedback:**")
    st.info(result['technical_feedback'])
    with st.expander("💡 Ideal Answer"):
        st.markdown(st.session_state.ideal_answer)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("➡️ Next Question"):
            st.session_state.q_index += 1
            st.session_state.question = generate_question(st.session_state.job_role, st.session_state.level)
            st.session_state.step = "question"
            st.rerun()
    with col2:
        if st.button("❌ Exit Interview"):
            st.session_state.step = "summary"
            st.rerun()

# Summary step
elif st.session_state.step == "summary":
    st.subheader("📋 Interview Summary")
    avg = st.session_state.total_score / st.session_state.q_index
    st.write(f"**You answered {st.session_state.q_index} questions.**")
    st.write(f"**Average Score:** {avg:.2f}/1.00")
    st.success("Thanks for using the AI Interview Simulator!")
    if st.button("🔁 Restart"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()
