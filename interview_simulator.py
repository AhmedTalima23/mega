import streamlit as st
# import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util

# Set page config
st.set_page_config(
    page_title="Technical Interview Simulator",
    page_icon="💼",
    layout="centered",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_models():
    """Load all models with GPU support if available"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    t5_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(device)
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)
    return t5_tokenizer, t5_model, embedding_model, device

def generate_question_prompt(job_role, level):
    """Generate prompt for question generation"""
    return f"""
Generate a challenging technical interview question for a {level} {job_role}.
The question should:
1. Test practical knowledge, not just definitions
2. Be specific and unambiguous
3. Require in-depth {job_role} knowledge
4. Be appropriate for {level} level
5. Challenge the candidate's expertise

Return ONLY the question with no additional text.
"""

def generate_answer_prompt(question, job_role, level):
    """Generate prompt for answer generation"""
    return f"""
Create a model answer for this {level} {job_role} question:
"{question}"

Include:
1. Technical depth with examples
2. Best practices and tradeoffs
3. Both theory and practical application
4. What a top candidate would say

Return a detailed, well-structured answer.
"""

def generate_question(job_role, level, models):
    """Generate interview question"""
    t5_tokenizer, t5_model, _, device = models
    prompt = generate_question_prompt(job_role, level)
    input_ids = t5_tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    output = t5_model.generate(
        input_ids, 
        max_length=150,
        do_sample=True,
        top_p=0.92,
        temperature=0.85,
        no_repeat_ngram_size=3
    )
    
    question = t5_tokenizer.decode(output[0], skip_special_tokens=True).strip()
    if question.startswith('"') and question.endswith('"'):
        question = question[1:-1].strip()
    return question

def generate_ideal_answer(question, job_role, level, models):
    """Generate model answer"""
    t5_tokenizer, t5_model, _, device = models
    prompt = generate_answer_prompt(question, job_role, level)
    input_ids = t5_tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    output = t5_model.generate(
        input_ids,
        max_length=512,
        do_sample=True,
        top_p=0.95,
        temperature=0.7
    )
    
    return t5_tokenizer.decode(output[0], skip_special_tokens=True).strip()

def evaluate_answer(user_answer, ideal_answer, models):
    """Evaluate user's answer"""
    _, _, embedding_model, _ = models
    
    if not user_answer.strip() or not ideal_answer.strip():
        return 0.0
    
    with torch.no_grad():
        emb1 = embedding_model.encode(user_answer, convert_to_tensor=True)
        emb2 = embedding_model.encode(ideal_answer, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(emb1, emb2).item()
    
    return similarity

def generate_feedback(user_answer, ideal_answer, models):
    """Generate technical feedback"""
    t5_tokenizer, t5_model, _, device = models
    prompt = f"""
Compare these answers to the same interview question:

USER ANSWER:
{user_answer}

IDEAL ANSWER:
{ideal_answer}

Provide specific technical feedback about:
1. Key concepts missed
2. Technical strengths
3. Improvement suggestions

Return ONLY the feedback points.
"""
    
    input_ids = t5_tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    output = t5_model.generate(input_ids, max_length=200)
    return t5_tokenizer.decode(output[0], skip_special_tokens=True).strip()

def main():
    """Streamlit app interface"""
    st.title("💼 Technical Interview Simulator")
    st.write("Practice your technical interview skills with AI-generated questions and feedback.")
    
    # Initialize session state
    if 'question_count' not in st.session_state:
        st.session_state.question_count = 0
    if 'total_score' not in st.session_state:
        st.session_state.total_score = 0
    if 'current_question' not in st.session_state:
        st.session_state.current_question = ""
    if 'ideal_answer' not in st.session_state:
        st.session_state.ideal_answer = ""
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Interview Settings")
        job_role = st.selectbox(
            "Job Role",
            ["Software Engineer", "Data Scientist", "Frontend Developer", 
             "Backend Engineer", "DevOps Engineer", "Machine Learning Engineer"],
            index=0
        )
        level = st.radio("Experience Level", ["Junior", "Mid", "Senior"], index=1)
        num_questions = st.slider("Number of Questions", 1, 5, 3)
        
        if st.button("New Interview"):
            st.session_state.question_count = 0
            st.session_state.total_score = 0
            st.session_state.current_question = ""
            st.session_state.ideal_answer = ""
            st.experimental_rerun()
    
    # Load models (cached)
    models = load_models()
    
    # Main interview area
    if st.session_state.question_count < num_questions:
        if not st.session_state.current_question:
            st.session_state.current_question = generate_question(job_role, level, models)
            st.session_state.question_count += 1
        
        st.subheader(f"Question {st.session_state.question_count}/{num_questions}")
        st.markdown(f"**{st.session_state.current_question}**")
        
        user_answer = st.text_area("Your Answer:", height=150, key=f"answer_{st.session_state.question_count}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Submit Answer"):
                if user_answer.strip():
                    st.session_state.ideal_answer = generate_ideal_answer(
                        st.session_state.current_question, job_role, level, models
                    )
                    similarity = evaluate_answer(user_answer, st.session_state.ideal_answer, models)
                    st.session_state.total_score += similarity
                    
                    st.subheader("Feedback")
                    st.progress(similarity)
                    st.write(f"**Similarity Score:** {similarity:.2f}/1.00")
                    
                    if similarity >= 0.8:
                        st.success("Excellent! You covered all key points well.")
                    elif similarity >= 0.6:
                        st.info("Good answer, but could use more depth/examples.")
                    else:
                        st.warning("Keep practicing - focus on technical details.")
                    
                    st.write("**Technical Feedback:**")
                    feedback = generate_feedback(user_answer, st.session_state.ideal_answer, models)
                    st.write(feedback)
                    
                    with st.expander("View Ideal Answer"):
                        st.markdown(st.session_state.ideal_answer)
                else:
                    st.warning("Please enter your answer before submitting.")
        
        with col2:
            if st.button("Skip Question"):
                st.session_state.current_question = ""
                st.experimental_rerun()
    else:
        # Interview completion
        st.balloons()
        st.success("🎉 Interview Completed!")
        avg_score = st.session_state.total_score / num_questions
        
        st.subheader("Final Results")
        st.metric("Average Score", f"{avg_score:.2f}/1.00")
        
        if avg_score >= 0.8:
            st.success("Excellent performance! You're well prepared for technical interviews.")
        elif avg_score >= 0.6:
            st.info("Good job! With a bit more practice you'll be interview-ready.")
        else:
            st.warning("Keep practicing! Focus on technical depth and examples.")
        
        st.write("**Tips for improvement:**")
        st.markdown("""
        - Structure answers with clear beginning/middle/end
        - Include specific examples from your experience
        - Discuss tradeoffs and alternatives
        - Cover both theory and practical application
        """)
        
        if st.button("Start New Interview"):
            st.session_state.question_count = 0
            st.session_state.total_score = 0
            st.session_state.current_question = ""
            st.session_state.ideal_answer = ""
            st.experimental_rerun()

if __name__ == "__main__":
    main()
