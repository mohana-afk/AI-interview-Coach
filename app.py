import json
import random
import re
from pathlib import Path

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Data Loading & Filtering
# ---------------------------------------------------------------------------

def load_questions() -> list:
    """Load the JSON question bank silently into memory."""
    questions_path = Path(__file__).parent / "questions.json"
    if not questions_path.exists():
        return []
    with open(questions_path, "r", encoding="utf-8") as f:
        return json.load(f)

def filter_questions(questions, role, qtype, difficulty):
    matched = [
        q for q in questions
        if (q.get("role") == role or role == "All")
        and (q.get("type") == qtype or qtype == "Mixed")
        and (q.get("difficulty") == difficulty or difficulty == "All")
    ]
    if len(matched) < 5:
        role_matched = [q for q in questions if q.get("role") == role or role == "All"]
        if len(role_matched) >= 5:
            matched = role_matched
        else:
            matched = questions
    return matched

# ---------------------------------------------------------------------------
# NLP Evaluation Logic
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())

def match_concept_presence(concept: dict, answer_clean: str):
    synonyms = concept.get("synonyms", [])
    negations = concept.get("negations", [])

    for neg in negations:
        neg_clean = clean_text(neg)
        if neg_clean in answer_clean:
            return False, True

    tokens = set(answer_clean.split())
    for phrase in synonyms:
        p_clean = clean_text(phrase)
        if p_clean in answer_clean:
            return True, False

        p_words = p_clean.split()
        for pw in p_words:
            if len(pw) >= 4:
                root = pw[:4]
                if any(t.startswith(root) for t in tokens):
                    return True, False

    return False, False

def evaluate_single_answer(user_answer: str, question_obj: dict) -> dict:
    cleaned_ans = clean_text(user_answer)
    words = cleaned_ans.split()
    word_count = len(words)

    if word_count == 0:
        return {
            "score": 0.0,
            "assessment": "No Answer",
            "got_right": [],
            "missing": [c["name"] for c in question_obj.get("key_concepts", [])],
            "expected_answer": question_obj.get("expected_answer", "N/A"),
            "suggestion": "No answer provided. Please enter a response to practice."
        }

    expected_answer = question_obj.get("expected_answer", "")
    key_concepts = question_obj.get("key_concepts", [])
    q_type = question_obj.get("type", "Technical")

    concepts_found = []
    concepts_missed = []
    contradictions = []

    for c in key_concepts:
        matched, contradicted = match_concept_presence(c, cleaned_ans)
        if matched:
            concepts_found.append(c["name"])
        elif contradicted:
            contradictions.append(c["name"])
            concepts_missed.append(f"{c['name']} (Incorrectly stated)")
        else:
            concepts_missed.append(c["name"])

    total_concepts = len(key_concepts) if key_concepts else 1
    concept_ratio = len(concepts_found) / total_concepts

    try:
        corpus = [cleaned_ans, clean_text(expected_answer)]
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        tfidf = vectorizer.fit_transform(corpus)
        sim = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
    except Exception:
        sim = 0.0

    if len(concepts_found) == 0 and len(contradictions) == 0 and sim < 0.09:
        return {
            "score": 1.2 if word_count > 6 else 0.5,
            "assessment": "Incorrect / Irrelevant",
            "got_right": [],
            "missing": [c["name"] for c in key_concepts],
            "expected_answer": expected_answer,
            "suggestion": "This response does not address the question asked. Please review the expected answer."
        }

    if q_type == "HR":
        concept_score = concept_ratio * 5.0
        depth_score = min(word_count / 25.0, 1.0) * 3.0
        sim_score = min(sim * 3.0, 2.0)
        total = concept_score + depth_score + sim_score
    else:
        concept_score = concept_ratio * 7.5
        sim_score = min(sim * 2.0, 1.5)
        clarity_bonus = 1.0 if word_count >= 10 else 0.0
        total = concept_score + sim_score + clarity_bonus

    if contradictions:
        total = min(total, 5.0)

    if concept_ratio >= 0.95 and not contradictions:
        total = max(total, 8.8)
    elif concept_ratio >= 0.65 and not contradictions:
        total = max(total, 7.5)

    final_score = round(min(max(total, 1.0), 10.0), 1)

    if final_score >= 8.5:
        assessment = "Correct"
    elif final_score >= 7.0:
        assessment = "Mostly Correct"
    elif final_score >= 4.0:
        assessment = "Partially Correct"
    else:
        assessment = "Incorrect"

    if contradictions:
        suggestion = "You covered some parts well, but note: Python lists are mutable (their contents CAN be changed). Tuples are the immutable collection."
    elif assessment == "Correct":
        suggestion = "Great job! You explained the main ideas accurately and covered all important concepts."
    elif assessment == "Mostly Correct":
        suggestion = f"Strong answer! You captured the core idea accurately. Mentioning {', '.join(concepts_missed)} would make your answer completely thorough."
    elif assessment == "Partially Correct":
        suggestion = f"Good start! You explained part of the concept. To improve, make sure to address: {', '.join(concepts_missed)}."
    else:
        suggestion = "Review the expected answer below to build a stronger understanding of the core concept."

    return {
        "score": final_score,
        "assessment": assessment,
        "got_right": concepts_found,
        "missing": concepts_missed,
        "expected_answer": expected_answer,
        "suggestion": suggestion
    }

# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def restart_interview():
    st.session_state.interview_started = False
    st.session_state.current_index = 0
    st.session_state.questions = []
    st.session_state.user_answers = []
    st.session_state.evaluation_results = []
    st.session_state.completed = False

def init_session():
    if "interview_started" not in st.session_state:
        st.session_state.interview_started = False
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    if "questions" not in st.session_state:
        st.session_state.questions = []
    if "user_answers" not in st.session_state:
        st.session_state.user_answers = []
    if "evaluation_results" not in st.session_state:
        st.session_state.evaluation_results = []
    if "completed" not in st.session_state:
        st.session_state.completed = False

def main():
    st.set_page_config(page_title="AI Interview Coach", page_icon="🎯", layout="centered")
    init_session()

    st.title("🎯 AI Interview Coach")
    st.markdown("Practice interview questions with local, concept-driven NLP evaluation.")
    st.divider()

    # 1. Config Screen
    if not st.session_state.interview_started and not st.session_state.completed:
        st.subheader("Configure Your Mock Interview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            role = st.selectbox("Job Role", ["Python Developer", "Data Analyst", "AI/ML Engineer"])
        with col2:
            qtype = st.selectbox("Interview Type", ["Technical", "HR", "Mixed"])
        with col3:
            difficulty = st.selectbox("Difficulty", ["Beginner", "Intermediate", "Advanced"])

        st.info("ℹ️ You will answer **5 questions**. Complete feedback and scores will be shown **at the end**.")

        if st.button("🚀 Start Interview", use_container_width=True, type="primary"):
            all_questions = load_questions()
            if not all_questions:
                st.error("No questions found in questions.json.")
                return

            candidates = filter_questions(all_questions, role, qtype, difficulty)
            random.shuffle(candidates)
            selected = candidates[:5]

            while len(selected) < 5 and len(all_questions) > len(selected):
                rem = [q for q in all_questions if q not in selected]
                if rem:
                    selected.append(random.choice(rem))

            st.session_state.questions = selected
            st.session_state.current_index = 0
            st.session_state.user_answers = []
            st.session_state.interview_started = True
            st.session_state.completed = False
            st.rerun()

    # 2. In-Progress Screen
    elif st.session_state.interview_started and not st.session_state.completed:
        idx = st.session_state.current_index
        total_questions = len(st.session_state.questions)
        current_q = st.session_state.questions[idx]

        st.progress(idx / total_questions, text=f"Question {idx + 1} of {total_questions}")

        st.markdown(f"### Question {idx + 1}")
        st.markdown(f"**{current_q.get('question')}**")
        st.caption(f"Role: {current_q.get('role', 'General')} | Type: {current_q.get('type', 'Technical')} | Difficulty: {current_q.get('difficulty', 'Beginner')}")

        answer_key = f"ans_input_{idx}"
        answer = st.text_area(
            "Your Answer:",
            height=160,
            key=answer_key,
            placeholder="Type your explanation here..."
        )

        col_next, col_cancel = st.columns([4, 1])
        with col_next:
            btn_text = "Next Question ➡️" if idx < total_questions - 1 else "Finish & View Results 🏁"
            if st.button(btn_text, type="primary", use_container_width=True):
                if not answer.strip():
                    st.warning("Please type an answer before proceeding.")
                else:
                    st.session_state.user_answers.append({
                        "question_data": current_q,
                        "user_answer": answer.strip()
                    })

                    if idx + 1 < total_questions:
                        st.session_state.current_index += 1
                        st.rerun()
                    else:
                        with st.spinner("Evaluating your responses..."):
                            eval_list = []
                            for item in st.session_state.user_answers:
                                eval_data = evaluate_single_answer(item["user_answer"], item["question_data"])
                                eval_list.append({
                                    "question": item["question_data"].get("question"),
                                    "user_answer": item["user_answer"],
                                    "eval": eval_data
                                })
                            st.session_state.evaluation_results = eval_list
                            st.session_state.completed = True
                            st.session_state.interview_started = False
                            st.rerun()

        with col_cancel:
            if st.button("Cancel", use_container_width=True):
                restart_interview()
                st.rerun()

    # 3. Final Results Screen
    elif st.session_state.completed:
        st.balloons()
        st.header("📊 Final Performance Report")
        st.markdown("Here is your detailed question-by-question evaluation.")

        results = st.session_state.evaluation_results
        total_score = sum(r["eval"]["score"] for r in results)
        avg_score = round(total_score / len(results), 1) if results else 0.0

        col1, col2, col3 = st.columns(3)
        col1.metric("Average Score", f"{avg_score} / 10")
        col2.metric("Questions Evaluated", f"{len(results)} of 5")

        if avg_score >= 8.0:
            perf_level = "🌟 Interview Ready"
        elif avg_score >= 6.5:
            perf_level = "👍 Strong Contender"
        elif avg_score >= 4.5:
            perf_level = "🌱 Developing Foundation"
        else:
            perf_level = "📚 Needs Practice"
        col3.metric("Performance Level", perf_level)

        if avg_score >= 7.5:
            st.success("🎉 **Well done!** You demonstrated a clear grasp of the required principles.")
        elif avg_score >= 5.5:
            st.info("💡 **Good effort!** You communicated several key ideas well.")
        else:
            st.info("🌱 **Good practice session!** Review the expected answers below and try another practice run.")

        st.divider()
        st.subheader("Question-by-Question Breakdown")

        badge_colors = {
            "Correct": "🟢",
            "Mostly Correct": "🔵",
            "Partially Correct": "🟡",
            "Incorrect": "🔴",
            "Incorrect / Irrelevant": "🔴",
            "No Answer": "⚪"
        }

        for i, item in enumerate(results, start=1):
            e = item["eval"]
            badge = badge_colors.get(e["assessment"], "🔵")
            with st.expander(f"Q{i}: {item['question'][:65]}... — {badge} {e['assessment']} ({e['score']}/10)", expanded=(i == 1)):
                st.markdown(f"**Question:** {item['question']}")
                st.markdown(f"**Your Answer:**")
                st.info(item["user_answer"])

                col_stat1, col_stat2 = st.columns(2)
                col_stat1.metric("Score", f"{e['score']} / 10")
                col_stat2.markdown(f"**Assessment:** {badge} **{e['assessment']}**")

                col_good, col_miss = st.columns(2)
                with col_good:
                    st.markdown("**✅ Concepts You Covered:**")
                    if e["got_right"]:
                        for c in e["got_right"]:
                            st.write(f"- `{c}`")
                    else:
                        st.write("*(None detected)*")

                with col_miss:
                    st.markdown("**⚠️ Concepts Missing / Incomplete:**")
                    if e["missing"]:
                        for m in e["missing"]:
                            st.write(f"- `{m}`")
                    else:
                        st.write("*(All primary concepts covered)*")

                st.markdown("**📖 Expected / Reference Answer:**")
                st.success(e["expected_answer"])

                st.markdown(f"**💡 Specific Suggestion:** {e['suggestion']}")

        st.divider()
        if st.button("🔄 Restart Interview", type="primary", use_container_width=True):
            restart_interview()
            st.rerun()

if __name__ == "__main__":
    main()