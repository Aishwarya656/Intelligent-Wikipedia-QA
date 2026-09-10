import streamlit as st

from modules.qa_system import SQuADQA


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Intelligent Wikipedia QA Assistant",
    page_icon="🧠",
    layout="wide"
)


# ==========================================================
# LOAD SYSTEM
# ==========================================================

@st.cache_resource
def load_qa_system():

    return SQuADQA(
        "data/train-v1.1.json",
        "data/dev-v1.1.json"
    )


qa_system = load_qa_system()


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🧠 Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "🏠 Home",
        "❓ QA-Based",
        "🧠 Knowledge-Based",
        "🔎 Wikipedia Explorer"
    ]
)

st.sidebar.markdown("---")

st.sidebar.write(
    "**Dataset:** SQuAD v1.1"
)

st.sidebar.write(
    "**Source:** Wikipedia"
)


# ==========================================================
# HEADER
# ==========================================================

st.title(
    "🧠 Intelligent Wikipedia Question Answering Assistant"
)

st.caption(
    "SQuAD-based Question Answering + "
    "Knowledge-Based Question Answering"
)


# ==========================================================
# HOME
# ==========================================================

if page == "🏠 Home":

    st.header("Welcome")

    st.write(
        """
        This system uses Wikipedia knowledge contained in
        the Stanford Question Answering Dataset (SQuAD v1.1).

        It provides two main approaches:

        **QA-Based:** finds a relevant question-answer pair
        and its supporting Wikipedia passage.

        **Knowledge-Based:** extracts structured
        Subject → Relation → Object knowledge.
        """
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("❓ QA-Based")

        st.write(
            "Question retrieval and answer extraction "
            "using the SQuAD dataset."
        )

    with col2:

        st.subheader("🧠 Knowledge-Based")

        st.write(
            "Knowledge extraction using "
            "Subject → Relation → Object triples."
        )

    st.markdown("---")

    st.subheader("System Flow")

    st.code(
        """
User Question
      ↓
Question Processing
      ↓
Information Retrieval
      ↓
Wikipedia / SQuAD Knowledge
      ↓
 ┌──────────────────────┐
 │                      │
 ▼                      ▼
QA-Based            Knowledge-Based
 │                      │
 ▼                      ▼
Answer + Context    Subject → Relation → Object
        """,
        language="text"
    )


# ==========================================================
# QA-BASED
# ==========================================================

elif page == "❓ QA-Based":

    st.header("❓ QA-Based Question Answering")

    st.write(
        "Ask a question and retrieve the most relevant "
        "answer from the SQuAD knowledge base."
    )

    question = st.text_input(
        "Enter your question:",
        placeholder=(
            "Where is the University of Notre Dame located?"
        )
    )

    if st.button(
        "Get Answer",
        type="primary"
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Finding the best answer..."
            ):

                result = qa_system.answer(
                    question
                )

            if result and result.get("answer"):

                st.success(
                    "Answer found!"
                )

                st.subheader("💡 Answer")

                st.info(
                    result["answer"]
                )

                st.subheader(
                    "📖 Supporting Context"
                )

                st.write(
                    result.get(
                        "context",
                        ""
                    )
                )

                st.subheader("📌 Source")

                st.write(
                    result.get(
                        "title",
                        "Wikipedia"
                    )
                )

                if result.get("score") is not None:

                    st.caption(
                        f"Relevance score: "
                        f"{float(result['score']):.3f}"
                    )

            else:

                st.warning(
                    "No reliable answer was found."
                )


# ==========================================================
# KNOWLEDGE-BASED
# ==========================================================

elif page == "🧠 Knowledge-Based":

    st.header(
        "🧠 Knowledge-Based Question Answering"
    )

    st.write(
        """
        This module represents information as a
        **Subject → Relation → Object** knowledge triple.
        """
    )

    question = st.text_input(
        "Enter your knowledge question:",
        placeholder=(
            "Where is the University of Notre Dame located?"
        ),
        key="knowledge_input"
    )

    if st.button(
        "Get Knowledge Answer",
        type="primary"
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Searching knowledge..."
            ):

                result = qa_system.knowledge_answer(
                    question
                )

            if result.get("answer"):

                answer = result["answer"]

                if answer.startswith(
                    "No sufficiently"
                ):

                    st.warning(answer)

                else:

                    st.success(
                        "Knowledge Answer Found!"
                    )

                    st.subheader("💡 Answer")

                    st.info(answer)

                    subject = result.get(
                        "subject",
                        ""
                    )

                    relation = result.get(
                        "relation",
                        ""
                    )

                    object_value = result.get(
                        "object",
                        ""
                    )

                    if (
                        subject
                        and relation
                        and object_value
                    ):

                        st.subheader(
                            "🔗 Knowledge Triple"
                        )

                        col1, col2, col3 = (
                            st.columns(3)
                        )

                        with col1:

                            st.markdown(
                                "**SUBJECT**"
                            )

                            st.success(
                                subject
                            )

                        with col2:

                            st.markdown(
                                "**RELATION**"
                            )

                            st.warning(
                                relation
                            )

                        with col3:

                            st.markdown(
                                "**OBJECT**"
                            )

                            st.info(
                                object_value
                            )

                    if result.get("sentence"):

                        st.subheader(
                            "📖 Supporting Fact"
                        )

                        st.write(
                            result["sentence"]
                        )

                    if result.get("source"):

                        st.subheader(
                            "📌 Source"
                        )

                        st.write(
                            result["source"]
                        )

            else:

                st.warning(
                    "No relevant knowledge was found."
                )


# ==========================================================
# WIKIPEDIA EXPLORER
# ==========================================================

elif page == "🔎 Wikipedia Explorer":

    st.header("🔎 Wikipedia Explorer")

    st.write(
        "Search Wikipedia passages contained in SQuAD."
    )

    query = st.text_input(
        "Enter search query:",
        placeholder="University of Notre Dame"
    )

    number = st.slider(
        "Number of results",
        1,
        10,
        5
    )

    if st.button(
        "Search Wikipedia",
        type="primary"
    ):

        if not query.strip():

            st.warning(
                "Please enter a search query."
            )

        else:

            with st.spinner(
                "Searching..."
            ):

                results = qa_system.search_articles(
                    query,
                    top_k=number
                )

            if results:

                st.success(
                    f"Found {len(results)} result(s)."
                )

                for i, result in enumerate(
                    results,
                    1
                ):

                    with st.expander(
                        f"📚 {i}. {result['title']}"
                    ):

                        st.write(
                            result["context"]
                        )

                        st.caption(
                            "Relevance score: "
                            f"{result['score']:.3f}"
                        )

            else:

                st.warning(
                    "No relevant Wikipedia passages found."
                )


# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.caption(
    "Intelligent Wikipedia Question Answering Assistant "
    "| SQuAD v1.1 | NLP + Information Retrieval"
)