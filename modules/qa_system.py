import json
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SQuADQA:

    def __init__(self, train_path, dev_path):

        self.contexts = []
        self.qa_pairs = []

        # Load both SQuAD files
        self._load_dataset(train_path)
        self._load_dataset(dev_path)

        # Remove duplicate contexts
        seen = set()
        unique_contexts = []

        for item in self.contexts:

            key = (
                item["title"],
                item["context"]
            )

            if key not in seen:

                seen.add(key)

                unique_contexts.append(item)

        self.contexts = unique_contexts

        # --------------------------------------------------
        # CONTEXT SEARCH INDEX
        # --------------------------------------------------

        self.documents = [
            item["title"].replace("_", " ")
            + " "
            + item["context"]
            for item in self.contexts
        ]

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=50000
        )

        self.document_matrix = (
            self.vectorizer.fit_transform(
                self.documents
            )
        )

        # --------------------------------------------------
        # SQUAD QA INDEX
        # --------------------------------------------------

        self._load_qa_pairs(train_path)
        self._load_qa_pairs(dev_path)

        self.qa_questions = [
            item["question"]
            for item in self.qa_pairs
        ]

        self.qa_vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=30000
        )

        if self.qa_questions:

            self.qa_matrix = (
                self.qa_vectorizer.fit_transform(
                    self.qa_questions
                )
            )

        else:

            self.qa_matrix = None

        # Knowledge facts are generated only when needed
        self.knowledge_facts = None

    # ======================================================
    # LOAD DATASET
    # ======================================================

    def _load_dataset(self, path):

        with open(
            Path(path),
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        for article in data.get("data", []):

            title = article.get(
                "title",
                "Wikipedia"
            )

            for paragraph in article.get(
                "paragraphs",
                []
            ):

                context = paragraph.get(
                    "context",
                    ""
                ).strip()

                if context:

                    self.contexts.append({
                        "title": title,
                        "context": context
                    })

    # ======================================================
    # LOAD QA PAIRS
    # ======================================================

    def _load_qa_pairs(self, path):

        with open(
            Path(path),
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        for article in data.get("data", []):

            title = article.get(
                "title",
                "Wikipedia"
            )

            for paragraph in article.get(
                "paragraphs",
                []
            ):

                context = paragraph.get(
                    "context",
                    ""
                ).strip()

                for qa in paragraph.get(
                    "qas",
                    []
                ):

                    question = qa.get(
                        "question",
                        ""
                    ).strip()

                    answers = qa.get(
                        "answers",
                        []
                    )

                    if not question:
                        continue

                    if not answers:
                        continue

                    answer = answers[0].get(
                        "text",
                        ""
                    ).strip()

                    if not answer:
                        continue

                    self.qa_pairs.append({
                        "question": question,
                        "answer": answer,
                        "context": context,
                        "title": title
                    })

    # ======================================================
    # TEXT HELPERS
    # ======================================================

    def _words(self, text):

        return set(
            re.findall(
                r"\b[a-zA-Z]{2,}\b",
                text.lower().replace("_", " ")
            )
        )

    def _important_words(self, text):

        stopwords = {
            "what",
            "where",
            "when",
            "which",
            "who",
            "whom",
            "whose",
            "why",
            "how",
            "does",
            "did",
            "do",
            "is",
            "are",
            "was",
            "were",
            "the",
            "a",
            "an",
            "of",
            "to",
            "in",
            "on",
            "for",
            "and",
            "or",
            "with",
            "from",
            "by",
            "located",
            "tell",
            "me",
            "about"
        }

        return {
            word
            for word in self._words(text)
            if word not in stopwords
        }

    # ======================================================
    # QUESTION TYPE
    # ======================================================

    def _question_type(self, question):

        q = question.lower().strip()

        if q.startswith("where"):
            return "where"

        if q.startswith("when"):
            return "when"

        if q.startswith(("who", "whom")):
            return "who"

        if q.startswith(
            ("how many", "how much")
        ):
            return "number"

        if q.startswith("why"):
            return "why"

        return "general"

    # ======================================================
    # FIND RELEVANT CONTEXTS
    # ======================================================

    def _find_relevant_contexts(
        self,
        question,
        top_k=20
    ):

        question_vector = (
            self.vectorizer.transform(
                [question]
            )
        )

        similarities = cosine_similarity(
            question_vector,
            self.document_matrix
        )[0]

        question_words = (
            self._important_words(
                question
            )
        )

        ranked = []

        for index, similarity in enumerate(
            similarities
        ):

            item = self.contexts[index]

            title_words = (
                self._important_words(
                    item["title"]
                )
            )

            title_overlap = len(
                question_words.intersection(
                    title_words
                )
            )

            score = (
                float(similarity)
                + title_overlap * 0.35
            )

            ranked.append(
                (
                    score,
                    item,
                    title_overlap
                )
            )

        ranked.sort(
            key=lambda x: x[0],
            reverse=True
        )

        return ranked[:top_k]

    # ======================================================
    # LOCATION EXTRACTION
    # ======================================================

    def _extract_location(
        self,
        context
    ):

        sentences = re.split(
            r"(?<=[.!?])\s+",
            context
        )

        patterns = [

            r"\blocated\s+adjacent\s+to\s+"
            r"([A-Z][A-Za-z .'-]*(?:,\s*[A-Z][A-Za-z .'-]*)*)",

            r"\blocated\s+in\s+"
            r"([A-Z][A-Za-z .'-]*(?:,\s*[A-Z][A-Za-z .'-]*)*)",

            r"\blocated\s+at\s+"
            r"([A-Z][A-Za-z .'-]*(?:,\s*[A-Z][A-Za-z .'-]*)*)",

            r"\blocated\s+near\s+"
            r"([A-Z][A-Za-z .'-]*(?:,\s*[A-Z][A-Za-z .'-]*)*)",

            r"\bsituated\s+in\s+"
            r"([A-Z][A-Za-z .'-]*(?:,\s*[A-Z][A-Za-z .'-]*)*)"
        ]

        for sentence in sentences:

            for pattern in patterns:

                match = re.search(
                    pattern,
                    sentence,
                    flags=re.IGNORECASE
                )

                if not match:
                    continue

                answer = match.group(1).strip()

                answer = re.split(
                    r",\s*(?:in|and|where|which)\b",
                    answer,
                    maxsplit=1,
                    flags=re.IGNORECASE
                )[0]

                answer = answer.strip(
                    " .,;:"
                )

                parts = [
                    part.strip()
                    for part in answer.split(",")
                ]

                if len(parts) >= 2:

                    answer = (
                        parts[0]
                        + ", "
                        + parts[1]
                    )

                return answer, sentence

        return None, None

    # ======================================================
    # YEAR EXTRACTION
    # ======================================================

    def _extract_year(
        self,
        context
    ):

        sentences = re.split(
            r"(?<=[.!?])\s+",
            context
        )

        for sentence in sentences:

            match = re.search(
                r"\b(?:18|19|20)\d{2}\b",
                sentence
            )

            if match:

                return (
                    match.group(0),
                    sentence
                )

        return None, None

    # ======================================================
    # FIND BEST SQUAD QUESTION INSIDE ONE ARTICLE
    # ======================================================

    def _best_article_qa(
        self,
        question,
        title,
        threshold=0.25
    ):

        pairs = [
            item
            for item in self.qa_pairs
            if item["title"] == title
        ]

        if not pairs:
            return None

        questions = [
            item["question"]
            for item in pairs
        ]

        local_vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )

        matrix = (
            local_vectorizer.fit_transform(
                questions
            )
        )

        question_vector = (
            local_vectorizer.transform(
                [question]
            )
        )

        similarities = cosine_similarity(
            question_vector,
            matrix
        )[0]

        ranked = similarities.argsort()[::-1]

        question_words = (
            self._important_words(
                question
            )
        )

        best = None

        for index in ranked[:10]:

            score = float(
                similarities[index]
            )

            if score < threshold:
                continue

            candidate = pairs[index]

            candidate_words = (
                self._important_words(
                    candidate["question"]
                )
            )

            overlap = len(
                question_words.intersection(
                    candidate_words
                )
            )

            adjusted_score = (
                score
                + min(
                    overlap * 0.03,
                    0.15
                )
            )

            if (
                best is None
                or adjusted_score
                > best["score"]
            ):

                best = {
                    "answer":
                        candidate["answer"],
                    "context":
                        candidate["context"],
                    "title":
                        candidate["title"],
                    "score":
                        adjusted_score
                }

        return best

    # ======================================================
    # MAIN QA ANSWER
    # ======================================================

    def answer(self, question):

        question_type = (
            self._question_type(
                question
            )
        )

        candidates = (
            self._find_relevant_contexts(
                question,
                top_k=20
            )
        )

        if not candidates:

            return {
                "answer":
                    "I could not find a relevant answer.",
                "context": "",
                "title": "Wikipedia",
                "score": 0
            }

        # --------------------------------------------------
        # WHERE
        # --------------------------------------------------

        if question_type == "where":

            for (
                article_score,
                item,
                title_overlap
            ) in candidates:

                if title_overlap <= 0:
                    continue

                location, sentence = (
                    self._extract_location(
                        item["context"]
                    )
                )

                if location:

                    return {
                        "answer": location,
                        "context": sentence,
                        "title": item["title"],
                        "score": min(
                            1.0,
                            article_score
                        )
                    }

            for (
                article_score,
                item,
                title_overlap
            ) in candidates:

                location, sentence = (
                    self._extract_location(
                        item["context"]
                    )
                )

                if location:

                    return {
                        "answer": location,
                        "context": sentence,
                        "title": item["title"],
                        "score": min(
                            1.0,
                            article_score
                        )
                    }

        # --------------------------------------------------
        # WHEN
        # --------------------------------------------------

        if question_type == "when":

            for (
                article_score,
                item,
                title_overlap
            ) in candidates:

                if title_overlap <= 0:
                    continue

                year, sentence = (
                    self._extract_year(
                        item["context"]
                    )
                )

                if year:

                    return {
                        "answer": year,
                        "context": sentence,
                        "title": item["title"],
                        "score": min(
                            1.0,
                            article_score
                        )
                    }

        # --------------------------------------------------
        # WHO
        # --------------------------------------------------

        if question_type == "who":

            relevant_titles = []

            for (
                article_score,
                item,
                title_overlap
            ) in candidates:

                if title_overlap > 0:

                    if (
                        item["title"]
                        not in relevant_titles
                    ):

                        relevant_titles.append(
                            item["title"]
                        )

            best = None

            for title in relevant_titles:

                candidate = (
                    self._best_article_qa(
                        question,
                        title,
                        threshold=0.30
                    )
                )

                if candidate is None:
                    continue

                if (
                    best is None
                    or candidate["score"]
                    > best["score"]
                ):

                    best = candidate

            if best is not None:

                return best

        # --------------------------------------------------
        # GENERAL
        # --------------------------------------------------

        relevant_titles = []

        for (
            article_score,
            item,
            title_overlap
        ) in candidates:

            if title_overlap > 0:

                if (
                    item["title"]
                    not in relevant_titles
                ):

                    relevant_titles.append(
                        item["title"]
                    )

        best = None

        for title in relevant_titles:

            candidate = (
                self._best_article_qa(
                    question,
                    title,
                    threshold=0.35
                )
            )

            if candidate is None:
                continue

            if (
                best is None
                or candidate["score"]
                > best["score"]
            ):

                best = candidate

        if best is not None:

            return best

        # --------------------------------------------------
        # FALLBACK
        # --------------------------------------------------

        best_score, best_item, _ = (
            candidates[0]
        )

        return {
            "answer":
                "I could not find a reliable answer.",
            "context":
                best_item["context"],
            "title":
                best_item["title"],
            "score":
                float(best_score)
        }

    # ======================================================
    # WIKIPEDIA SEARCH
    # ======================================================

    def search_articles(
        self,
        query,
        top_k=10
    ):

        candidates = (
            self._find_relevant_contexts(
                query,
                top_k=top_k
            )
        )

        results = []

        for (
            score,
            item,
            title_overlap
        ) in candidates:

            results.append({
                "title":
                    item["title"],
                "context":
                    item["context"],
                "score":
                    float(score)
            })

        return results

    # ======================================================
    # KNOWLEDGE GRAPH
    # ======================================================

    def _build_knowledge_graph(self):

        facts = []

        patterns = [

            (
                r"(.+?)\s+is located in\s+(.+?)(?:\.|$)",
                "located in"
            ),

            (
                r"(.+?)\s+is located at\s+(.+?)(?:\.|$)",
                "located at"
            ),

            (
                r"(.+?)\s+was founded by\s+(.+?)(?:\.|$)",
                "founded by"
            ),

            (
                r"(.+?)\s+was founded in\s+(.+?)(?:\.|$)",
                "founded in"
            ),

            (
                r"(.+?)\s+was established in\s+(.+?)(?:\.|$)",
                "established in"
            ),

            (
                r"(.+?)\s+was born in\s+(.+?)(?:\.|$)",
                "born in"
            ),

            (
                r"(.+?)\s+is a\s+(.+?)(?:\.|$)",
                "is a"
            ),

            (
                r"(.+?)\s+is an\s+(.+?)(?:\.|$)",
                "is an"
            )
        ]

        for item in self.contexts:

            sentences = re.split(
                r"(?<=[.!?])\s+",
                item["context"]
            )

            for sentence in sentences:

                sentence = sentence.strip()

                if not sentence:
                    continue

                for pattern, relation in patterns:

                    match = re.search(
                        pattern,
                        sentence,
                        flags=re.IGNORECASE
                    )

                    if not match:
                        continue

                    subject = (
                        match.group(1).strip()
                    )

                    object_value = (
                        match.group(2).strip()
                    )

                    if (
                        len(subject) <= 150
                        and len(object_value) <= 200
                    ):

                        facts.append({
                            "subject": subject,
                            "relation": relation,
                            "object": object_value,
                            "title":
                                item["title"],
                            "sentence":
                                sentence
                        })

        return facts

    # ======================================================
    # KNOWLEDGE ANSWER
    # ======================================================

    def knowledge_answer(
        self,
        question
    ):

        q = question.lower().strip()

        # Fast WHERE handling
        if q.startswith("where"):

            candidates = (
                self._find_relevant_contexts(
                    question,
                    top_k=15
                )
            )

            for (
                score,
                item,
                title_overlap
            ) in candidates:

                location, sentence = (
                    self._extract_location(
                        item["context"]
                    )
                )

                if location:

                    return {
                        "answer": location,
                        "subject":
                            item["title"].replace(
                                "_",
                                " "
                            ),
                        "relation":
                            "located in",
                        "object":
                            location,
                        "source":
                            item["title"],
                        "sentence":
                            sentence,
                        "score":
                            float(score)
                    }

            return {
                "answer":
                    "No location fact was found.",
                "subject": "",
                "relation": "",
                "object": "",
                "source": ""
            }

        # Fast WHEN handling
        if q.startswith("when"):

            candidates = (
                self._find_relevant_contexts(
                    question,
                    top_k=15
                )
            )

            for (
                score,
                item,
                title_overlap
            ) in candidates:

                year, sentence = (
                    self._extract_year(
                        item["context"]
                    )
                )

                if year:

                    return {
                        "answer": year,
                        "subject":
                            item["title"].replace(
                                "_",
                                " "
                            ),
                        "relation":
                            "established in",
                        "object":
                            year,
                        "source":
                            item["title"],
                        "sentence":
                            sentence,
                        "score":
                            float(score)
                    }

            return {
                "answer":
                    "No date fact was found.",
                "subject": "",
                "relation": "",
                "object": "",
                "source": ""
            }

        # Build graph only for other knowledge questions
        if self.knowledge_facts is None:

            self.knowledge_facts = (
                self._build_knowledge_graph()
            )

        requested_relation = None

        if "founded" in q:
            requested_relation = "founded by"

        elif "established" in q:
            requested_relation = "established in"

        question_words = (
            self._important_words(
                question
            )
        )

        candidates = []

        for fact in self.knowledge_facts:

            fact_words = self._words(
                fact["subject"]
                + " "
                + fact["relation"]
                + " "
                + fact["object"]
            )

            overlap = len(
                question_words.intersection(
                    fact_words
                )
            )

            score = overlap

            if (
                requested_relation
                and fact["relation"]
                == requested_relation
            ):
                score += 5

            candidates.append(
                (
                    score,
                    fact
                )
            )

        candidates.sort(
            key=lambda x: x[0],
            reverse=True
        )

        if (
            not candidates
            or candidates[0][0] == 0
        ):

            return {
                "answer":
                    "No sufficiently relevant knowledge fact was found.",
                "subject": "",
                "relation": "",
                "object": "",
                "source": ""
            }

        best = candidates[0][1]

        return {
            "answer":
                best["object"],
            "subject":
                best["subject"],
            "relation":
                best["relation"],
            "object":
                best["object"],
            "source":
                best["title"],
            "sentence":
                best["sentence"],
            "score":
                candidates[0][0]
        }