**Final Project**

**"FAQ Chatbot Using Simple RAG with Free Hugging Face Models or OpenAI"**

---

**Objective**

Create a **basic chatbot** capable of answering frequently asked questions (FAQ) about a small domain, such as tourism, coffee, or movies, by combining:

* **RAG (Retrieval-Augmented Generation)** to search for information in local documents.
* An **open-source** model from Hugging Face, such as `mistralai/Mistral-7B-Instruct` via `text-generation-inference` or `google/flan-t5-base`. Or an OpenAI model.
* Simple indexing with **FAISS**.
* Do not use paid APIs; use only local models or free Hugging Face endpoints.

---

**Project Components**

1. **Local Document Corpus (chosen by the students)**

   * A set of `.txt` files containing information about the chosen domain.
   * Examples: "RJ Travel Guide", "Curiosities About Coffee", "Explanation of Famous Movies".

2. **Basic RAG Pipeline**

   * **Step 1: Indexing**

     * Use FAISS to index embeddings, using `sentence-transformers/all-MiniLM-L6-v2`.

   * **Step 2: Retrieval**

     * Search for the most relevant excerpts from the corpus based on the user's question.

   * **Step 3: Generation**

     * Use a model such as `flan-t5-base` to answer using the retrieved excerpt.

3. **Simple Interface**

   * A terminal script or notebook.
   * Basic interface: the user types a question, and the chatbot responds.

4. **Bonus (optional)**

   * Implement using **LangChain**, which also works with free models.
   * Save the history of questions and answers.