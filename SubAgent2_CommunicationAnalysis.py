import os
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI
import chromadb


# FILE PATHS
PROJECT_ROOT = Path(__file__).resolve().parent
CHROMA_DIR = PROJECT_ROOT / "ChromaEmbeddings"
COMMUNICATION_COLLECTION_NAME = "communication_kb"

SYNTHETIC_DATA_DIR = PROJECT_ROOT / "SyntheticEmployeeData"
COMMUNICATION_SUMMARY_FILE = SYNTHETIC_DATA_DIR / "communication_summary.txt"
# MAX_ACCEPTABLE_DISTANCE = 1 # This is to add a threshold to similarity of query and retrieved documents. Lower distance = Better match


class SubAgent2_CommunicationAnalysis:
    """
    SubAgent2_CommunicationAnalysis

    Responsibility:
    - Take a manager's communication-related query (meetings, channels, overload, Slack, email, etc.).
    - Retrieve relevant literature-based context from the 'communication_kb' Chroma collection.
    - Use an LLM to generate a practical communication-oriented response.
    - Always have access to an internal communication/workload summary for the synthetic team.
    - Never give purely generic responses without using the provided evidence/context.
    """

    def __init__(self) -> None:
        # Load environment variables
        load_dotenv()

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not found. Please set it in your .env file.\n"
            )

        # Default chat model can be overridden via .env
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-5-mini")

        # Embedding model can be overriden via .env [Should be same as the model used to create Chroma Embeddings]
        self.embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

        # Create OpenAI client
        self.client = OpenAI(api_key=api_key)

        # Connect to Chroma persistent store
        if not CHROMA_DIR.exists():
            raise RuntimeError(
                f"Chroma directory not found: {CHROMA_DIR}. "
                "Did you build the communication embeddings successfully?\n"
            )

        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.chroma_client.get_collection(COMMUNICATION_COLLECTION_NAME)

        # Set number of chunks to retrieve per query
        self.n_results = 7

        # Check that the summary file exists
        if not COMMUNICATION_SUMMARY_FILE.exists():
            raise RuntimeError(
                f"Communication summary file not found: {COMMUNICATION_SUMMARY_FILE}\n"
                "Please generate it by running:\n"
                "  python communication_data_analysis.py > SyntheticEmployeeData\\communication_summary.txt\n"
            )


    def _embed_query(self, text: str) -> List[float]:
        """
        Create an embedding for the manager's query using the same embedding
        model that was used for the knowledge base.
        """
        response = self.client.embeddings.create(
            model=self.embed_model,
            input=[text]
        )
        return response.data[0].embedding

    def _retrieve_context(self, manager_query: str, debug: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve top 7 relevant chunks from the communication_kb collection.

        Returns a list of dicts:
        [
          {
            "id": ...,
            "text": ...,
            "paper_title": ...,
            "paper_authors": ...,
            "year": ...,
            "chunk_index": ...,
            "distance": ...
          },
          ...
        ]
        """
        query_embedding = self._embed_query(manager_query)

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.n_results,
            include=["documents", "metadatas", "distances"]
        )

        # result structure:
        # {
        #   "ids": [[...]],
        #   "documents": [[...]],
        #   "metadatas": [[...]],
        #   "distances": [[...]],
        #   "embeddings": [[...]] (optional and not included here- only seen by LLM)
        # }

        ids_list = result.get("ids", [[]])[0] if result.get("ids") else []
        docs_list = result.get("documents", [[]])[0] if result.get("documents") else []
        metas_list = result.get("metadatas", [[]])[0] if result.get("metadatas") else []
        dists_list = result.get("distances", [[]])[0] if result.get("distances") else []

        context_items: List[Dict[str, Any]] = []

        for idx, (doc_id, doc_text, meta, dist) in enumerate(zip(ids_list, docs_list, metas_list, dists_list)):
            if not doc_text or not doc_text.strip():
                continue

            # Optional future distance filter (currently disabled)
            # if dist is not None and dist > MAX_ACCEPTABLE_DISTANCE:
            #     continue

            context_items.append(
                {
                    "id": doc_id,
                    "text": doc_text.strip(),
                    "paper_title": meta.get("paper_title"),
                    "paper_authors": meta.get("paper_authors"),
                    "year": meta.get("year"),
                    "chunk_index": meta.get("chunk_index"),
                    "distance": dist,
                }
            )

        if debug:
            print(f"\n[Communication Sub-Agent] Retrieved {len(context_items)} context chunks (requested {self.n_results}).")
            for i, item in enumerate(context_items):
                title = item.get("paper_title") or "Unknown title"
                year = item.get("year")
                dist = item.get("distance")
                print(
                    f"  {i+1}. {title} ({year})  - id={item['id']}  (distance={dist:.3f})")

        return context_items


    def _is_valid_communication_query(self, manager_query: str, debug: bool = True) -> bool:
        """
        Use the LLM as a simple classifier:
        - Return True only if the text describes a workplace communication / meetings issue (channels, frequency, overload, information flow, Slack, email, etc.).
        - Otherwise return False.
        """
        classifier_messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict classifier.\n"
                    "Your job is ONLY to decide whether the user's text is a meaningful description of a workplace communication or meeting-related issue.\n\n"
                    "Answer with exactly one word: 'yes' or 'no'.\n"
                    "- 'yes' = clearly about communication, meetings, channels, information flow message overload, or similar.\n"
                    "- 'no' = nonsense, too short, off-topic, or not about communication/meetings."
                ),
            },
            {
                "role": "user",
                "content": manager_query
            },
        ]

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=classifier_messages
        )
        answer = (resp.choices[0].message.content or "").strip().lower()

        is_valid = answer.startswith("y")

        if debug:
            print(
                f"\n[Communication Sub-Agent] Classifier decision for query: "
                f"{answer} -> valid = {is_valid}\n"
            )

        return is_valid

   

    def _build_system_prompt(self) -> str:
        """
        System prompt defining the role and constraints of this sub-agent.
        """
        return """
You are Sub-Agent Communication Analysis in a Leadership Mentoring System.

You specialize in:
- Workplace communication patterns (meetings, email, Slack/IM, channels).
- Information overload, too many meetings, poor channel choice, and async/sync balance.
- Helping managers improve communication structures, load, and clarity for their teams.

You receive:
1) A manager's description of a communication/meeting-related situation.
2) A set of evidence-based context chunks derived from workplace communication and digital collaboration literature.
3) An internal communication/workload analysis summary for a specific synthetic team of 5 employees:
   Alice, Bob, Sarah, John, and Martin.

Your task:
- Use ONLY the provided literature chunks and the internal communication/workload summary as your knowledge base.
- Generate a clear, practical response that helps the manager address communication patterns (e.g. overload, too many meetings, poor use of channels).
- Do NOT invent theories that are not supported by the provided context or summary.
- Do NOT rely on your own general world knowledge beyond what is in the context.

Important rules about employees and the internal summary:
- You ONLY have internal communication/workload data for: Alice, Bob, Sarah, John, and Martin.
- If the manager's question is about this team as a whole, you MAY use the internal summary in a general way.
- If the manager asks specifically about Alice, Bob, Sarah, John, or Martin, you MAY use their individual meeting, email, Slack, and task patterns from the summary.
- If the manager explicitly names a person who is NOT one of Alice, Bob, Sarah, John, or Martin, you MUST:
  - Say that the system has no internal communication/workload information for that employee name.
  - NOT provide any generic or alternative advice in that case.
- If neither the literature context nor the internal summary gives a clear basis for advice, you MUST say that
  the available knowledge is insufficient and avoid making up recommendations.

Output style:
- Provide a SHORT answer: 3 to 4 numbered steps only.
- Each step should be 1 to 3 short sentences.
- Focus on concrete actions the manager can take (what to change in meetings, channels, norms, or load).
- It is okay to refer to 'research' or 'evidence' in general terms, but do NOT cite technical details.
- Do NOT ask follow-up questions.
- Do NOT invite the manager to provide more details.
- Do NOT offer extra options like 'if you want, I can...'.
- You are simply giving a brief, evidence-based action plan.
"""
    def _load_communication_summary(self) -> str:
        """
        Load the pre-generated internal communication/workload summary from file.
        Handles common Windows encodings (UTF-8 and UTF-16).
        """
        # First try UTF-8 (including BOM if present)
        try:
            with open(COMMUNICATION_SUMMARY_FILE, "r", encoding="utf-8-sig") as f:
                return f.read().strip()
        except UnicodeDecodeError:
        # Fallback: try UTF-16 (typical for PowerShell redirection)
            with open(COMMUNICATION_SUMMARY_FILE, "r", encoding="utf-16") as f:
                return f.read().strip()


    def _build_user_prompt(
        self,
        manager_query: str,
        context_items: List[Dict[str, Any]],
        communication_summary: str,
    ) -> str:
        """
        Build the user-facing prompt that includes:
        - the manager's problem
        - retrieved literature context
        - the internal communication summary
        """
        # Build literature context block
        if not context_items:
            context_block = "No relevant context could be retrieved from the communication knowledge base."
        else:
            parts = []
            for idx, item in enumerate(context_items, start=1):
                title = item.get("paper_title") or "Unknown title"
                authors = item.get("paper_authors") or "Unknown authors"
                year = item.get("year") or "Unknown year"
                text = item.get("text") or ""

                parts.append(
                    f"--- Context {idx} ---\n"
                    f"Source: {title} ({authors}, {year})\n"
                    f"Content:\n{text}\n"
                )

            context_block = "\n".join(parts)

        summary_block = (
            communication_summary
            if communication_summary
            else "No internal communication/workload summary is currently available."
        )

        user_content = (
            "Manager's communication description:\n"
            f"\"\"\"{manager_query}\"\"\"\n\n"
            "Relevant evidence-based context from workplace communication and collaboration literature:\n"
            f"{context_block}\n\n"
            "Internal communication/workload analysis summary for your synthetic team "
            "(meetings, emails, Slack messages, tasks):\n"
            f"{summary_block}\n\n"
            "Using ONLY the ideas and guidance from the literature context and the internal summary above, "
            "provide a practical communication-focused action plan for the manager.\n"
            "Your answer MUST:\n"
            "- Be 3 to 4 numbered steps only.\n"
            "- Each step 1 to 3 short sentences.\n"
            "- Focus on clear, concrete actions or phrases the manager can use.\n"
            "- NOT ask any follow-up questions.\n"
            "- NOT invite the manager to provide more details.\n"
            "- NOT invent new theories that are not supported by the context or the internal summary.\n"
            "- If the question is about an employee name that is not Alice, Bob, Sarah, John, or Martin, you MUST say you have no internal information for that name and give no further advice.\n"
        )

        return user_content

  

    def run(self, manager_query: str, debug: bool = True) -> str:
        """
        Main entrypoint for the Communication Analysis sub-agent.

        Steps:
        1) Check if the query is related to workplace communication / meetings at all.
        2) Retrieve top 7 relevant communication chunks from Chroma.
        3) Load the internal communication/workload summary.
        4) Build a prompt with the manager_query, the literature context, and the summary.
        5) Ask the LLM to generate a communication-focused response.
        6) Return the response text as a string.
        """
        if debug:
            print("\n[Communication Sub-Agent] Received manager query:\n")
            print(manager_query)

        # Check if the query looks like a communication issue
        if not self._is_valid_communication_query(manager_query, debug=debug):
            if debug:
                print(
                    "[Communication Sub-Agent] Query is not a clear communication/meeting issue and so cannot be answered.\n"
                )
            return (
                "The Communication Analysis Sub-Agent could not detect a clear communication or meeting-related issue in your description."
            )

        # Retrieve context from Chroma
        context_items = self._retrieve_context(manager_query, debug=debug)

        # Load internal communication summary
        communication_summary = self._load_communication_summary()

        # If both literature context AND summary are empty, refuse
        if not context_items and not communication_summary:
            if debug:
                print(
                    "\n[Communication Sub-Agent] No context or internal summary available; refusing to answer without evidence-based knowledge.")
            return (
                "The Communication Analysis Sub-Agent could not retrieve any relevant evidence-based communication knowledge or internal team data for this query. "
                "It cannot safely provide a recommendation in this case."
            )

        # Build main prompt with context + summary
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(manager_query, context_items, communication_summary)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if debug:
            print(
                "\n[Communication Sub-Agent] Sending request to LLM for final communication-focused response...")

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )

        answer = response.choices[0].message.content

        if debug:
            print("\n[Communication Sub-Agent] Final response from Communication Analysis Sub-Agent:")
            print(answer)

        return answer


if __name__ == "__main__":
    # For Testing
    agent = SubAgent2_CommunicationAnalysis()

    print("Hi I am Sub-Agent Communication Analysis. Type your communication / meetings related description (or 'q' to quit).")
    while True:
        user_input = input("\nManager communication description:\n")
        if user_input.strip().lower() in {"q", "quit", "exit"}:
            break

        result = agent.run(user_input, debug=True)
        print("\n[Agent Output]...\n")
        print(result)
