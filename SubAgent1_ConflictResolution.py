import os
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI
import chromadb


# FILE PATHS
PROJECT_ROOT = Path(__file__).resolve().parent
CHROMA_DIR = PROJECT_ROOT / "ChromaEmbeddings"
CONFLICT_COLLECTION_NAME = "conflict_kb"
# MAX_ACCEPTABLE_DISTANCE = 1 # This is to add a threshold to similarity of query and retrieved documents. Lower distance = Better match


class SubAgent1_ConflictResolution:
    """
    SubAgent1_ConflictResolution

    Responsibility:
    - Take a manager's conflict-related query.
    - Retrieve relevant literature-based context from the 'conflict_kb' Chroma collection.
    - Use an LLM to generate a practical mediation-oriented response.
    - Weave in evidence and ideas from the retrieved chunks.
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
                "Did you build the conflict embeddings successfully?\n"
            )

        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.chroma_client.get_collection(CONFLICT_COLLECTION_NAME)

        # Set number of chunks to retrieve per query
        self.n_results = 7

   
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
        Retrieve top 7 relevant chunks from the conflict_kb collection.

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
            print(f"\n[Conflict Sub-Agent] Retrieved {len(context_items)} context chunks (requested {self.n_results}).")
            for i, item in enumerate(context_items):
                title = item.get("paper_title") or "Unknown title"
                year = item.get("year")
                dist = item.get("distance")
                print(f"  {i+1}. {title} ({year})  - id={item['id']}  (distance={dist:.3f})")

        return context_items

    def _build_system_prompt(self) -> str:
        """
        System prompt defining the role and constraints of this sub-agent.
        """
        return """
You are Sub-Agent Conflict Resolution in a Leadership Mentoring System.

You specialize in:
- Workplace conflicts between colleagues, teams, or managers and employees.
- Mediation and conflict de-escalation.
- Helping managers run constructive conversations and follow-up actions.

You receive:
1) A manager's conflict description.
2) A set of evidence-based context chunks derived from papers, articles and professional reports on workplace conflict and mediation.

Your task:
- Use ONLY the provided context chunks as your knowledge base.
- Generate a clear, practical response that helps the manager mediate the conflict.
- Do not invent theories that are not supported by the context.
- Do NOT rely on your own general world knowledge.
- If you cannot find enough guidance in the context, clearly say that the knowledge is insufficient.

Output style: 
- Provide a SHORT answer: 3 to 4 numbered steps only.
- Each step should be 1 to 3 short sentences.
- Focus on concrete actions the manager can take (what to say/do).
- It is okay to refer to 'research' or 'evidence' in general terms, but do NOT cite chunk IDs or technical details.
- Do NOT ask follow-up questions.
- Do NOT invite the manager to give more information.
- Do NOT offer extra options like 'if you want, I can...'.
- You are simply giving a brief, evidence-based action plan.

Very Important:
- If the context is clearly insufficient or irrelevant, explicitly say so and give a cautious,
  high-level recommendation rather than pretending you have strong evidence.
- Please keep your response short and do not generate huge paragraphs of text.
"""

    def _build_user_prompt(self, manager_query: str, context_items: List[Dict[str, Any]]) -> str:
        """
        Build the user-facing prompt that includes the manager's problem and
        the retrieved context in a readable way.
        """
        # Build a readable context block
        if not context_items:
            context_block = "No relevant context could be retrieved from the knowledge base."
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

        user_content = (
            "Manager's conflict description:\n"
            f"\"\"\"{manager_query}\"\"\"\n\n"
            "Relevant evidence-based context from workplace conflict literature:\n"
            f"{context_block}\n\n"
            "Using ONLY the ideas and guidance from this context, provide a practical mediation plan for the manager.\n"
             "Your answer MUST:\n"
            "- Be 3 to 4 numbered steps only.\n"
            "- Each step 1 to 3 short sentences.\n"
            "- Focus on clear, concrete actions or phrases the manager can use.\n"
            "- NOT ask any follow-up questions.\n"
            "- NOT invite the manager to provide more details.\n"
            "- NOT invent new theories that are not supported by the context.\n"
        )

        return user_content
    

    def _is_valid_conflict_query(self, manager_query: str, debug: bool = True) -> bool:
        """
        Use the LLM as a simple classifier:
        - Return True only if the text describes a workplace conflict/tension between people that might require mediation.
        - Otherwise return False.
        """
        classifier_messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict classifier.\n "
                    "Your job is ONLY to decide whether the user's text is a meaningful description of a workplace interpersonal conflict or tension.\n\n"
                    "Answer with exactly one word: 'yes' or 'no'.\n"
                    "- 'yes' = there is clearly a workplace conflict/tension between people.\n"
                    "- 'no' = text is nonsense, too short, off-topic, or not about conflict.\n"
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
            print(f"\n[Conflict Sub-Agent] Classifier decision for query: {answer} -> valid = {is_valid}\n")

        return is_valid



    def run(self, manager_query: str, debug: bool = True) -> str:
        """
        Main entrypoint for the Conflict Resolution sub-agent.

        Steps:
        1) Checks if the query by the manager is related to workplace conflict at all.
        2) Retrieve top 7 relevant conflict resolution chunks from Chroma.
        3) Build a prompt with the manager_query and the retrieved context.
        4) Ask the LLM to generate a mediation-focused response.
        5) Return the response text as a string.
        """
        if debug:
            print("\n[Conflict Sub-Agent] Received manager query:\n")
            print(manager_query)

        # Check if the conflict query looks like a workplace conflict description
        if not self._is_valid_conflict_query(manager_query, debug=debug):
            if debug:
                print("[Conflict Sub-Agent] Query is not a clear workplace conflict and so cannot be answered.\n")
            return (
                "The Conflict Resolution Sub-Agent could not detect a clear workplace conflict in your description."
            
            )
        

        # Retrieve context from Chroma
        context_items = self._retrieve_context(manager_query, debug=debug)

         # If no context at all, No answer from Generic LLM knowledge
        if not context_items:
            if debug:
                print("\n[Conflict Sub-Agent] No context retrieved; refusing to answer without evidence-based knowledge.")
            return (
                "The Conflict Resolution Sub-Agent could not retrieve any relevant evidence-based knowledge from its conflict resolution sources for this query. "
                "It cannot safely provide a recommendation in this case."
            )
    
        # Build main prompt with context
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(manager_query, context_items)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if debug:
            print("\n[Conflict Sub-Agent] Sending request to LLM for final mediated response...")

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )

        answer = response.choices[0].message.content

        if debug:
            print("\n[Conflict Sub-Agent] Final response from Conflict Resolution Sub-Agent:")
            print(answer)

        return answer


if __name__ == "__main__":
    # For Testing
    agent = SubAgent1_ConflictResolution()

    print("Hi I am Sub-Agent Conflict Resolution. Type your conflict related problem description (or 'q' to quit).")
    while True:
        user_input = input("\nManager conflict description: \n")
        if user_input.strip().lower() in {"q", "quit", "exit"}:
            break

        result = agent.run(user_input, debug=True)
        print("\n[Agent Output]...\n")
        print(result)


# OPTIONAL LLM GENRIC ANSWER IN CASE OF NO CONTEXTS
"""
        # If no context at all, fall back to a cautious direct answer
        if not context_items:
            if debug:
                print("[Conflict Sub-Agent] No context retrieved; falling back to cautious generic guidance.")
            fallback_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a workplace conflict resolution assistant. "
                        "The knowledge base is currently empty or unavailable. "
                        "Provide cautious, high-level guidance, and explicitly state that "
                        "you do not have access to the normal evidence base."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"The manager describes this conflict:\n\"\"\"{manager_query}\"\"\"\n\n"
                        "Give a careful, general suggestion on how they might begin to address it."
                    ),
                },
            ]
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=fallback_messages
            )
            answer = response.choices[0].message.content
            return answer
        """
