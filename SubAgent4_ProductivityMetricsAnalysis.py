import os
import json
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI
import chromadb

from employee_configuration import Team


# FILE PATHS
PROJECT_ROOT = Path(__file__).resolve().parent
CHROMA_DIR = PROJECT_ROOT / "ChromaEmbeddings"
PRODUCTIVITY_COLLECTION_NAME = "productivity_kb"

SYNTHETIC_DATA_DIR = PROJECT_ROOT / "SyntheticEmployeeData"
PRODUCTIVITY_STATS_FILE = SYNTHETIC_DATA_DIR / "synthetic_productivity_overall_stats.json"

# MAX_ACCEPTABLE_DISTANCE = 1 # This is to add a threshold to similarity of query and retrieved documents. Lower distance = Better match

# Getting Employee Names
KNOWN_EMPLOYEES = [member["name"] for member in Team]
KNOWN_EMPLOYEE_SET = {name.lower() for name in KNOWN_EMPLOYEES}


class SubAgent4_ProductivityMetricsAnalysis:
    """
    SubAgent4_ProductivityMetricsAnalysis

    Responsibility:
    - Handle manager queries about: Daily / multi-day productivity patterns, Overload vs sustainable pace, Relationship between self-reported well-being and productivity, Which days or employees are at risk from a productivity perspective.
    - RAG over synthetic productivity evaluation records (Chroma: 'productivity_kb').
    - Synthetic overall productivity statistics (per employee, over multiple days).
    - Reject junk/off-topic queries.
    - Refuse to advise on employees that are not in the synthetic team configuration.
    - Never hallucinate productivity data when no context is retrieved.
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
                "Did you build the productivity embeddings successfully?\n"
            )

        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.chroma_client.get_collection(PRODUCTIVITY_COLLECTION_NAME)

        # Load synthetic overall productivity stats
        self.productivity_stats = self._load_productivity_stats()

        # Set number of chunks to retrieve per query
        self.n_results = 5



    def _load_productivity_stats(self) -> Dict[str, Any]:
        """
        Load synthetic overall productivity statistics from JSON.

        Expected structure (typical example):
        {
          "Alice": {
             "stretch_exceeds": ...,
             "below_baseline": ...,
             "total_days": [...],
             "low_motivation_days": [...],
             ...
          },
          "Bob": {...},
          ...
        }
        
        """
        if not PRODUCTIVITY_STATS_FILE.exists():
            raise RuntimeError(
                f"Productivity stats file not found: {PRODUCTIVITY_STATS_FILE}.\n"
                "Please generate it by running:python evaluation_of_productivity.py\n"
            )

        with open(PRODUCTIVITY_STATS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise RuntimeError(
                f"Expected a dict at top level in {PRODUCTIVITY_STATS_FILE}, "
                f"got {type(data)} instead."
            )

        return data

    
    
    def _embed_query(self, text: str) -> List[float]:
        """
        Create an embedding for the manager's query using the same embedding
        model that was used for the productivity knowledge base.
        """
        response = self.client.embeddings.create(
            model=self.embed_model,
            input=[text],
        )
        return response.data[0].embedding

    def _retrieve_context(self, manager_query: str, debug: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve top N relevant chunks from the productivity_kb collection.

        Returns a list of dicts:
        [
          {
            "id": ...,
            "text": ...,
            "day": ...,
            "employee_name": ...,
            "distance": ...,
          },
          ...
        ]
        """
        query_embedding = self._embed_query(manager_query)

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.n_results,
            include=["documents", "metadatas", "distances"],
        )

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
                    "day": meta.get("day"),
                    "employee_name": meta.get("employee_name"),
                    "distance": dist,
                }
            )

        if debug:
            print(
                f"\n[Productivity Metrics Analysis Sub-Agent] Retrieved {len(context_items)} context chunks "
                f"(requested {self.n_results})."
            )
            for i, item in enumerate(context_items):
                day = item.get("day") or "Unknown day"
                emp = item.get("employee_name") or "Unknown employee"
                dist = item.get("distance")
                print(f"  {i+1}. Day={day}, Employee={emp}  - id={item['id']}  (distance={dist:.3f})")

        return context_items



    def _is_valid_productivity_query(self, manager_query: str, debug: bool = True) -> bool:
        """
        Use the LLM as a simple classifier:
        - Return True only if the text is about productivity, workload, sustainable pace, daily performance, or relationships between well-being and output.
        - Otherwise return False.
        """
        classifier_messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict classifier.\n"
                    "Your job is ONLY to decide whether the user's text is a meaningful description of a productivity or workload question.\n\n"
                    "Answer with exactly one word: 'yes' or 'no'.\n"
                    "- 'yes' = clearly about productivity, workload, output per day, sustainable pace, overwork, or productivity trends.\n"
                    "- 'no' = nonsense, too short, off-topic, or not about productivity."
                ),
            },
            {
                "role": "user",
                "content": manager_query,
            },
        ]

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=classifier_messages,
        )
        answer = (resp.choices[0].message.content or "").strip().lower()

        is_valid = answer.startswith("y")

        if debug:
            print(
                f"\n[Productivity Metrics Analysis Sub-Agent] Classifier decision for query: "
                f"{answer} -> valid={is_valid}\n"
            )

        return is_valid

    def _detect_employee_scope_with_llm(self, manager_query: str, debug: bool = True) -> Dict[str, Any]:
        """
        Ask the LLM whether the query is about:
        - only known employees (Example here -Alice, Bob, Sarah, John, Martin)
        - only unknown employees
        - a mix
        - or no specific person

        Returns a dict:
        {
          "scope": "only_known" | "only_unknown" | "mixed" | "none",
          "known_employees": [list of known employee names mentioned]
        }
        """
        known_list_str = ", ".join(KNOWN_EMPLOYEES)

        system_content = (
            "You are a strict JSON-only classifier for employee name scope.\n"
            "You receive a manager's query about productivity or workload.\n"
            f"The only valid employee names you know about are: {known_list_str}.\n\n"
            "You must respond with a single JSON object only.\n"
            "The JSON MUST have exactly two fields:\n"
            '  - "scope": one of "none", "only_known", "only_unknown", or "mixed".\n'
            '  - "known_employees": a list of strings of any of the known names explicitly mentioned.\n\n'
            'Definitions:\n'
            '  - "none": no specific person name is mentioned at all.\n'
            '  - "only_known": any specific names mentioned are ONLY from the known set.\n'
            '  - "only_unknown": one or more person-like names are mentioned and NONE of them are from the known set.\n'
            '  - "mixed": at least one name is from the known set and at least one appears to be some other person.\n'
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": manager_query},
        ]

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            response_format={"type": "json_object"},
        )

        raw_content = resp.choices[0].message.content or "{}"

        if debug:
            print(f"[Productivity Metrics Analysis Sub-Agent] Employee scope raw JSON: {raw_content}")

        try:
            data = json.loads(raw_content)
            scope = data.get("scope", "none")
            known_emps = data.get("known_employees", [])
            if not isinstance(known_emps, list):
                known_emps = []
            # Filter to only actually known names
            known_emps = [
                name for name in known_emps if name in KNOWN_EMPLOYEES
            ]
            parsed = {"scope": scope, "known_employees": known_emps}
        except Exception:
            if debug:
                print("[Productivity Metrics Analysis Sub-Agent] Failed to parse employee scope JSON; defaulting to scope='none'.")
            parsed = {"scope": "none", "known_employees": []}

        if debug:
            print(
                f"[Productivity Metrics Analysis Sub-Agent] Employee scope parsed: {parsed}"
            )

        return parsed

    

    def _build_system_prompt(self) -> str:
        """
        System prompt defining the role and constraints of this sub-agent.
        """
        return """
You are Sub-Agent Productivity Metrics Analysis in a Leadership Mentoring System.

You specialize in:
- Interpreting daily and multi-day productivity metrics and self-reported survey data.
- Identifying overload vs sustainable workload patterns over several days.
- Highlighting risk signals (e.g., repeated low productivity, low well-being, high stress).
- Suggesting short, concrete manager actions to rebalance workload or adjust work patterns.

You receive:
1) A manager's description of a productivity or workload question.
2) Retrieved synthetic productivity evaluation records for specific days and employees.
3) An internal overall productivity statistics summary for the small synthetic team.

Your task:
- Use ONLY the provided evaluation records and the internal statistics summary.
- Generate a clear, practical, data-driven response for the manager.
- Do NOT invent extra data or statistics beyond what is implied by the context.
- You are NOT responsible for development plans or training recommendations.

Output style:
- Provide a SHORT answer: 3 to 4 numbered steps only.
- Each step should be 1 to 3 short sentences.
- Focus on concrete actions the manager can take (what to adjust, what to monitor, how to respond).
- Do NOT ask follow-up questions.
- Do NOT invite the manager to give more information.
- Do NOT offer options like "if you want, I can...".
"""

    def _build_stats_block(self, known_emps: List[str]) -> str:
        """
        Convert the loaded productivity stats JSON into a readable block.

        If known_emps is non-empty, focus stats on those employees.
        Otherwise, provide a short overview for the whole team.
        """
        lines: List[str] = []

        if known_emps:
            lines.append(
                "Internal productivity statistics for the relevant employees:\n"
            )
            target_names = known_emps
        else:
            lines.append(
                "Internal productivity statistics overview for the synthetic team:\n"
            )
            target_names = list(self.productivity_stats.keys())

        for name in target_names:
            entry = self.productivity_stats.get(name)
            if not entry:
                continue

            lines.append(f"- {name}:")

            # We don't assume exact keys; we just dump the main fields in a readable way.
            for key, value in entry.items():
                # Turn lists / dicts into compact strings to keep it short
                if isinstance(value, list):
                    value_str = ", ".join(str(v) for v in value)
                elif isinstance(value, dict):
                    value_str = "; ".join(
                        f"{k}={v}" for k, v in value.items()
                    )
                else:
                    value_str = str(value)
                lines.append(f"  {key}: {value_str}")

            lines.append("")

        return "\n".join(lines)

    def _build_user_prompt(self,manager_query: str,context_items: List[Dict[str, Any]],stats_block: str) -> str:
        """
        Build the user-facing prompt that includes:
        - the manager's question
        - retrieved productivity evaluation records
        - internal productivity statistics (aggregated)
        """
        if not context_items:
            context_block = (
                "No specific daily productivity records could be retrieved from "
                "the productivity knowledge base for this query."
            )
        else:
            parts = []
            for idx, item in enumerate(context_items, start=1):
                day = item.get("day") or "Unknown day"
                emp = item.get("employee_name") or "Unknown employee"
                text = item.get("text") or ""

                parts.append(
                    f"--- Productivity Context {idx} ---\n"
                    f"Day: {day}, Employee: {emp}\n"
                    f"Content:\n{text}\n"
                )

            context_block = "\n".join(parts)

        user_content = (
            "Manager's productivity / workload description:\n"
            f"\"\"\"{manager_query}\"\"\"\n\n"
            "Relevant synthetic productivity evaluation records:\n"
            f"{context_block}\n\n"
            "Internal aggregated productivity statistics:\n"
            f"{stats_block}\n\n"
            "Using ONLY this data, provide a practical, short analysis for the manager.\n"
            "Your answer MUST:\n"
            "- Be 3 to 4 numbered steps only.\n"
            "- Each step 1 to 3 short sentences.\n"
            "- Be clearly data-driven from the provided context and statistics.\n"
            "- NOT ask any follow-up questions.\n"
            "- NOT invite the manager to provide more details.\n"
            "- NOT invent new metrics or days that are not supported by the data.\n"
        )

        return user_content


    def run(self, manager_query: str, debug: bool = True) -> str:
        """
        Main entrypoint for the Productivity Metrics Analysis Sub-Agent.

        Steps:
        1) Check if the query is related to productivity/workload at all.
        2) Use LLM to detect whether it refers to specific employees and whether those employees are in the known synthetic team.
        3) If the query is only about unknown employees (or mixed known+unknown), refuse to provide person-specific advice.
        4) Retrieve top N relevant productivity evaluation chunks from Chroma.
        5) Build a prompt with the manager_query, retrieved context, and internal stats.
        6) Ask the LLM to generate a short, data-driven response.
        7) Return the response text as a string.
        """
        if debug:
            print("\n[Productivity Metrics Analysis Sub-Agent] Received manager query:\n")
            print(manager_query)

        # Check if the query looks like a productivity/workload question
        if not self._is_valid_productivity_query(manager_query, debug=debug):
            if debug:
                print(
                    "[Productivity Metrics Analysis Sub-Agent] Query is not a clear productivity or "
                    "workload question and so cannot be answered.\n"
                )
            return ("The Productivity Metrics Sub-Agent could not detect a clear productivity or workload question in your description.")

        # Detect employee scope via LLM
        scope_info = self._detect_employee_scope_with_llm(manager_query, debug=debug)
        scope = scope_info.get("scope", "none")
        known_emps = scope_info.get("known_employees", [])

        # If the query is clearly about only unknown employees or mixed, refuse
        if scope in {"only_unknown", "mixed"}:
            if debug:
                print("[Productivity Metrics Analysis Sub-Agent] Query is about specific employees outside the known synthetic team.\n")
            return (
                "The Productivity Metrics Analysis Sub-Agent can only provide suggestions for "
                "the members of the team configured.\n"
                "It cannot safely provide person-specific productivity guidance for "
                "other named employees."
            )

        # Retrieve productivity evaluation context from Chroma
        context_items = self._retrieve_context(manager_query, debug=debug)

        # If nothing is retrieved, refuse to hallucinate an answer
        if not context_items:
            if debug:
                print("\n[Productivity Metrics Analysis Sub-Agent] No context retrieved; refusing to answer without evidence-based productivity data.")
            return (
                "The Productivity Metrics Analysis Sub-Agent could not retrieve any relevant "
                "productivity evaluation data for this query. It cannot safely provide "
                "a recommendation in this case."
            )

        # Build internal stats block focused on known employees
        stats_block = self._build_stats_block(known_emps)

        # Build main prompt
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            manager_query, context_items, stats_block
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if debug:
            print("\n[Productivity Metrics Analysis Sub-Agent] Sending request to LLM for final productivity-focused response.")

        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )

        answer = response.choices[0].message.content

        if debug:
            print(
                "\n[Productivity Metrics Analysis Sub-Agent] Final response from Productivity Metrics Analysis Sub-Agent:"
            )
            print(answer)

        return answer


if __name__ == "__main__":
    # For Testing
    agent = SubAgent4_ProductivityMetricsAnalysis()

    print("Hi I am Sub-Agent Productivity Metrics Analysis. Type your productivity / workload related question (or 'q' to quit).")
    while True:
        user_input = input("\nManager productivity / workload problem description for your team: \n")
        if user_input.strip().lower() in {"q", "quit", "exit"}:
            break

        result = agent.run(user_input, debug=True)
        print("\n[Agent Output]...\n")
        print(result)
