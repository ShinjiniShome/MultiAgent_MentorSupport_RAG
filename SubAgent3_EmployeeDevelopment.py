import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from openai import OpenAI
import chromadb

from employee_configuration import Team
import json

# FILE PATHS
PROJECT_ROOT = Path(__file__).resolve().parent
CHROMA_DIR = PROJECT_ROOT / "ChromaEmbeddings"
EMPDEV_COLLECTION_NAME = "employeedev_kb"

SYNTHETIC_DATA_DIR = PROJECT_ROOT / "SyntheticEmployeeData"
EMPLOYEE_DATA_FILE = SYNTHETIC_DATA_DIR / "synthetic_employeeData.json"
AVAILABLE_TRAININGS_FILE = SYNTHETIC_DATA_DIR / "available_trainings.json"

# MAX_ACCEPTABLE_DISTANCE = 1 # This is to add a threshold to similarity of query and retrieved documents. Lower distance = Better match

# Getting Employee Names
KNOWN_EMPLOYEES = [member["name"] for member in Team]
KNOWN_EMPLOYEE_SET = {name.lower() for name in KNOWN_EMPLOYEES}


class SubAgent3_EmployeeDevelopment:
    """
    SubAgent3_EmployeeDevelopment

    Responsibility:
    - Handle manager queries about: Employee development and growth, Strengths and weaknesses, Performance and skill gaps, Burnout or risk-related development support.
    - Evidence-based employee-development literature from Chroma (employeedev_kb).
    - Synthetic employee profiles (strengths, weaknesses, scores, burnout risk, training history).
    - Available trainings catalog (to recommend only relevant, not-yet-taken trainings).
    - Reject junk/off-topic queries.
    - Refuse to advise on employees that do not exist in the synthetic team configuration.
    - Never invent trainings or generic advice without context.
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
                "Did you build the employee development embeddings successfully?\n"
            )

        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.chroma_client.get_collection(EMPDEV_COLLECTION_NAME)

        # Load synthetic employee data and available trainings
        self.employee_profiles = self._load_employee_profiles()
        self.available_trainings = self._load_available_trainings()

        # Set number of chunks to retrieve per query
        self.n_results = 7

    
    def _load_employee_profiles(self) -> Dict[str, Dict[str, Any]]:
        """
        Load synthetic employee data from JSON and index by lowercased name.

        Expected structure (top-level):
        - Either: a list of employee dicts, each with at least "name"
        - Or: a dict with key "employees" that is a list of such dicts
        """
        if not EMPLOYEE_DATA_FILE.exists():
            raise RuntimeError(
                f"Employee data file not found: {EMPLOYEE_DATA_FILE}.\n"
                "Please generate it by running:\n"
                "  python generate_synthetic_employeeData.py\n"
            )

        with open(EMPLOYEE_DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "employees" in data:
            employees = data["employees"]
        elif isinstance(data, list):
            employees = data
        else:
            raise RuntimeError(
                "Unexpected structure in synthetic_employeeData.json. "
                "Expected a list or a dict with key 'employees'."
            )

        profiles: Dict[str, Dict[str, Any]] = {}
        for emp in employees:
            name = emp.get("name")
            if not name:
                continue
            profiles[name.lower()] = emp

        return profiles

    def _load_available_trainings(self) -> List[Dict[str, Any]]:
        """
        Load the available trainings catalog from JSON.
        """
        if not AVAILABLE_TRAININGS_FILE.exists():
            raise RuntimeError(
                f"Available trainings file not found: {AVAILABLE_TRAININGS_FILE}.\n"
                "Please ensure the trainings JSON is present in SyntheticEmployeeData.\n"
            )

        with open(AVAILABLE_TRAININGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

        raise RuntimeError(
            "Unexpected structure in available_trainings.json. Expected a list."
        )

   

    def _embed_query(self, text: str) -> List[float]:
        """
        Create an embedding for the manager's query using the same embedding
        model that was used for the knowledge base.
        """
        response = self.client.embeddings.create(
            model=self.embed_model,
            input=[text],
        )
        return response.data[0].embedding

    def _retrieve_context(self, manager_query: str, debug: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve top N relevant chunks from the employeedev_kb collection.

        Returns a list of dicts:
        [
          {
            "id": ...,
            "text": ...,
            "paper_title": ...,
            "paper_authors": ...,
            "year": ...,
            "chunk_index": ...,
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
            print(f"\n[Employee Development Sub-Agent] Retrieved {len(context_items)} context chunks requested {self.n_results}).")
            for i, item in enumerate(context_items):
                title = item.get("paper_title") or "Unknown title"
                year = item.get("year")
                dist = item.get("distance")
                print(f"  {i+1}. {title} ({year})  - id={item['id']}  (distance={dist:.3f})")

        return context_items



    def _is_valid_dev_query(self, manager_query: str, debug: bool = True) -> bool:
        """
        Use the LLM as a simple classifier:
        - Return True only if the text is about employee development, growth, performance, strengths/weaknesses, coaching, burnout risk, etc.
        - Otherwise return False.
        """
        classifier_messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict classifier.\n"
                    "Your job is ONLY to decide whether the user's text is a meaningful description of an employee-development or performance question.\n\n"
                    "Answer with exactly one word: 'yes' or 'no'.\n"
                    "- 'yes' = clearly about coaching, growth, performance, potential, strengths/weaknesses, burnout risk, engagement, or development plans.\n"
                    "- 'no' = nonsense, too short, off-topic, or not about employee development.\n"
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
                f"\n[Employee Development Sub-Agent] Classifier decision for query: "
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

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict classifier.\n"
                    "You are given:\n"
                    f"- A list of known employees: {known_list_str}\n"
                    "- A manager's text.\n\n"
                    "Your job is ONLY to decide whether the text refers to those known employees, "
                    "or to other people not in the list, or to no specific person.\n\n"
                    "Return ONLY a single JSON object with this structure:\n"
                    "{\n"
                    '  \"scope\": \"only_known\" | \"only_unknown\" | \"mixed\" | \"none\",\n'
                    '  \"known_employees\": [list of known employees explicitly mentioned]\n'
                    "}\n\n"
                    "Definitions:\n"
                    "- \"only_known\"  = all specifically named people are in the known list.\n"
                    "- \"only_unknown\" = it clearly talks about specific people, but none of them are in the known list.\n"
                    "- \"mixed\"       = both known and unknown specific people are clearly mentioned.\n"
                    "- \"none\"        = no specific person is mentioned (e.g. just 'my team', 'people', 'colleagues').\n"
                ),
            },
            {
                "role": "user",
                "content": manager_query,
            },
        ]

        resp = self.client.chat.completions.create(
            model=self.model_name,
            response_format={"type": "json_object"},
            messages=messages,
        )

        raw = resp.choices[0].message.content
        if debug:
            print(f"[Employee Development Sub-Agent] Employee scope raw JSON: {raw}")

        try:
            data = json.loads(raw)
        except Exception:
            # If parsing fails, fall back to assuming no specific employee scope
            if debug:
                print(
                    "[Employee Development Sub-Agent] Failed to parse employee scope JSON; defaulting to scope='none'.")
            return {"scope": "none", "known_employees": []}

        scope = data.get("scope", "none")
        known_emps = data.get("known_employees", [])
        if not isinstance(known_emps, list):
            known_emps = []

        # Normalize recognized known names to exact canonical forms
        normalized_known: List[str] = []
        lower_to_canonical = {name.lower(): name for name in KNOWN_EMPLOYEES}
        for name in known_emps:
            if isinstance(name, str) and name.lower() in lower_to_canonical:
                normalized_known.append(lower_to_canonical[name.lower()])

        result = {
            "scope": scope,
            "known_employees": normalized_known,
        }

        if debug:
            print(f"[Employee Development Sub-Agent] Employee scope parsed: {result}")

        return result

    

    def _get_employee_profile(self, name_lower: str) -> Optional[Dict[str, Any]]:
        """
        Fetch the profile for a given employee name (lowercased).
        """
        return self.employee_profiles.get(name_lower)

    def _get_suggestable_trainings(self,employee_profile: Dict[str, Any],manager_query: str,) -> List[Dict[str, Any]]:
        """
        Filter available trainings that:
        - The employee has NOT already taken (by title).
        - Roughly match the topic/concerns in the query or the profile's weaknesses.

        We rely on the LLM to interpret which of these
        to prioritise in the final answer, but we avoid large irrelevant noise.
        """
        if not self.available_trainings:
            return []

        q_lower = manager_query.lower()
        employee_trainings = {
            (t.get("title") or "").strip().lower()
            for t in employee_profile.get("training_history", [])
        }

        weaknesses = employee_profile.get("weaknesses", [])
        weaknesses_lower = [w.lower() for w in weaknesses]

        suggestable: List[Dict[str, Any]] = []

        for training in self.available_trainings:
            title = (training.get("title") or "").strip()
            title_lower = title.lower()
            category = (training.get("category") or "").lower()
            description = (training.get("description") or "").lower()

            if not title:
                continue

            if title_lower in employee_trainings:
                continue

            
            relevance_score = 0
            for w in weaknesses_lower:
                if w and (w in title_lower or w in description):
                    relevance_score += 1

            if category and category in q_lower:
                relevance_score += 1
            if "burnout" in q_lower and "burnout" in description:
                relevance_score += 1
            if "leadership" in q_lower and "leader" in description:
                relevance_score += 1

            if relevance_score > 0:
                suggestable.append(training)

        return suggestable


    def _build_system_prompt(self) -> str:
        """
        System prompt defining the role and constraints of this sub-agent.
        """
        return """
You are Sub-Agent Employee Development in a Leadership Mentoring System.

You specialize in:
- Employee growth and development.
- Strengths and weaknesses assessment.
- Performance and skill-gap development planning.
- Burnout risk awareness and sustainable performance.
- Choosing between coaching, mentoring, and training interventions.

You receive:
1) A manager's description of an employee or team development situation.
2) A set of evidence-based context chunks derived from research papers, articles and reports on employee development, coaching, performance, training and burnout.
3) For some queries, a structured profile of a specific employee (synthetic data).
4) For some queries, a list of available trainings that can be suggested.

Your task:
- Use ONLY the provided context as your knowledge base (papers, profile, trainings list).
- Do NOT rely on your own general world knowledge.
- Do NOT invent training programs or interventions that are not present in the context.
- If an employee profile is provided, align your advice with their strengths, weaknesses, scores (engagement, leadership, burnout_risk) and training history.
- If a trainings list is provided, you may recommend trainings ONLY from that list.

Output style:
- For development action plans, provide a SHORT answer: 3 to 4 numbered steps only.
- Each step should be 1 to 3 short sentences.
- Focus on concrete actions the manager can take (what to say/do or which training to assign).
- It is okay to refer to 'research' or 'evidence' in general terms, but do NOT cite chunk IDs or technical details.
- Do NOT ask follow-up questions.
- Do NOT invite the manager to provide more information.
- Do NOT offer extra options like 'if you want, I can...'.
- You are simply giving a brief, evidence-based plan.

Very Important:
- If you are told that there is no profile for the mentioned employee, clearly say that the system has no internal data and cannot safely advise.
- If the context from papers and profile is clearly insufficient or irrelevant, explicitly say so and give a cautious, high-level recommendation or refuse to advise.
- Please keep your response short and do not generate huge paragraphs of text.
"""

    def _build_user_prompt(self,manager_query: str,context_items: List[Dict[str, Any]],employee_profile: Optional[Dict[str, Any]],suggestable_trainings: Optional[List[Dict[str, Any]]]) -> str:
        """
        Build the user-facing prompt that includes:
        - the manager's problem
        - retrieved literature context
        - the internal employee profile (if any)
        - suggested trainings list (if any)
        """
        # Build literature context block
        if not context_items:
            context_block = (
                "No relevant context could be retrieved from the employee development knowledge base."
            )
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

        # Build employee profile block (if any)
        if employee_profile:
            name = employee_profile.get("name", "Unknown")
            skills = employee_profile.get("skills", [])
            weaknesses = employee_profile.get("weaknesses", [])
            engagement = employee_profile.get("engagement_score", "Unknown")
            leadership = employee_profile.get("leadership_score", "Unknown")
            burnout = employee_profile.get("burnout_risk", "Unknown")

            profile_block = (
                f"Internal synthetic profile for the employee mentioned:\n"
                f"- Name: {name}\n"
                f"- Skills: {', '.join(skills) if skills else 'None reported'}\n"
                f"- Weaknesses / development areas: {', '.join(weaknesses) if weaknesses else 'None reported'}\n"
                f"- Engagement score: {engagement}\n"
                f"- Leadership score: {leadership}\n"
                f"- Burnout risk: {burnout}\n"
            )
        else:
            profile_block = (
                "No specific employee profile is provided. You may give team-level development advice only."
            )

        # Build trainings block (if any)
        if suggestable_trainings:
            t_parts = []
            for idx, t in enumerate(suggestable_trainings, start=1):
                title = t.get("title", "Unknown title")
                category = t.get("category", "Unspecified category")
                description = t.get("description", "").strip()
                t_parts.append(
                    f"- Training {idx}: {title} "
                    f"(Category: {category})"
                    + (f" — {description}" if description else "")
                )
            trainings_block = (
                "Available trainings that the employee has NOT yet taken:\n"
                + "\n".join(t_parts)
            )
        else:
            trainings_block = (
                "No additional trainings are available for suggestion, or "
                "no relevant trainings could be identified."
            )

        user_content = (
            "Manager's employee development description:\n"
            f"\"\"\"{manager_query}\"\"\"\n\n"
            "Relevant evidence-based context from employee development literature:\n"
            f"{context_block}\n\n"
            f"{profile_block}\n\n"
            f"{trainings_block}\n\n"
            "Using ONLY the ideas and guidance from this context, provide a concise "
            "employee-development recommendation.\n"
            "Your answer MUST:\n"
            "- Be 3 to 4 numbered steps only (unless simply summarizing strengths/weaknesses).\n"
            "- Each step 1 to 3 short sentences.\n"
            "- Focus on clear, concrete actions or phrases the manager can use.\n"
            "- NOT ask any follow-up questions.\n"
            "- NOT invite the manager to provide more details.\n"
            "- NOT invent new trainings or theories not supported by the context.\n"
        )

        return user_content

    

    def _answer_strengths_weaknesses_only(self,name_lower: str,employee_profile: Dict[str, Any],debug: bool = True) -> str:
        """
        Handle queries like "What are Bob's strengths and weaknesses?"
        using ONLY the synthetic profile, without calling the LLM.
        """
        name = employee_profile.get("name", name_lower)
        skills = employee_profile.get("skills", [])
        weaknesses = employee_profile.get("weaknesses", [])
        engagement = employee_profile.get("engagement_score", "Unknown")
        leadership = employee_profile.get("leadership_score", "Unknown")
        burnout = employee_profile.get("burnout_risk", "Unknown")

        if debug:
            print(
                f"[Employee Development Sub-Agent] Returning direct strengths/weaknesses summary for {name}.\n"
            )

        lines = []

        lines.append(f"Strengths and weaknesses summary for {name}:")
        lines.append("")
        lines.append("Strengths:")
        if skills:
            for s in skills:
                lines.append(f"- {s}")
        else:
            lines.append("- None specifically reported in the internal data.")

        lines.append("")
        lines.append("Weaknesses / development areas:")
        if weaknesses:
            for w in weaknesses:
                lines.append(f"- {w}")
        else:
            lines.append("- None specifically reported in the internal data.")

        lines.append("")
        lines.append("Additional internal indicators:")
        lines.append(f"- Engagement score: {engagement}")
        lines.append(f"- Leadership score: {leadership}")
        lines.append(f"- Burnout risk: {burnout}")

        return "\n".join(lines)


    def run(self, manager_query: str, debug: bool = True) -> str:
        """
        Main entrypoint for the Employee Development sub-agent.

        Steps:
        1) Check if the query is about employee development/performance at all.
        2) Determine whether the query mentions specific employees and whether they are known/unknown.
        3) For pure strengths/weaknesses questions about a known employee:
           - Return a direct, LLM-free summary from the synthetic profile.
        4) Otherwise, retrieve top N relevant employee-development chunks from Chroma.
        5) Build a prompt with the manager_query, retrieved context, employee profile,
           and suggestable trainings (if applicable).
        6) Ask the LLM to generate a short, development-focused response.
        7) Return the response text as a string.
        """
        if debug:
            print("\n[Employee Development Sub-Agent] Received manager query:\n")
            print(manager_query)

        # Check if this even looks like an employee development query
        if not self._is_valid_dev_query(manager_query, debug=debug):
            if debug:
                print(
                    "[Employee Development Sub-Agent] Query is not a clear employee development or performance question and so cannot be answered.\n"
                )
            return (
                "The Employee Development Sub-Agent could not detect a clear employee development, performance, or growth question in your description."
            )

        # Check whether this query is about known/unknown employees (via LLM)
        scope_info = self._detect_employee_scope_with_llm(manager_query, debug=debug)
        scope = scope_info.get("scope", "none")
        known_emps = scope_info.get("known_employees", [])

        # If the query is ONLY about unknown named people or mixed, refuse to answer
        if scope in ("only_unknown", "mixed"):
            if debug:
                print("[Employee Development Sub-Agent] Query is about specific employees outside the known synthetic team.\n")
            return (
                "The Employee Development Sub-Agent can only provide suggestions for the members of the team configured.\n"
                "It cannot safely provide person-specific development guidance for other named employees."
            )

        # Decide if we have a single, specific known employee to personalise for
        name_lower: Optional[str] = None
        if len(known_emps) == 1:
            name_lower = known_emps[0].lower()

        employee_profile: Optional[Dict[str, Any]] = None
        suggestable_trainings: Optional[List[Dict[str, Any]]] = None

        if name_lower is not None:
            employee_profile = self._get_employee_profile(name_lower)
            if employee_profile is None:
                if debug:
                    print("[Employee Development Sub-Agent] The query mentions an employee that does not exist in the synthetic internal data.\n")
                return (
                    "The Employee Development Sub-Agent has no internal development "
                    "data for the mentioned employee and cannot safely provide a "
                    "personalized recommendation."
                )

            # For recognized employees, pre-compute suggestable trainings
            suggestable_trainings = self._get_suggestable_trainings(
                employee_profile, manager_query
            )

        # If the query is clearly about strengths/weaknesses only, answer directly
        q_lower = manager_query.lower()
        if (
            name_lower is not None
            and employee_profile is not None
            and (
                "strength" in q_lower
                or "strengths" in q_lower
                or "weakness" in q_lower
                or "weaknesses" in q_lower
            )
        ):
            return self._answer_strengths_weaknesses_only(
                name_lower, employee_profile, debug=debug
            )

        # Retrieve context from Chroma (evidence-based literature)
        context_items = self._retrieve_context(manager_query, debug=debug)

        # If no context at all, refuse to answer with generic LLM knowledge
        if not context_items:
            if debug:
                print("\n[Employee Development Sub-Agent] No context retrieved; refusing to answer without evidence-based knowledge.")
            return (
                "The Employee Development Sub-Agent could not retrieve any relevant "
                "evidence-based knowledge from its employee development sources for "
                "this query. It cannot safely provide a recommendation in this case."
            )

        # Build main prompt with context, profile, and trainings
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            manager_query, context_items, employee_profile, suggestable_trainings
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if debug:
            print("\n[Employee Development Sub-Agent] Sending request to LLM for final development-focused response.")

        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )

        answer = response.choices[0].message.content

        if debug:
            print("\n[Employee Development Sub-Agent] Final response from Employee Development Sub-Agent:")
            print(answer)

        return answer


if __name__ == "__main__":
    # For Testing
    agent = SubAgent3_EmployeeDevelopment()

    print("Hi I am Sub-Agent Employee Development. Type your employee development / performance related problem description (or 'q' to quit). ")
    while True:
        user_input = input("\nManager employee development description: \n")
        if user_input.strip().lower() in {"q", "quit", "exit"}:
            break

        result = agent.run(user_input, debug=True)
        print("\n[Agent Output]...\n")
        print(result)
