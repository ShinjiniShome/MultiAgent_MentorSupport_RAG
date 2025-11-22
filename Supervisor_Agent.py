import os
import json
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

from SubAgent1_ConflictResolution import SubAgent1_ConflictResolution
from SubAgent2_CommunicationAnalysis import SubAgent2_CommunicationAnalysis
from SubAgent3_EmployeeDevelopment import SubAgent3_EmployeeDevelopment
from SubAgent4_ProductivityMetricsAnalysis import SubAgent4_ProductivityMetricsAnalysis


AGENT_MAP = {
    "SubAgent1_ConflictResolution": SubAgent1_ConflictResolution,
    "SubAgent2_CommunicationAnalysis": SubAgent2_CommunicationAnalysis,
    "SubAgent3_EmployeeDevelopment": SubAgent3_EmployeeDevelopment,
    "SubAgent4_ProductivityMetricsAnalysis": SubAgent4_ProductivityMetricsAnalysis
}



ALLOWED_AGENTS = [
    "SubAgent1_ConflictResolution",
    "SubAgent2_CommunicationAnalysis",
    "SubAgent3_EmployeeDevelopment",
    "SubAgent4_ProductivityMetricsAnalysis",
]


class SupervisorAgent:
    """
    SupervisorAgent:
    - Takes a manager's free-text query
    - Asks an LLM which sub-agents are most relevant
    - Returns a JSON-serializable dict:
        {
          "agents": [...],
          "fallback_agents": [...],
          "status": "ok" | "no_strong_match"
        }
    """

    def __init__(self) -> None:
        # Load environment variables
        load_dotenv()

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not found. Please set it in your .env file."
            )

        # Default model can be overridden via .env
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-5-mini")

        # Create OpenAI client
        self.client = OpenAI(api_key=api_key)

    @staticmethod
    def _build_system_prompt() -> str: # Task Prompts for LLM for Main Agent
        """
        System instructions for the LLM.
        We define:
        - available sub-agents
        - what they are responsible for
        - the JSON schema and rules
        """
        return """
You are Supervisor Agent in a Leadership Mentoring System.

Your task:
Given a manager's problem description, decide:
- Which sub-agents are the best primary handlers ("agents").
- Which sub-agents are reasonable secondary options ("fallback_agents") if no strong match exists.
- Whether there is at least one strong match ("status" = "ok") or not ("status" = "no_strong_match").

Available sub-agents (use these exact identifiers only):

1. SubAgent1_ConflictResolution
   - Focus: interpersonal tensions, conflicts, trust breakdowns, disagreements, emotional friction.

2. SubAgent2_CommunicationAnalysis
   - Focus: too many meetings, miscommunication, information overload, unclear channels, async/sync issues.

3. SubAgent3_EmployeeDevelopment
   - Focus: motivation, growth, low engagement, skills, training needs, career development.

4. SubAgent4_ProductivityMetricsAnalysis
   - Focus: KPIs, performance issues, workload, burnout risk, productivity, deadlines, output metrics.

JSON output format:
You MUST respond with **only** a single JSON object, no extra text, in this exact structure:

{
  "agents": [
    "SubAgent1_ConflictResolution",
    "SubAgent2_CommunicationAnalysis"
  ],
  "fallback_agents": [
    "SubAgent3_EmployeeDevelopment"
  ],
  "status": "ok"
}

Rules:
- "agents": list of the strongest relevant sub-agents (0, 1 or more).
- "fallback_agents": list of possible alternatives (0, 1 or more), different from "agents".
- If there is at least one clearly relevant agent, set "status": "ok".
- If there is no clearly relevant agent, leave "agents": [] and use "fallback_agents" (if any)
  with "status": "no_strong_match".
- Only use agent names from the allowed list above.
- Never invent new agent names.
- Do NOT include any explanation or text outside the JSON object.
- If the manager's description is extremely short, clearly random, or meaningless
  (for example: "uuu", "ccc", "asdf", a single word with no leadership context, or only symbols),
  then you MUST return:
  {
    "agents": [],
    "fallback_agents": [],
    "status": "no_strong_match"
  }
  and you must NOT select any agents.
- If you are unsure or the query is too ambiguous to confidently assign to a sub-agent,
  prefer returning no agents (empty lists) with "status": "no_strong_match" instead of guessing.
  Example (nonsense input):
  Manager problem description: "xxx","tyupoi", "no problem"

  Expected JSON response:
  {
    "agents": [],
    "fallback_agents": [],
    "status": "no_strong_match"
  }

"""

    def _build_messages(self, manager_query: str) -> List[Dict[str, str]]:
        """
        Build messages for the chat completion call.
        """
        system_content = self._build_system_prompt()
        user_content = (
            "Manager problem description:\n"
            f"\"\"\"{manager_query}\"\"\"\n\n"
            "Decide which sub-agents are relevant and respond with the JSON object."
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    def route(self, manager_query: str) -> Dict[str, Any]:
        """
        Main entrypoint:
        - Sends the query to the model in JSON mode.
        - Parses and validates the JSON.
        - Returns a safe dict that the rest of the system can rely on.
        """
        messages = self._build_messages(manager_query)

        response = self.client.chat.completions.create(
            model=self.model_name,
            response_format={"type": "json_object"},
            messages=messages
        )

        raw_content = response.choices[0].message.content

        # Parse JSON and normalize
        try:
            data = json.loads(raw_content)
        except json.JSONDecodeError:
            # Fallback: treat as unsupported / no match
            return {
                "agents": [],
                "fallback_agents": [],
                "status": "no_strong_match",
                "error": "invalid_json_from_model",
            }

        # Normalize and validate keys
        agents = data.get("agents", [])
        fallback_agents = data.get("fallback_agents", [])
        status = data.get("status", "no_strong_match")

        # Ensure lists are lists of strings
        if not isinstance(agents, list):
            agents = []
        if not isinstance(fallback_agents, list):
            fallback_agents = []

        agents = [a for a in agents if isinstance(a, str) and a in ALLOWED_AGENTS]
        fallback_agents = [
            a for a in fallback_agents if isinstance(a, str) and a in ALLOWED_AGENTS
        ]

        # Remove duplicates and collisions
        agents = list(dict.fromkeys(agents))
        fallback_agents = [
            a for a in dict.fromkeys(fallback_agents) if a not in agents
        ]

        # Normalize status
        if status not in ("ok", "no_strong_match"):
            status = "no_strong_match"

        # If model claims "ok" but agents is empty, downgrade
        if status == "ok" and not agents:
            status = "no_strong_match"

        # If everything is empty, we keep "no_strong_match"
        result = {
            "agents": agents,
            "fallback_agents": fallback_agents,
            "status": status,
        }

        return result
    
    def execute_agents(self, manager_query: str, agent_names: list, debug: bool = True):
        """
        Execute the selected sub-agents and collect their responses.
        Uses the AGENT_MAP to instantiate and call each agent's `run()` method.
        """
        outputs = []

        for name in agent_names:
            agent_class = AGENT_MAP.get(name)
            if agent_class is None:
                if debug:
                    print(f"[WARNING] No implementation found for agent: {name}")
                continue

            try:
                instance = agent_class()
                result = instance.run(manager_query)
                outputs.append(
                    {
                        "agent": name,
                        "response": result,
                    }
                )
            except Exception as e:
                if debug:
                    print(f"[ERROR] Failed to run {name}: {e}")
                outputs.append(
                    {
                        "agent": name,
                        "response": f"[ERROR] Exception while running {name}: {e}",
                    }
                )

        return outputs
    
    def consolidate_responses(self, manager_query: str, subagent_outputs: list) -> str:
        """
        Ask the LLM to combine all sub-agent outputs into ONE final mentoring response.
        No fine-tuning, just a second LLM call.
        """
        if not subagent_outputs:
            return (
                "No sub-agent responses were available. "
                "The prototype could not generate a consolidated recommendation."
            )

        # Build a readable block of all sub-agent outputs
        parts = []
        for item in subagent_outputs:
            agent_name = item.get("agent", "UnknownAgent")
            resp = item.get("response", "")
            parts.append(f"{agent_name}:\n{resp}")

        combined = "\n\n".join(parts)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are the Supervisor Agent in a Leadership Mentoring System. "
                    "You receive analyses and suggestions from several specialized sub-agents. "
                    "Your task is to synthesize them into ONE clear, coherent, and non-repetitive response. "
                    "You ONLY have access to the text produced by the sub-agents. "
                    "You MUST base your answer strictly and exclusively on the sub-agent outputs. "
                    "DO NOT invent new advice, strategies, or ideas that are not already implied in the sub-agent text. "
                    "If the sub-agent outputs are vague, generic, junk or clearly placeholders, you must say that the prototype does not yet have enough implemented logic to give a concrete recommendation. "
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Manager's original problem:\n{manager_query}\n\n"
                    "Sub-agent outputs: Please note this is the ONLY information you may use\n"
                    f"{combined}\n\n"
                    "Task:\n"
                    "- Combine the sub-agent outputs into ONE clear response for the manager.\n"
                    "- Do NOT add any new ideas or external knowledge.\n"
                    "- If the sub-agent outputs are only placeholders or too generic, explicitly say that the system is not yet fully implemented and cannot provide a detailed recommendation. \n"
                    "- Now provide a single, integrated recommendation for the manager. \n"
                    "- Do not mention that you are aggregating multiple agents; just speak as one advisor. \n"
                ),
            },
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )

        return response.choices[0].message.content
    
    def run_full_flow(self, manager_query: str, debug: bool = True):
        """
        High-level flow:
        - Route to sub-agents (existing JSON routing).
        - Execute selected agents.
        - Optionally execute fallback agents if user agrees.
        - Consolidate all responses into one final answer.
        - Return everything in a dict for easier debugging.
        """
        routing = self.route(manager_query)

        if debug:
            print("\n[Routing decision]:")
            print(json.dumps(routing, indent=2))

        agents_to_run = routing.get("agents", []) or []
        fallback_agents = routing.get("fallback_agents", []) or []

        # Execute primary agents
        if debug:
            print("\n[Executing primary agents]...")
        primary_outputs = self.execute_agents(manager_query, agents_to_run, debug=debug)

        # Optionally execute fallback agents (only if list not empty)
        fallback_outputs = []
        if fallback_agents:
            if debug:
                print("\n[Fallback agents available]:")
                print(f"Fallback agents: {fallback_agents}")

            choice = input(
                "Do you also want to run the fallback agents? (yes/no): "
            ).strip().lower()

            if choice in {"yes", "y"}:
                if debug:
                    print("\n[Executing fallback agents]...")
                fallback_outputs = self.execute_agents(
                    manager_query, fallback_agents, debug=debug
                )
            else:
                if debug:
                    print("\n[Skipping fallback agents by user choice]...")

        all_outputs = primary_outputs + fallback_outputs

        if debug:
            print("\n[Sub-agent outputs]:")
            print(json.dumps(all_outputs, indent=2, ensure_ascii=False))

        final_answer = self.consolidate_responses(manager_query, all_outputs)

        if debug:
            print("\n[Final consolidated response]:")
            print(final_answer)

        return {
            "routing": routing,
            "subagent_outputs": all_outputs,
            "final_consolidated_response": final_answer,
        }




if __name__ == "__main__":
    agent = SupervisorAgent()

    print("Hi I am your Supervisor Agent: Please type your problem (or 'q' to quit).")
    while True:
        user_input = input("\nManager problem: ")
        if user_input.strip().lower() in {"q", "quit", "exit"}:
            break

        result = agent.run_full_flow(user_input, debug=True)
        # Dictionary output as a result.
