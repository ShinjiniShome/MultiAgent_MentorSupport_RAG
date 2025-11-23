import json
import random
from faker import Faker
from pathlib import Path
from employee_configuration import Team, NUM_DAYS

DATA_DIR = Path(__file__).resolve().parent / "SyntheticEmployeeData"
DATA_DIR.mkdir(exist_ok=True)

fake = Faker()

# Corporate/team-related phrases to inject into Slack messages
work_phrases = [
    "Deadline approaching, please update your tasks.",
    "Can someone review my PR?",
    "Stand-up meeting in 10 minutes, please be on time.",
    "Found a bug in the payment module.",
    "Client sent feedback, need to discuss.",
    "Deploy scheduled for tomorrow afternoon.",
    "Can you update the project documentation?",
    "Sprint retrospective meeting on Friday.",
    "Reminder: Submit your timesheets today.",
    "Any blockers for this week's deliverables?"
]

slack_data = []

for day in range(1, NUM_DAYS + 1):
    participants = random.sample(Team, k=random.randint(2, 4))
    log = []
    for _ in range(random.randint(4, 8)):
        user = random.choice(participants)["name"]
        # 70% chance to use work-related phrases, else Faker sentence
        if random.random() < 0.7:
            message = random.choice(work_phrases)
        else:
            message = fake.sentence()
        timestamp = fake.time()
        log.append(f"[{timestamp}] {user}: {message}")
    slack_data.append({
        "day": day,
        "log": log
    })

with open(DATA_DIR /"synthetic_slack_logs.json", "w") as f:
    json.dump(slack_data, f, indent=2)

print(f"Generated Synthetic Slack logs for {NUM_DAYS} days.\n")
