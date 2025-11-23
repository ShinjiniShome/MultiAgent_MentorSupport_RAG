import json
import random
from faker import Faker
from pathlib import Path
from employee_configuration import Team, NUM_DAYS

DATA_DIR = Path(__file__).resolve().parent / "SyntheticEmployeeData"
DATA_DIR.mkdir(exist_ok=True)

fake = Faker()
email_data = []

subjects = [
    "Sprint Planning", "Status Update", "Client Feedback", "Release Prep", "Bug Report",
    "Design Review", "Follow-up", "Reschedule", "New Tasks", "Meeting Summary"
]

work_phrases = [
    "Please review the latest pull request for the authentication module.",
    "The client has provided feedback on the wireframes. Let's discuss in the next sync.",
    "We've scheduled a code review for tomorrow. Can you take a look?",
    "Reminder: Submit your timesheets by end of day.",
    "The deployment for the payment service is planned for Friday.",
    "Let's schedule a meeting to go over the new sprint tasks.",
    "A critical bug was found in the reporting module. Please investigate.",
    "The product team shared new design updates. Feedback is welcome.",
    "QA has completed testing on the latest build. See attached report.",
    "Please update the project documentation with the latest changes."
]

for day in range(1, NUM_DAYS + 1):
    daily_emails = []
    pairs = []
    # Generate 1-5 unique sender-receiver pairs
    num_pairs = random.randint(1, 5)
    attempts = 0
    while len(pairs) < num_pairs and attempts < 20:
        participants = random.sample(Team, 2)
        pair = tuple(sorted([participants[0]['name'], participants[1]['name']]))
        if pair not in pairs:
            pairs.append(pair)
        attempts += 1
    
    # Generate one email per pair
    for sender_name, receiver_name in pairs:
        subject = random.choice(subjects)
        if random.random() < 0.7:
            body = random.choice(work_phrases)
        else:
            body = fake.sentence(nb_words=12)
        daily_emails.append({
            "day": day,
            "from": sender_name,
            "to": receiver_name,
            "subject": subject,
            "body": body
        })
    
    email_data.extend(daily_emails)

# Save to JSON
with open(DATA_DIR /"synthetic_emails.json", "w") as f:
    json.dump(email_data, f, indent=2)

print(f"Synthetically Generated {len(email_data)} emails across {NUM_DAYS} days with unique sender-receiver pairs each day.\n")

