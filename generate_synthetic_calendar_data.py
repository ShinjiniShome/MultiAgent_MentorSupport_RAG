import json
import random
from faker import Faker
from pathlib import Path
from employee_configuration import Team, NUM_DAYS

DATA_DIR = Path(__file__).resolve().parent / "SyntheticEmployeeData"
DATA_DIR.mkdir(exist_ok=True)

fake = Faker()
calendar_data = []

# Work-related meeting subjects
meeting_titles = [
    "Daily Stand-up",
    "Sprint Planning",
    "Client Feedback Review",
    "Design Review",
    "QA Sync",
    "Backend Architecture Discussion",
    "Frontend Feature Planning",
    "Project Retrospective",
    "One-on-One Meeting",
    "Cross-Team Collaboration"
]

# Meeting types and locations
meeting_types = ["Virtual", "In-Person"]
locations = ["Zoom", "Teams", "Conference Room A", "Conference Room B", "Board Room"]

for day in range(1, NUM_DAYS + 1):
    daily_entry = {"day": day, "schedules": []}
    for member in Team:
        num_meetings = random.randint(1, 4)
        meetings = []
        for _ in range(num_meetings):
            start_hour = random.randint(8, 17)  # 8 AM to 5 PM
            start_minute = random.choice([0, 15, 30, 45])
            time = f"{start_hour:02d}:{start_minute:02d}"
            duration = f"{random.choice([15, 30, 45, 60])} mins"
            meeting = {
                "title": random.choice(meeting_titles),
                "time": time,
                "duration": duration,
                "type": random.choice(meeting_types),
                "location": random.choice(locations)
            }
            meetings.append(meeting)
        daily_entry["schedules"].append({
            "employee": member["name"],
            "meetings": meetings
        })
    calendar_data.append(daily_entry)

with open(DATA_DIR /"synthetic_calendar_data.json", "w") as f:
    json.dump(calendar_data, f, indent=2)

print(f"Synthetic Calendar Data generated for {NUM_DAYS} days.\n")
