import json
from collections import defaultdict
from pathlib import Path
from employee_configuration import Team, NUM_DAYS
from overload_threshold_configuration import (
    meeting_threshold,
    email_sent_threshold,
    email_received_threshold,
    slack_message_threshold,
    task_count_threshold
)


DATA_DIR = Path(__file__).resolve().parent / "SyntheticEmployeeData"

# Calendar Data Analysis
def analyze_calendar(calendar_data):
    total_meetings = 0
    total_duration_minutes = 0
    
    # Per employee per day meeting counts
    meeting_counts_per_employee_day = {emp["name"]: defaultdict(int) for emp in Team}
    
    for day_data in calendar_data:
        day = day_data.get("day", None)
        if day is None:
            continue
        for schedule in day_data.get("schedules", []):
            employee = schedule["employee"]
            meetings = schedule.get("meetings", [])
            meeting_counts_per_employee_day[employee][day] += len(meetings)
            total_meetings += len(meetings)
            for m in meetings:
                dur_str = m.get("duration", "0 mins")
                try:
                    dur_min = int(dur_str.split()[0])
                except Exception:
                    dur_min = 0
                total_duration_minutes += dur_min
    
    avg_meetings_per_day = total_meetings / NUM_DAYS if NUM_DAYS else 0
    avg_duration_per_meeting = total_duration_minutes / total_meetings if total_meetings > 0 else 0
    
    # Flag overload days per employee
    overload_days_per_employee = {}
    for emp in meeting_counts_per_employee_day:
        overload_days = [day for day, count in meeting_counts_per_employee_day[emp].items() if count > meeting_threshold]
        overload_days_per_employee[emp] = overload_days
    
    return {
        "total_meetings": total_meetings,
        "avg_meetings_per_day": avg_meetings_per_day,
        "avg_duration_per_meeting": avg_duration_per_meeting,
        "meeting_counts_per_employee_day": meeting_counts_per_employee_day,
        "overload_days_per_employee": overload_days_per_employee
    }

# Emails Data Analysis
def analyze_emails(email_data):
    emails_sent_per_employee_day = {emp["name"]: defaultdict(int) for emp in Team}
    emails_received_per_employee_day = {emp["name"]: defaultdict(int) for emp in Team}

    for email in email_data:
        day = email.get("day", None)
        sender = email.get("from")
        receiver = email.get("to")
        if day is None or sender is None or receiver is None:
            continue
        if sender in emails_sent_per_employee_day:
            emails_sent_per_employee_day[sender][day] += 1
        if receiver in emails_received_per_employee_day:
            emails_received_per_employee_day[receiver][day] += 1

    # Flag overload days per employee
    sent_overload_days = {}
    recv_overload_days = {}

    for emp in emails_sent_per_employee_day:
        sent_overload_days[emp] = [day for day, count in emails_sent_per_employee_day[emp].items() if count > email_sent_threshold]

    for emp in emails_received_per_employee_day:
        recv_overload_days[emp] = [day for day, count in emails_received_per_employee_day[emp].items() if count > email_received_threshold]

    return {
        "emails_sent_per_employee_day": emails_sent_per_employee_day,
        "emails_received_per_employee_day": emails_received_per_employee_day,
        "sent_overload_days": sent_overload_days,
        "recv_overload_days": recv_overload_days
    }

# Slack Data Analysis
def analyze_slack(slack_data):
    messages_per_employee_day = {emp["name"]: defaultdict(int) for emp in Team}

    for day_data in slack_data:
        day = day_data.get("day", None)
        logs = day_data.get("log", [])
        if day is None:
            continue
        for log_entry in logs:
            try:
                # Format: [HH:MM:SS] User: message
                parts = log_entry.split("] ")
                user_message = parts[1]
                user = user_message.split(":")[0]
                if user in messages_per_employee_day:
                    messages_per_employee_day[user][day] += 1
            except Exception:
                continue
    
    # Flag overload days
    overload_days_per_employee = {}
    for emp in messages_per_employee_day:
        overload_days_per_employee[emp] = [day for day, count in messages_per_employee_day[emp].items() if count > slack_message_threshold]

    return {
        "messages_per_employee_day": messages_per_employee_day,
        "overload_days_per_employee": overload_days_per_employee
    }

# Task Board Data Analysis
def analyze_tasks(task_data):
    tasks_per_employee_day = {emp["name"]: defaultdict(int) for emp in Team}
    task_status_counts = defaultdict(int)

    for day_data in task_data:
        day = day_data.get("day", None)
        tasks = day_data.get("tasks", [])
        if day is None:
            continue
        for task in tasks:
            assignee = task.get("assignee")
            status = task.get("status")
            if assignee in tasks_per_employee_day:
                tasks_per_employee_day[assignee][day] += 1
            if status:
                task_status_counts[status] += 1

    # Flag overload days
    overload_days_per_employee = {}
    for emp in tasks_per_employee_day:
        overload_days_per_employee[emp] = [day for day, count in tasks_per_employee_day[emp].items() if count > task_count_threshold]

    return {
        "tasks_per_employee_day": tasks_per_employee_day,
        "task_status_counts": dict(task_status_counts),
        "overload_days_per_employee": overload_days_per_employee
    }

# Compose a Detailed Summary with Flags
def compose_summary(calendar_analysis, email_analysis, slack_analysis, task_analysis):
    summary_lines = []

    summary_lines.append(f"Calendar Analysis:\n"
                         f"- Total meetings scheduled: {calendar_analysis['total_meetings']}\n"
                         f"- Average meetings per day: {calendar_analysis['avg_meetings_per_day']:.2f}\n"
                         f"- Average duration per meeting: {calendar_analysis['avg_duration_per_meeting']:.1f} mins\n")

    summary_lines.append("Meeting counts per employee per day:")
    for emp, day_counts in calendar_analysis['meeting_counts_per_employee_day'].items():
        day_counts_str = ", ".join(f"Day {day}: {count}" for day, count in sorted(day_counts.items()))
        summary_lines.append(f"  - {emp}: {day_counts_str if day_counts_str else 'No meetings'}")
    summary_lines.append("\nMeeting overload days per employee (days exceeding threshold):")
    for emp, days in calendar_analysis['overload_days_per_employee'].items():
        summary_lines.append(f"  - {emp}: {days if days else 'None'}")

    summary_lines.append("\nEmails Sent and Received per employee per day:")
    for emp in email_analysis["emails_sent_per_employee_day"]:
        sent_counts = email_analysis["emails_sent_per_employee_day"][emp]
        recv_counts = email_analysis["emails_received_per_employee_day"][emp]
        sent_str = ", ".join(f"Day {day}: {count}" for day, count in sorted(sent_counts.items()))
        recv_str = ", ".join(f"Day {day}: {count}" for day, count in sorted(recv_counts.items()))
        sent_overload = email_analysis["sent_overload_days"].get(emp, [])
        recv_overload = email_analysis["recv_overload_days"].get(emp, [])
        summary_lines.append(f"  - {emp}: Sent: {sent_str if sent_str else 'None'}, Received: {recv_str if recv_str else 'None'}")
        summary_lines.append(f"     Sent overload days: {sent_overload if sent_overload else 'None'}, Received overload days: {recv_overload if recv_overload else 'None'}")

    summary_lines.append("\nSlack messages per employee per day:")
    for emp, day_counts in slack_analysis["messages_per_employee_day"].items():
        day_counts_str = ", ".join(f"Day {day}: {count}" for day, count in sorted(day_counts.items()))
        overload_days = slack_analysis["overload_days_per_employee"].get(emp, [])
        summary_lines.append(f"  - {emp}: {day_counts_str if day_counts_str else 'None'}")
        summary_lines.append(f"     Overload days: {overload_days if overload_days else 'None'}")

    summary_lines.append("\nTask counts per employee per day:")
    for emp, day_counts in task_analysis["tasks_per_employee_day"].items():
        day_counts_str = ", ".join(f"Day {day}: {count}" for day, count in sorted(day_counts.items()))
        overload_days = task_analysis["overload_days_per_employee"].get(emp, [])
        summary_lines.append(f"  - {emp}: {day_counts_str if day_counts_str else 'None'}")
        summary_lines.append(f"     Overload days: {overload_days if overload_days else 'None'}")

    summary_lines.append("\nTask Status Counts:")
    for status, count in task_analysis["task_status_counts"].items():
        summary_lines.append(f"  - {status}: {count}")

    return "\n".join(summary_lines)

# Load JSON files
def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)



if __name__ == "__main__":
    # Update these filenames if JSON files have different names or paths
    calendar_file = DATA_DIR / "synthetic_calendar_data.json"
    email_file = DATA_DIR / "synthetic_emails.json"
    slack_file = DATA_DIR / "synthetic_slack_logs.json"
    task_file = DATA_DIR / "synthetic_task_board.json"

    calendar_data = load_json(calendar_file)
    email_data = load_json(email_file)
    slack_data = load_json(slack_file)
    task_data = load_json(task_file)

    calendar_analysis = analyze_calendar(calendar_data)
    email_analysis = analyze_emails(email_data)
    slack_analysis = analyze_slack(slack_data)
    task_analysis = analyze_tasks(task_data)

    report = compose_summary(calendar_analysis, email_analysis, slack_analysis, task_analysis)

    print("\n=== Analysis Summary Report ===\n")
    print(report)



