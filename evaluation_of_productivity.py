import json
from collections import defaultdict
from pathlib import Path
from employee_configuration import Team, NUM_DAYS
from productivity_baseline_configuration import role_productivity_baseline, role_stretch_goals

DATA_DIR = Path(__file__).resolve().parent / "SyntheticEmployeeData"
DATA_DIR.mkdir(exist_ok=True)


def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def evaluate_productivity(data, survey_data=None):
    evaluations = []
    overall_stats = defaultdict(lambda: {
        "stretch_exceeds": 0, 
        "below_baseline": 0, 
        "total_days": 0,
        "low_motivation_days": 0,
        "high_stress_days": 0,
        "heavy_workload_days": 0,
        "low_role_clarity_days": 0,
        "low_contribution_days": 0,
        "overload_flags": 0  # combined flags for potential overload/burnout
    })

    survey_lookup = {(entry["day"], entry["name"]): entry for entry in survey_data} if survey_data else {}

    for entry in data:
        name = entry["name"]
        role = next((member["role"] for member in Team if member["name"] == name), None)
        baseline = role_productivity_baseline.get(role, {})
        stretch = role_stretch_goals.get(role, {})
        day = entry["day"]

        evaluation = {
            "day": day,
            "name": name,
            "evaluations": [],
            "notes": []
        }

        # Evaluate productivity metrics
        productivity_below_baseline = False
        productivity_exceeds_stretch = False

        for metric, value in entry.items():
            if metric in ("day", "name"):
                continue
            baseline_range = baseline.get(metric)
            stretch_goal = stretch.get(metric)
            if baseline_range:
                status = "Within baseline"
                if value < baseline_range[0]:
                    status = "Below baseline"
                    productivity_below_baseline = True
                    overall_stats[name]["below_baseline"] += 1
                elif value > baseline_range[1]:
                    status = "Exceeds baseline"
                if stretch_goal and value >= stretch_goal:
                    status = "Exceeds stretch goal"
                    productivity_exceeds_stretch = True
                    overall_stats[name]["stretch_exceeds"] += 1

                evaluation["evaluations"].append({
                    "metric": metric,
                    "value": value,
                    "status": status
                })

        overall_stats[name]["total_days"] += 1

        # Survey data integration
        survey_entry = survey_lookup.get((day, name))
        if survey_entry:
            motivation = survey_entry.get("daily_motivation", 3)
            stress = survey_entry.get("stress_level", 3)
            workload = survey_entry.get("perceived_workload", "Normal")
            role_clarity = survey_entry.get("satisfaction_with_role_clarity", "Maybe")
            contribution = survey_entry.get("feeling_of_contribution", "Medium")

            # Flags
            if motivation <= 2:
                evaluation["notes"].append("Low motivation reported.")
                overall_stats[name]["low_motivation_days"] += 1
            if stress >= 4:
                evaluation["notes"].append("High stress level reported.")
                overall_stats[name]["high_stress_days"] += 1
            if workload.lower() == "heavy":
                evaluation["notes"].append("Heavy perceived workload reported.")
                overall_stats[name]["heavy_workload_days"] += 1
            if role_clarity.lower() in ("no", "maybe"):
                evaluation["notes"].append("Low satisfaction with role clarity.")
                overall_stats[name]["low_role_clarity_days"] += 1
            if contribution.lower() == "low":
                evaluation["notes"].append("Low feeling of contribution reported.")
                overall_stats[name]["low_contribution_days"] += 1

            # Combined overload/burnout potential
            if productivity_below_baseline and (stress >= 4 or workload.lower() == "heavy"):
                evaluation["notes"].append("Potential overload/burnout risk: low productivity with high stress or workload.")
                overall_stats[name]["overload_flags"] += 1

            # Unsustainable work: high productivity + low motivation
            if productivity_exceeds_stretch and motivation <= 2:
                evaluation["notes"].append("High productivity but low motivation: possible unsustainable work.")

        evaluations.append(evaluation)

    return evaluations, overall_stats


def main():
    productivity_data = load_json(DATA_DIR /"synthetic_productivity_data.json")
    survey_data = load_json(DATA_DIR /"synthetic_employee_survey_data.json")
    evaluations, overall_stats = evaluate_productivity(productivity_data, survey_data)

    with open(DATA_DIR /"synthetic_productivity_evaluation.json", "w") as f:
        json.dump(evaluations, f, indent=2)
    with open(DATA_DIR /"synthetic_productivity_overall_stats.json", "w") as f:
        json.dump(overall_stats, f, indent=2)
    print("Generated productivity_evaluation.json and productivity_overall_stats.json")

if __name__ == "__main__":
    main()
