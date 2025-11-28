import random
import json
from pathlib import Path
from employee_configuration import Team, NUM_DAYS
from productivity_baseline_configuration import role_productivity_baseline, role_stretch_goals

DATA_DIR = Path(__file__).resolve().parent / "SyntheticEmployeeData"
DATA_DIR.mkdir(exist_ok=True)

def generate_productivity_data():
    data = []
    for day in range(NUM_DAYS):
        date = f"day {day+1}"
        for member in Team:
            role = member["role"]
            baseline = role_productivity_baseline.get(role, {})
            stretch = role_stretch_goals.get(role, {})
            daily_data = {
                "day": date,
                "name": member["name"]
            }
            for metric, range_val in baseline.items():
                rand_val = random.random()
                if rand_val < 0.1:
                    # Below baseline: generate value lower than the baseline's minimum
                    below_value = max(int(range_val[0] * random.uniform(0.5, 0.9)), 0)  # Ensure non-negative
                    daily_data[metric] = below_value
                elif rand_val < 0.7:
                    # Within baseline
                    daily_data[metric] = random.randint(range_val[0], range_val[1])
                else:
                    # Above baseline/stretch
                    stretch_goal = stretch.get(metric)
                    if stretch_goal:
                        exceed_value = int(stretch_goal * random.uniform(1.0, 1.2))
                        daily_data[metric] = exceed_value
                    else:
                        exceed_value = int(range_val[1] * random.uniform(1.1, 1.5))
                        daily_data[metric] = exceed_value
            data.append(daily_data)
    return data

def main():
    productivity = generate_productivity_data()
    with open(DATA_DIR /"synthetic_productivity_data.json", "w") as f:
        json.dump(productivity, f, indent=2)
    print("Synthetic Productivity Data generated\n")

if __name__ == "__main__":
    main()
