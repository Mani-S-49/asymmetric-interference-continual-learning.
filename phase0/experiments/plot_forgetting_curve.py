import json
from pathlib import Path
import matplotlib.pyplot as plt

# Project paths
results_dir = Path("phase0/results")
baseline_file = results_dir / "english_baseline_before_hindi.json"
after_file = results_dir / "english_after_hindi.json"
output_file = results_dir / "forgetting_curve.png"

# Load the saved experimental results
with open(baseline_file, "r", encoding="utf-8") as f:
    baseline = json.load(f)

with open(after_file, "r", encoding="utf-8") as f:
    after = json.load(f)

# English IMDb accuracy at each stage
stages = ["Before Hindi training", "After Hindi training"]
accuracies = [
    baseline["accuracy"] * 100,
    after["english_accuracy_after_hindi"] * 100
]

# Calculate forgetting
forgetting = accuracies[0] - accuracies[1]

# Plot the forgetting curve
plt.figure(figsize=(9, 5.5))
plt.plot(stages, accuracies, marker="o", linewidth=2.5, markersize=9)

for i, accuracy in enumerate(accuracies):
    plt.annotate(
        f"{accuracy:.2f}%",
        (i, accuracy),
        textcoords="offset points",
        xytext=(0, 12),
        ha="center",
        fontsize=11
    )

plt.title("English Performance After Hindi Continual Learning")
plt.xlabel("Training Stage")
plt.ylabel("English IMDb Test Accuracy (%)")
plt.ylim(0, 100)
plt.grid(True, linestyle="--", alpha=0.5)

plt.figtext(
    0.5, 0.01,
    f"English accuracy drop: {forgetting:.2f} percentage points",
    ha="center",
    fontsize=11
)

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.close()

print("Forgetting curve saved successfully!")
print(f"File: {output_file.resolve()}")
print(f"English accuracy before Hindi: {accuracies[0]:.2f}%")
print(f"English accuracy after Hindi:  {accuracies[1]:.2f}%")
print(f"Forgetting: {forgetting:.2f} percentage points")
