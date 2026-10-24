import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SEED = 42
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# --------------------------------------------------
# Create three simple classification tasks
# --------------------------------------------------

def create_task(task_id, n_samples=1000):
    """
    Create a simple 2D classification task.

    Each task has a different decision boundary.
    """

    x = torch.randn(n_samples, 2)

    if task_id == 0:
        # Task A: classify according to x1 + x2
        y = (x[:, 0] + x[:, 1] > 0).long()

    elif task_id == 1:
        # Task B: classify according to x1 - x2
        y = (x[:, 0] - x[:, 1] > 0).long()

    elif task_id == 2:
        # Task C: classify according to distance from origin
        y = ((x[:, 0] ** 2 + x[:, 1] ** 2) > 1.0).long()

    else:
        raise ValueError("Unknown task")

    return x, y


# --------------------------------------------------
# Model
# --------------------------------------------------

class SimpleMLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.network(x)


# --------------------------------------------------
# Training
# --------------------------------------------------

def train(model, loader, epochs=10):

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.05
    )

    model.train()

    for epoch in range(epochs):

        for x, y in loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            output = model(x)

            loss = criterion(output, y)

            loss.backward()

            optimizer.step()


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate(model, loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            output = model(x)

            predictions = output.argmax(dim=1)

            correct += (predictions == y).sum().item()
            total += y.size(0)

    return correct / total


# --------------------------------------------------
# Prepare tasks
# --------------------------------------------------

tasks = []

for task_id in range(3):

    train_x, train_y = create_task(task_id, 1000)
    test_x, test_y = create_task(task_id, 500)

    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256
    )

    tasks.append(
        {
            "train": train_loader,
            "test": test_loader
        }
    )


# --------------------------------------------------
# Continual learning experiment
# --------------------------------------------------

model = SimpleMLP().to(DEVICE)

results = []

for task_id in range(3):

    print()
    print("=" * 50)
    print(f"Learning Task {task_id + 1}")
    print("=" * 50)

    # Train on current task
    train(
        model,
        tasks[task_id]["train"],
        epochs=10
    )

    # Evaluate on every task seen so far
    task_results = {}

    for eval_task in range(task_id + 1):

        accuracy = evaluate(
            model,
            tasks[eval_task]["test"]
        )

        task_results[eval_task + 1] = accuracy

        print(
            f"Task {eval_task + 1} accuracy: "
            f"{accuracy:.4f}"
        )

    results.append(task_results)


# --------------------------------------------------
# Final summary
# --------------------------------------------------

print()
print("=" * 50)
print("CONTINUAL LEARNING SUMMARY")
print("=" * 50)

for step, task_results in enumerate(results):

    print(f"\nAfter learning Task {step + 1}:")

    for task_id, accuracy in task_results.items():

        print(
            f"  Task {task_id}: "
            f"{accuracy:.4f}"
        )