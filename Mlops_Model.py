import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import wandb
from sklearn.metrics import precision_score, recall_score


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, dropout_rate):
        super(LogisticRegression, self).__init__()
        # FIXME: Implement the model architecture
        # Hint: Consider adding hidden layers and dropout
        self.network = nn.Sequential(*self._create_layers(input_dim, output_dim, hidden_layers, dropout_rate))

    def _create_layers(self, input_dim, output_dim, hidden_layers, dropout_rate):
        layers = []
        in_dim = input_dim

        # Define hidden layers with activation and dropout
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim

        # Define output layer
        layers.append(nn.Linear(in_dim, output_dim))
        return layers

    def forward(self, x):
        # Flatten input tensor
        x = x.view(x.size(0), -1)
        return self.network(x)


# Hyperparameters to experiment with
config = {
    "learning_rate": 0.01,
    "epochs": 10,
    "batch_size": 64,
    "hidden_layers": [128, 64],  # Example: two hidden layers
    "dropout_rate": 0.2
}
wandb.login(key="6b99631c072440566d571022bbe19279e7b0e09f")
# Initialize wandb
wandb.init(project="mnist-mlops", config=config)

# Load and preprocess data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

# Initialize model, loss, and optimizer
model = LogisticRegression(input_dim=28*28, output_dim=10, hidden_layers=config['hidden_layers'], dropout_rate=config['dropout_rate'])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

wandb.watch(model, log="all")

# Visualize sample predictions
def log_predictions(model, data, target, num_samples=10):
    # FIXME: Implement this function to log prediction samples
    model.eval()
    with torch.no_grad():
        output = model(data[:num_samples])
        pred = output.argmax(dim=1, keepdim=True)
        wandb.log({"Sample Predictions": [wandb.Image(data[i], caption=f"Pred: {pred[i].item()}, Truth: {target[i].item()}") for i in range(num_samples)]})


# Training loop
for epoch in range(config["epochs"]):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # FIXME: Implement the training step
        # Hint: Remember to log the loss with wandb
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1}/{config['epochs']} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item()}")

        # Log loss to wandb
        wandb.log({
        "loss": loss.item(),
        "learning_rate": optimizer.param_groups[0]['lr'],
        "gradient_norm": sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        })

    avg_loss = total_loss / len(train_loader)
    wandb.log({"Epoch Training Loss": avg_loss, "epoch": epoch + 1})

    # Validation
    model.eval()
    correct = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        # FIXME: Implement the validation step
        # Hint: Remember to log the accuracy with wandb

        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_preds.extend(pred.squeeze().cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    accuracy = 100 * correct / len(test_loader.dataset)
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    
    wandb.log({
        "Validation Accuracy": accuracy,
        "Validation Precision": precision,
        "Validation Recall": recall,
        "epoch": epoch + 1
    })

    # Log predictions at the end of each epoch
    sample_data, sample_target = next(iter(test_loader))
    log_predictions(model, sample_data, sample_target)

wandb.finish()

torch.save(model.state_dict(), "mnist_model.pth")
print("Model saved as mnist_model.pth")
