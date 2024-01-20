import pandas as pd
from graph_data_set import GraphDataset
from heuristic_predictor import HeuristicPredictor
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


class TrainModel:
    def __init__(self):
        self.csv_file_path = "../output2.csv"
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.X = None
        self.y = None
        self.model = None
        pass

    def load_data(self):
        dataset = pd.read_csv(self.csv_file_path)
        heuristics = dataset["best_heuristic"].str.get_dummies(",")
        data = pd.concat([dataset, heuristics], axis=1)

        self.X = data.drop(
            columns=[
                "Instances",
                "best_heuristic",
                "Results",
                "afisa_original",
                "dsatur",
                "ilsts",
                "redls",
                "afisa",
                "tabu_weight",
            ],
        )

        self.y = torch.tensor(
            data.iloc[:, -5:].values, dtype=torch.float32
        )  # Assuming the binary columns are the last 5 columns

        # random olmucak
        X_train, X_temp, y_train, y_temp = train_test_split(
            self.X, self.y, test_size=0.6, shuffle=False  # random_state=42
        )
        print(X_train)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def create_datasets(self):
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_data()

        train_dataset = GraphDataset(X_train, y_train)
        val_dataset = GraphDataset(X_val, y_val)
        test_dataset = GraphDataset(X_test, y_test)

        return train_dataset, val_dataset, test_dataset

    def create_dataloaders(self):
        train_dataset, val_dataset, test_dataset = self.create_datasets()
        self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    def train_model(self, save_path="model_checkpoint.pth"):
        self.create_dataloaders()

        self.model = HeuristicPredictor(
            input_size=self.X.shape[1], hidden_size=64, output_size=len(self.y[1])
        )
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        num_epochs = 300
        for epoch in range(num_epochs):
            self.model.train()
            for inputs, labels in self.train_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for inputs, labels in self.val_loader:
                    outputs = self.model(inputs)
                    val_loss += criterion(outputs, labels).item()

            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Loss: {val_loss / len(self.val_loader)}"
            )
            if val_loss / len(self.val_loader) < 2:
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": val_loss / len(self.val_loader),
                }
                torch.save(checkpoint, save_path)
                break

    def test_model(self):
        self.model.eval()

        all_predictions = []
        all_labels = []

        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                outputs = self.model(inputs)

                predictions = torch.sigmoid(outputs) > 0.7

                all_predictions.append(predictions.numpy())
                _, predicted = torch.max(outputs.data, 1)

                label_enum = ["afisa", "dsatur", "ilsts", "redls", "tabu_weight"]
                # for i in range(len(labels)):
                #     counter = 0
                #     single_result = ""
                #     for j in range(len(labels[i])):
                #         if labels[i][j] == 1:
                #             counter += 1
                #             single_result += label_enum[j]
                #     if counter == 1 or counter == 2:
                #         total += 1
                #         print("Actual:", single_result)
                #         print("Predicted:", label_enum[predicted[i]], end="\n\n")

                #     if (counter == 1 or counter == 2) and labels[i][predicted[i]] == 1:
                #         correct += 1

        all_predictions = np.vstack(all_predictions)
        all_labels = np.vstack(all_labels)

        confusion_matrices = multilabel_confusion_matrix(all_labels, all_predictions)

        target = torch.sigmoid(labels) > 0.7
        target = np.vstack(target.numpy())

        counter = 0
        for i in range(len(target)):
            single_result = ""
            for j in range(len(target[i])):
                if target[i][j] == all_predictions[i][j]:
                    counter += 1
                    # single_result += label_enum[j]
                    # print(label_enum[j], end=" ")

        print(target, end="\n\n")
        print(all_predictions)

        accuracy = counter / (len(target) * 5) * 100
        print("Total :", len(target) * 5)
        print("Correct values:", counter)
        print(f"Test Accuracy: {accuracy}")
        # print("Counter: ", counter)
        # print("Total: ", len(target) * 5)
        matching_rows = np.all(target == all_predictions, axis=1)

        # Count the number of matching rows
        counter = np.sum(matching_rows)

        # print(target)
        # print(all_predictions)
        # print(all_labels)

        # for i, confusion_matrix in enumerate(confusion_matrices):
        #     labels = [f"Heuristic_{i}" for i in range(confusion_matrix.shape[0])]
        #     df_cm = pd.DataFrame(
        #         confusion_matrix,
        #         index=labels,
        #         columns=[f"Predicted_{i}" for i in range(confusion_matrix.shape[1])],
        #     )
        #     plt.figure(figsize=(7, 5))
        #     sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
        #     plt.title(f"Confusion Matrix for Heuristic_{i}")
        #     plt.show()

        # accuracy = correct / total * 100
        # print("Total single values:", total)
        # print("Correct single values:", correct)
        # print(f"Test Accuracy: {accuracy}")


graph_predictor = TrainModel()
graph_predictor.train_model()
graph_predictor.test_model()

# # Step 1: Data Preparation
# # Load your CSV file into a Pandas DataFrame


# # Extract features and labels
# X = data.drop(columns=['Instances', 'best_heuristic', "Results", "afisa_original", "dsatur", "ilsts", "redls", "afisa", "tabu_weight" ])
# y = data.iloc[:, -5:]  # Assuming the binary columns are the last 5 columns
# # Convert labels to tensor
# y = torch.tensor(y.values, dtype=torch.float32)

# #   afisa dsatur ilsts  redls  tabu_weight
# #     0     1      2      3        4


# # Split the data into training, validation, and test sets

# # print(y_train)

# # Step 2: Create PyTorch Dataset


# # Create datasets and dataloaders
# train_dataset = GraphDataset(X_train, y_train)
# val_dataset = GraphDataset(X_val, y_val)
# test_dataset = GraphDataset(X_test, y_test)

# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# # Step 3: Define the Neural Network Model

# # Step 4: Loss Function and Optimizer

# model = HeuristicPredictor(input_size=X.shape[1], hidden_size=64, output_size=len(y[1]))
# criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)


# # Step 5: Training Loop
# num_epochs = 10
# for epoch in range(num_epochs):
#     model.train()
#     for inputs, labels in train_loader:
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#     # Validation
#     model.eval()
#     with torch.no_grad():
#         val_loss = 0.0
#         for inputs, labels in val_loader:
#             outputs = model(inputs)
#             val_loss += criterion(outputs, labels).item()

#     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {val_loss / len(val_loader)}')

# # Step 6: Evaluation
# model.eval()
# correct = 0
# total = 0
# with torch.no_grad():
#     for inputs, labels in test_loader:
#         outputs = model(inputs)
#         # print(inputs.data.shape)
#         print(outputs.data)
#         _, predicted = torch.max(outputs.data, 1)

#         total += labels.size(0)
#         print(torch.max(outputs.data, 1))
#         print(labels)

#         for i in range(len(labels)):
#             if labels[i][predicted[i]] == 1:
#                 correct += 1

# accuracy = correct / total *100
# print(f'Test Accuracy: {accuracy}')

# # Step 7: Inference
# # Use the trained model for predicting the heuristic for new graphs
# # Example: new_graph_features = torch.tensor(new_graph_features, dtype=torch.float32)
# # predicted_heuristic = model(new_graph_features)


#   self.X = data.drop(
#             columns=[
#                 "Instances",
#                 "best_heuristic",
#                 "Results",
#                 "afisa_original",
#                 "dsatur",
#                 "ilsts",
#                 "redls",
#                 "afisa",
#                 "tabu_weight",
#             ],
#         )
#         self.y = torch.tensor(data.iloc[:, -6:-1].values, dtype=torch.float32)

# with open("../DIMACS_hard.txt", "r") as file:
#     hard_instances = [line.strip() for line in file.readlines()]

# data["is_test_instance"] = data["Instances"].isin(hard_instances)

# train_data = data[data["is_test_instance"] == False]
# test_data = data[data["is_test_instance"] == True]

# # Training data (DIMACS_easy)
# X_train = train_data.drop(
#     columns=[
#         "is_test_instance",
#         "Instances",
#         "best_heuristic",
#         "Results",
#         "afisa_original",
#         "dsatur",
#         "ilsts",
#         "redls",
#         "afisa",
#         "tabu_weight",
#     ],
# )
# print(train_data)
# print(test_data)
# y_train = torch.tensor(train_data.iloc[:, -6:-1].values, dtype=torch.float32)

# # Test and Validate data (DIMACS_hard)
# X_temp = test_data.drop(
#     columns=[
#         "is_test_instance",
#         "Instances",
#         "best_heuristic",
#         "Results",
#         "afisa_original",
#         "dsatur",
#         "ilsts",
#         "redls",
#         "afisa",
#         "tabu_weight",
#     ],
# )
# y_temp = torch.tensor(test_data.iloc[:, -6:-1].values, dtype=torch.float32)

# X_val, X_test, y_val, y_test = train_test_split(
#     X_temp, y_temp, test_size=0.7, random_state=42
# )

# print(train_data)
# print(test_data)
