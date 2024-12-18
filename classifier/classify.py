import csv

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertForSequenceClassification


class NewsDataPreparation:
    def __init__(self, file_path, sample_size=250, test_size=0.2, random_state=42):
        self.file_path = file_path
        self.sample_size = sample_size
        self.test_size = test_size
        self.random_state = random_state
        self.data = self._load_data()
        self.titles, self.labels = self._extract_titles_labels()
        (
            self.titles_train,
            self.titles_test,
            self.labels_train,
            self.labels_test,
        ) = self._train_test_split()

    def _load_data(self):
        data = []
        with open(self.file_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        return data

    def _extract_titles_labels(self):
        titles = [item["title"] for item in self.data][:self.sample_size]
        labels = [1 if item["label"].lower() == "real" else 0 for item in self.data][:self.sample_size]
        return titles, labels

    def _train_test_split(self):
        return train_test_split(
            self.titles,
            self.labels,
            test_size=self.test_size,
            random_state=self.random_state,
        )


class TokenizedDataset(Dataset):
    def __init__(self, titles, labels, tokenizer):
        self.tokens = tokenizer(titles, padding=True, truncation=True)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {k: torch.tensor(v[idx]) for k, v in self.tokens.items()}
        sample["labels"] = torch.tensor(self.labels[idx])
        return sample


class NewsModelTrainer:
    def __init__(
        self, model_name, train_data, test_data, batch_size=40, lr=1e-5, num_epochs=3
    ):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
        self.test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.num_epochs = num_epochs
        self.model.to(self.device)
        self.best_accuracy = 0.0
        self.save_path = "bert_ro_best_model.pt"

    def train(self):
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            self.model.train()
            train_loss = 0.0
            for i, batch in enumerate(self.train_loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self.optimizer.zero_grad()
                outputs = self.model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                )
                loss = self.loss_fn(outputs.logits, batch["labels"])
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                print(
                    f"Training batch {i + 1} loss: {loss.item() / len(batch['labels'])}"
                )
            print(
                f"Epoch {epoch + 1} training loss: {train_loss / len(self.train_loader)}"
            )
            self.evaluate(epoch + 1)

    def evaluate(self, epoch):
        self.model.eval()
        correct = 0
        total_loss = 0.0
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                )
                loss = self.loss_fn(outputs.logits, batch["labels"])
                total_loss += loss.item()
                correct += (outputs.logits.argmax(1) == batch["labels"]).sum().item()
                print(
                    f"Testing batch {i + 1} loss: {loss.item() / len(batch['labels'])}"
                )
            accuracy = correct / len(self.test_loader.dataset)
            print(
                f"Testing loss: {total_loss / len(self.test_loader)}, Accuracy: {accuracy}"
            )

        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            torch.save(self.model.state_dict(), self.save_path)
            print(f"New best model saved with accuracy: {accuracy:.4f} at epoch {epoch}")


if __name__ == "__main__":
    data_prep = NewsDataPreparation(
        "./data/labeled_news_dataset.csv", sample_size=150
    )
    tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")
    train_dataset = TokenizedDataset(
        data_prep.titles_train,
        data_prep.labels_train,
        tokenizer=tokenizer,
    )
    test_dataset = TokenizedDataset(
        data_prep.titles_test,
        data_prep.labels_test,
        tokenizer=tokenizer,
    )

    trainer = NewsModelTrainer(
        "dumitrescustefan/bert-base-romanian-cased-v1", train_dataset, test_dataset, num_epochs=20
    )
    trainer.train()
