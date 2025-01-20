from load_data import dataloader
from load_config import load_config
from tqdm import tqdm 
import torch 
from loss import SoftTriple
from sklearn.metrics import accuracy_score, classification_report
from model import BERTClassifier
from torch import nn
from transformers import AdamW, get_linear_schedule_with_warmup
def train(model, dataloader, optimizer, scheduler, device, propose_loss = None, alpha = 0):
    model.train()
    for batch in tqdm(dataloader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask).softmax(1)
        if propose_loss is None:
            loss = nn.CrossEntropyLoss()(outputs, labels)
        else:
            loss = alpha * nn.CrossEntropyLoss()(outputs, labels) + (1 - alpha) * propose_loss(outputs, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()
    

def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    actual_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PATH = "config.yaml"

    cfg = load_config(PATH)
    train_loader = dataloader(cfg, split = "train")
    valid_dataset  = dataloader(cfg, split = "validation")
    model = BERTClassifier(cfg["MODEL"]["NAME"], 2)
    model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=cfg["OPTIMIZER"]["LR"])
    total_steps = len(train_loader) * cfg["OPTIMIZER"]["NUM_EPOCHS"]

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


    if cfg["OPTIMIZER"]["PROPOSE"]:
        proposal_criteria = SoftTriple(cfg["OPTIMIZER"]["la"], cfg["OPTIMIZER"]["gamma"], cfg["OPTIMIZER"]["tau"],
                                        cfg["OPTIMIZER"]["margin"], model.module.hidden_size, 2, cfg["OPTIMIZER"]["NUMBER_K"],device = device)
    
    else:
        proposal_criteria = None
    for epoch in range(cfg["OPTIMIZER"]["NUM_EPOCHS"]):
        print(f"Epoch {epoch + 1}")
        train(model, train_loader, optimizer, scheduler, device = device, propose_loss = proposal_criteria, alpha = cfg["OPTIMIZER"]["alpha"])
        accuracy, report = evaluate(model, valid_dataset, device)
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(report)