
from datasets import load_dataset 
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class CustomDataset(Dataset):
    def __init__(self, name, tokenizer, max_length = 512, split = "train"):
        super().__init__()

        self.name = name
        self.tokenizer = tokenizer
        self.max_length = max_length
        # print(name)
        self.data = load_dataset(str(name))[split]

    
    def getitem(self, index):
        text = self.data["sentence"][index]
        inputs = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_length)

        return {
            "input_ids": torch.tensor(inputs["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(inputs["attention_mask"], dtype=torch.long),
            "label": torch.tensor(self.data["label"], dtype=torch.long)
        }
    
def dataloader(cfg, split = "train"):
    tokenizer = BertTokenizer.from_pretrained(cfg["MODEL"]["NAME"])
    dataset = CustomDataset(cfg["DATA"]["NAME"], tokenizer, max_length = cfg["DATA"]["MAX_LENGTH"], split = split)
    dataloader = DataLoader(dataset, batch_size = cfg["DATA"]["BATCH_SIZE"])

    return dataloader

