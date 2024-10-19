import os
import json
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from datasets import Dataset, load_dataset
from ragatouille import RAGTrainer

DATA_DIR = os.environ.get("DATA_DIR", "./processed-data")

def get_data_ready(ds):
    pairs = []
    full_corpus = []
    
    for sample in tqdm(ds):
        pairs.append(
            (sample["query"], sample["information"])
        )
        full_corpus.append(sample["information"])
    
    assert len(pairs) == len(ds)

    return pairs, full_corpus


if __name__=="__main__":
    print(f"Loading dataset:")
    ds = load_dataset("abhinand/clini-colbert-pairs-v0.2-dev", split="train")
    print(ds)

    trainer = RAGTrainer(
        model_name="Med-ColBERT", 
        pretrained_model_name="colbert-ir/colbertv2.0", 
        language_code="en"
    )

    os.makedirs(DATA_DIR, exist_ok=True)
    
    if os.listdir(DATA_DIR) == 0:
        print(f"DATA_DIR is empty, preparing training data & creating triplets!")
        pairs, full_corpus = get_data_ready(ds)
        
        trainer.prepare_training_data(
            raw_data=pairs,
            data_out_path=DATA_DIR,
            all_documents=full_corpus,
            num_new_negatives=10,
            mine_hard_negatives=True
        )
    else:
        print(f"Loading data from DATA_DIR -> {DATA_DIR}")
        trainer.data_dir = Path(DATA_DIR)

    print("------------- Starting Training -------------")
    trainer.train(
        batch_size=64,
        nbits=4, # How many bits will the trained model use when compressing indexes
        maxsteps=500000, # Maximum steps hard stop
        use_ib_negatives=True, # Use in-batch negative to calculate loss
        dim=128, # How many dimensions per embedding. 128 is the default and works well.
        learning_rate=5e-6, # Learning rate, small values ([3e-6,3e-5] work best if the base model is BERT-like, 5e-6 is often the sweet spot)
        doc_maxlen=256, # Maximum document length. Because of how ColBERT works, smaller chunks (128-256) work very well.
        use_relu=False, # Disable ReLU -- doesn't improve performance
        warmup_steps="auto", # Defaults to 10%
     )
    print("------------- Training Complete! -------------")

    

        

    