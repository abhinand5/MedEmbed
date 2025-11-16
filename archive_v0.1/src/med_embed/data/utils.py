import os
import json
import nanoid
import pandas as pd
from typing import List, Dict
from datasets import Dataset, load_dataset, concatenate_datasets
from openai import OpenAI

def load_asclepius_data(dataset_name: str, num_samples: int, random_state: int) -> pd.DataFrame:
    dataset = load_dataset(dataset_name)
    df = pd.DataFrame(dataset['train'])
    return df.sample(n=num_samples, random_state=random_state) if num_samples else df

def setup_openai_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key)

def test_openai_api(client: OpenAI, model_name: str) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"},
            {"role": "assistant", "content": "The LA Dodgers won in 2020."},
            {"role": "user", "content": "Where was it played?"}
        ]
    )
    return response.choices[0].message.content

def parse_qa_pairs(json_str: str) -> List[Dict]:
    try:
        json_str = json_str.replace("```json", "").replace("```", "")
        data = json.loads(json_str)
        return data.get("qa_pairs", [])
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return []

def create_training_pairs_ds(client: OpenAI, model_name: str, sample: Dict) -> Dict:
    note = sample['note']
    json_output = generate_questions_and_answers(client, model_name, note)
    qa_pairs = parse_qa_pairs(json_output)
    
    sample["raw_output"] = json_output
    sample["qa_pairs"] = qa_pairs
    return sample

def add_summary_id(sample: Dict) -> Dict:
    sample["id"] = nanoid.generate()
    return sample

def process_dataset(ds: Dataset) -> Dataset:
    ds = ds.map(add_summary_id)
    ds_faulty = ds.filter(lambda x: len(x["qa_pairs"]) == 0)
    print(f"[WARNING] {len(ds_faulty)} faulty samples found in the dataset. You might need to fix those.")
    return ds

def create_pairs_dataset(ds: Dataset) -> Dataset:
    pairs_data = []
    for sample in ds:
        pairs = sample["qa_pairs"]
        for data in pairs:
            data["note_id"] = sample["id"]
            data["patient_id"] = sample["patient_id"]
            pairs_data.append(data)
    return Dataset.from_list(pairs_data)

def generate_questions_and_answers(client: OpenAI, model_name: str, note: str) -> str:
    prompt = f"""You are an advanced medical AI assistant tasked with generating diverse, realistic queries and extracting relevant information from clinical notes. Your goal is to create high-quality query-information pairs that simulate real-world medical information retrieval scenarios.

Given the following clinical note, please:

1. Generate 6 diverse queries that represent a range of information needs in medical/clinical domains. These should include:
   - 2 keyword-based queries (e.g., "interatrial septal mass symptoms")
   - 2 natural language questions (e.g., "What was the patient's main complaint?")
   - 2 queries related to treatment or procedure or follow-up (e.g., "post-operative care plan")

2. For each query, extract the most relevant information from the note. This can be:
   - 2 to 3 sentences
   - A larger chunk if the information is spread across the note 

3. Ensure that the queries cover different aspects of the patient's condition, diagnostic procedures, treatment, outcomes, and follow-up care.

4. Prioritize medically relevant information and maintain clinical accuracy.

5. Include queries that might not have a direct answer in the note, but for which the note contains related information.

6. If a query's relevance is too low (<0.5), do NOT consider adding them in the final JSON. Save tokens!

Clinical Note:
{note}

Generate the query-information pairs in the following JSON format:

{{
  "qa_pairs": [
    {{
      "query": "query 1",
      "query_type": "keyword|natural_language|procedure",
      "information": "Relevant information extracted from the note",
      "relevance_score": 0.0, # this should in range 0 to 1
    }},
    {{
      "query": "query 2",
      "query_type": "keyword|natural_language|procedure",
      "information": "Relevant information extracted from the note",
      "relevance_score": 0.0, # this should in range 0 to 1
    }},
    {{
      "query": "query 3",
      "query_type": "keyword|natural_language|procedure",
      "information": "Relevant information extracted from the note",
      "relevance_score": 0.0, # this should in range 0 to 1
    }},
  ]
}}

The relevance_score should indicate how directly the extracted information answers the query, with 1.0 being a perfect match and lower scores for partial or related information.

Your response must contain ONLY the valid JSON output and NOTHING ELSE."""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": """You are a highly skilled medical AI assistant specializing in analyzing clinical notes and generating relevant retrieval questions with precise answers. Your expertise includes:
    1. Understanding complex medical terminology and concepts
    2. Identifying key information in clinical notes
    3. Formulating diverse and clinically relevant questions
    4. Extracting the most appropriate answer chunks, from single sentences to larger contexts
    Always respond in valid JSON format, ensuring your output is directly usable for training data generation."""},
            {"role": "user", "content": prompt}
        ],
        max_tokens=4096,
        n=1,
        stop=None,
        temperature=0.8,
    )

    return response.choices[0].message.content