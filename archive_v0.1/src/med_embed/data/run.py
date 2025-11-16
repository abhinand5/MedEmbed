import yaml
from datasets import Dataset
from utils import (
    load_asclepius_data,
    setup_openai_client,
    test_openai_api,
    create_training_pairs_ds,
    process_dataset,
    create_pairs_dataset
)

def load_config(config_path: str = 'config.yaml') -> dict:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    # Load configuration
    config = load_config()

    # Setup OpenAI client
    client = setup_openai_client(
        base_url=config['openai']['base_url'],
        api_key=config['openai']['api_key']
    )
    
    # Test OpenAI API
    test_result = test_openai_api(client, config['openai']['model_name'])
    print("API Test Result:", test_result)

    # Load and process data
    df = load_asclepius_data(
        dataset_name=config['data']['dataset_name'],
        num_samples=config['data']['num_samples'],
        random_state=config['data']['random_state']
    )
    ds = Dataset.from_pandas(df)

    # Generate questions and answers
    ds = ds.map(
        lambda x: create_training_pairs_ds(client, config['openai']['model_name'], x),
        num_proc=config['processing']['num_proc']
    )

    # Process the dataset
    ds = process_dataset(ds)

    # Create pairs dataset
    ds_pairs = create_pairs_dataset(ds)

    # Save processed datasets
    ds.select_columns(["id", "patient_id", "note"]).push_to_hub(
        config['output']['corpus_hub'],
        "corpus",
        split="train"
    )
    ds_pairs.push_to_hub(config['output']['pairs_hub'], split="train")

if __name__ == "__main__":
    main()