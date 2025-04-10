import os
import json
from tqdm import tqdm
from typing import List
from argparse import ArgumentParser
from dotenv import load_dotenv
import pandas as pd
import yaml
from openai import OpenAI

from utils import initialize_client, generate_llm_response

def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def ask_agent(agent_name: str, 
              style: str, 
              text: str, 
              use_ollama: bool,
              model: str,
              temperature: float,
              client,
              ):
    prompt = f"""
    You are a {agent_name} reviewing a customer feedback. Your task is to evaluate whether the feedback is constructive or unconstructive.

    Text:
    "{text}"

    Your perspective:
    {style}

    Answer in the format:
    Judgment: [constructive/unconstructive]
    Reason: <your reasoning>
    """
    content = generate_llm_response(prompt, use_ollama, model, temperature, client)
    return content

def meta_decision(agent_responses: List[str], 
                  text: str,
                  use_ollama: bool, 
                  model: str,
                  temperature: float,
                  client,
                  ):
    joined = "\n\n".join(agent_responses)
    prompt = f"""
    You are a meta-analyst. Three agents have analyzed the same customer review. Your task is to make the final judgment.

    Original text:
    "{text}"

    Agent opinions:
    {joined}

    Make a final classification decision and explain why.
    Respond in JSON format like this:
    {{
    "judgment": "[constructive/unconstructive]"
    "justification": "<your reasoning>"
    }}
    """
    content = generate_llm_response(prompt, use_ollama, model, temperature, client)
    try:
        content = content.strip().strip("```json").strip("```")
        return json.loads(content)
    except json.JSONDecodeError as e:
        print("Failed to decode JSON:", content)
        return {"judgment": "error", "justification": f"JSON error {e}"}

def classify_review(text: str,
             config_path: str,
             use_ollama: bool,
             model: str, 
             agent_temperature: float,
             meta_temperature: float,
             client,
             ):
    config = load_config(config_path)
    agent_responses = []
    for agent in config['agents']:
        name = agent['name']
        style = agent['style']
        print(f"Asking agent: {name}...")
        response = ask_agent(name, style, text, use_ollama, model, agent_temperature, client)
        print(response)
        agent_responses.append(response)
    print("\nMeta-agent decision:")
    final = meta_decision(agent_responses, text, use_ollama, model, meta_temperature, client)
    return final
    
def process_and_save_review_classifications(input_path: str, 
                                            output_path: str, 
                                            config_path: str,
                                            use_ollama: bool,
                                            model: str, 
                                            agent_temperature: float,
                                            meta_temperature: float,
                                            client,
                                            ):
    """
    Loads customer reviews from a CSV, evaluates each for constructiveness,
    and writes the results to a new CSV with added columns.

    Parameters:
        input_path (str): Path to the CSV file containing a 'review' column.
        output_path (str): Path where the classified CSV will be saved.
    """
    # Load CSV to a DataFrame
    df = pd.read_csv(input_path)
    if "review" not in df.columns:
        raise ValueError("Input CSV must contain a 'review' column.")

    # Create empty lists to store results
    judgments = []
    justifications = []

    # Classify reviews
    for review in tqdm(df["review"]):
        review_evaluation = classify_review(review, config_path, use_ollama, model, agent_temperature, meta_temperature, client)
        judgments.append(review_evaluation.get("judgment", "error"))
        justifications.append(review_evaluation.get("justification", "N/A"))

    # Add new columns to DataFrame
    df["constructiveness_label"] = judgments
    df["classification_reason"] = justifications

    # Save the updated DataFrame to a new CSV
    df.to_csv(output_path, index=False)
    print(f"Classification complete. Results saved to {output_path}")
    return None

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing dataset.csv")
    parser.add_argument("--config_path", type=str, default="config.yaml", help="Path to configuration file ('config.yaml' by default)")
    parser.add_argument("--use_ollama", action="store_true", help="Whether to use Ollama (local LLM).")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model to use.")
    parser.add_argument("--agent_temperature", type=float, default=0.7, help="Sampling temperature for the agents.")
    parser.add_argument("--meta_temperature", type=float, default=0.5, help="Sampling temperature for the meta_decision.")
    args = parser.parse_args()
    
    # Initialize client
    client = initialize_client(args.use_ollama)

    process_and_save_review_classifications(input_path=os.path.join(args.data_dir, "dataset.csv"), 
                                            output_path=os.path.join(args.data_dir, "output.csv"), 
                                            config_path=args.config_path, 
                                            use_ollama=args.use_ollama,
                                            model=args.model, 
                                            agent_temperature=args.agent_temperature,
                                            meta_temperature=args.meta_temperature,
                                            client=client
                                            )
