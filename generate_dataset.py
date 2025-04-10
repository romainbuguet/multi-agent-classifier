import os
import random
from tqdm import tqdm
from argparse import ArgumentParser
import pandas as pd
from utils import initialize_client, generate_llm_response

def generate_sample(category: str, 
                    use_ollama: bool,
                    model: str, 
                    temperature: float,
                    client,
                    ):
    """
    Generates a negative customer review using Ollama or the OpenAI API.

    Args:
        category (str): Either 'constructive' or 'unconstructive'.
        use_ollama (bool): Use a local LLM via Ollama instead of OpenAI.
        client (OpenAI): OpenAI client instance.
        model (str): OpenAI model name.
        temperature (float): Sampling temperature.

    Returns:
        str: The generated review.
    """
    prompt = f"""
        Write a short, realistic negative customer review that would be categorized as "{category}".
        Make it sound like it could be posted online, and avoid repeating phrases.

        Review:
        """
    content = generate_llm_response(prompt, use_ollama, model, temperature, client)                  
    
    return content

def generate_dataset(data_dir: str,
                     n_samples: int, 
                     use_ollama: bool, 
                     model: str, 
                     temperature: float, 
                     client,
                     ):
    # Generate data
    data = []
    for _ in tqdm(range((n_samples))):
        category = random.choice(['constructive', 'unconstructive'])
        review = generate_sample(category, use_ollama, model, temperature, client)
        data.append({'review': review, 'category': category}) 

    # Save data as a CSV file
    df = pd.DataFrame(data)
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(os.path.join(data_dir, "dataset.csv"), index=False)
    print(f"Dataset saved to '{os.path.join(args.data_dir, 'dataset.csv')}'") 

    return None

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Directory where to save the dataset CSV file.")
    parser.add_argument("--n_samples", type=int, default=50, help="Number of samples to generate.")
    parser.add_argument("--use_ollama", action="store_true", help="Use a local LLM via Ollama instead of OpenAI.")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model to use (OpenAI or Ollama).")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    args = parser.parse_args()

    # Initialize client
    client = initialize_client(args.use_ollama)

    # Generate dataset
    generate_dataset(args.data_dir, args.n_samples, args.use_ollama, args.model, args.temperature, client)

