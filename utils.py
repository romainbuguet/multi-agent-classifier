import os
from dotenv import load_dotenv

def initialize_client(use_ollama: bool):
  """
  Initialize a language model client, either OpenAI or Ollama.

  Args:
      use_ollama (bool): Whether to use Ollama (local LLM).

  Returns:
      client: Initialized OpenAI or Ollama client instance.
  """
  if use_ollama:
    try:
      from ollama import Client as OllamaClient
    except ImportError:
      raise ImportError("The 'ollama' package must be installed to use local LLMs")

    client = OllamaClient(host="http://localhost:11434")
    print("Ollama client initialized.")
  else:
    try:
      from openai import OpenAI
    except ImportError:
      raise ImportError("The 'openai' package must be installed to use OpenAI")

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

    client = OpenAI(api_key=api_key)
    print("OpenAI client initialized.")
    
  return client

def generate_llm_response(prompt: str, 
                          use_ollama: bool,
                          model: str, 
                          temperature: float,
                          client,
                          ):
  """
  Generates a response from the specified LLM (OpenAI or Ollama).

  Args:
      prompt (str): Input text for the LLM.
      use_ollama (bool): Whether to use Ollama (local).
      client: Ollama or OpenAI client.
      model (str): Model name.
      temperature (float): Sampling temperature.

  Returns:
      str: Model-generated response content.
  """
  if use_ollama:

      response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature}
      )
      content = response['message']['content'].strip()
  else:
      response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
      content = response.output_text
  
  return content
    
if __name__ == '__main__':
  from argparse import ArgumentParser
  parser = ArgumentParser()
  parser.add_argument("--use_ollama", action="store_true", help="Whether to use Ollama (local LLM).")
  parser.add_argument("--model", type=str, help="LLM name.")
  parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
  args = parser.parse_args()

  client = initialize_client(args.use_ollama)
    
  content = generate_llm_response(
    prompt="Tell me a joke about bears",
    use_ollama=args.use_ollama,
    model=args.model,
    temperature=args.temperature,
    client=client,
  )

  print(content)



  
