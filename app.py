import os
import json
from tqdm import tqdm
from typing import List
from argparse import ArgumentParser
from dotenv import load_dotenv
import pandas as pd
import yaml
from openai import OpenAI
import streamlit as st

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

# Initialize client
client = initialize_client(use_ollama=True)

# Load config
config = load_config("config.yaml")

# Streamlit UI
st.set_page_config(page_title="Agent Feedback Classifier", page_icon="🧠")
st.title("Multi-Agent Feedback Classifier")
st.markdown("Classify a negative customer review as **constructive** or **unconstructive** using agent perspectives.")

review = st.text_area("Enter a customer review:")

if st.button("Classify"):
    if not review.strip():
        st.warning("Please enter some text to classify.")
    else:
        st.info("Running classification...")
        agent_outputs = []
        for agent in config['agents']:
            output = ask_agent(agent['name'], agent['style'], review, use_ollama=True, model="llama3.2:1b", temperature=0.7, client=client)
            agent_outputs.append(output)
            st.markdown(f"**{agent['name'].capitalize()}** says:\n```\n{output}\n```")

        final = meta_decision(agent_outputs, review, use_ollama=True, model="llama3.2:1b", temperature=0.7, client=client)
        st.markdown("---")
        st.subheader("Final Decision")
        st.markdown(f"**Judgment:** `{final['judgment']}`")
        st.markdown(f"**Justification:** {final['justification']}")
