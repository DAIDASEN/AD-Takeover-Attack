import pandas as pd
import os
import requests
import re
from tqdm import tqdm
import json

# --- LLM API Configuration ---
API_KEY = "sk-50500a47b4074df5941535558180493c"
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
LLM_MODEL = "qwen-plus"

def generate_llm_prompt(row):
    """
    Generates a structured prompt for an LLM to evaluate driving scenarios.
    """
    
    # The base prompt template
    base_prompt = """You are an expert AI assistant for evaluating autonomous driving scenarios. Your task is to analyze a series of timestamped actions and their justifications from a vehicle's driving log to determine if any action constitutes a "difficult scenario" requiring a human driver to take over.

A "Takeover" is required if the situation involves:
- **Sudden, Unexpected Events**: Actions like sudden swerving, hard braking, or abrupt maneuvers to avoid obstacles (e.g., other vehicles cutting in, pedestrians, animals, debris).
- **High Environmental Complexity**: Navigating through heavy, unpredictable traffic, construction zones, or dealing with emergency vehicles.
- **Adverse Conditions**: Driving in challenging weather (heavy rain, snow, fog) or on poor road surfaces (potholes, ice).
- **Ambiguous or Unsafe Situations**: When the vehicle's action is due to an "unknown reason," involves risky behavior (e.g., speeding on the wrong side of the road), or shows uncertainty.
- **Interaction with Vulnerable Road Users**: Close encounters with pedestrians, bicyclists, or motorcyclists that require exceptionally careful navigation.

A "Takeover" is NOT required for standard, predictable driving maneuvers, such as:
- Normal acceleration, deceleration, and stopping for traffic lights or stop signs.
- Routine lane changes in clear traffic.
- Following the flow of traffic.
- Standard parking or turning maneuvers.

Analyze the following list of actions from a single driving log. For each action, decide if it requires a human takeover.

If you determine that one or more actions require a takeover, provide your response ONLY in the following format for EACH such action:
Decision: Takeover
Time: [start_time]s - [end_time]s
Action: [The action description]
Reason: [Your summary of why this specific action requires a takeover based on the provided justification and the criteria.]

If, after analyzing ALL actions, you find that NONE of them require a takeover, respond with a single line:
Decision: No Takeover

Here is the driving log:
"""

    # Append the actions and justifications from the current row
    driving_log = ""
    for i in range(1, 16):  # Assuming up to 15 answer columns
        start_col = f'Answer.{i}start'
        end_col = f'Answer.{i}end'
        action_col = f'Answer.{i}action'
        justification_col = f'Answer.{i}justification'

        if action_col in row and pd.notna(row[action_col]):
            start_time = row.get(start_col, 'N/A')
            end_time = row.get(end_col, 'N/A')
            action = row[action_col]
            justification = row.get(justification_col, 'No justification provided.')
            
            driving_log += f"- Time: {start_time}s - {end_time}s, Action: {action}, Justification: {justification}\n"

    if not driving_log:
        return None # Skip rows with no action data

    return base_prompt + driving_log


def call_llm_api(prompt):
    """
    Sends a prompt to the LLM API using requests and returns the response.
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You are an expert AI assistant for evaluating autonomous driving scenarios. Answer strictly in English and in the required format only."},
            {"role": "user", "content": prompt}
        ]
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        return result['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print(f"An API error occurred: {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"Error parsing API response: {e}")
        return None

def parse_llm_response(response_text):
    """
    Parses the LLM's response to extract takeover events.
    """
    if not response_text or "Decision: No Takeover" in response_text:
        return []

    # Use a more robust regex to find all takeover blocks, handling variations
    pattern = re.compile(r"Decision: Takeover\s*\nTime: (.*?)\s*\nAction: (.*?)\s*\nReason: (.*?)(?=\nDecision: Takeover|\Z)", re.DOTALL)
    matches = pattern.findall(response_text)
    
    takeover_events = []
    for match in matches:
        takeover_events.append({
            "Time": match[0].strip(),
            "Action": match[1].strip(),
            "Reason": match[2].strip()
        })
    return takeover_events

def main():
    """
    Main function to read CSV, process with LLM, and save results.
    """
    input_file = r'./BDD-X-Dataset/BDD-X-Annotations_v1.csv'
    output_file = r'./BDD-X-Dataset/takeover_analysis_results.csv'

    # --- IMPORTANT: For testing, process only a subset of the data ---
    # For a full run, comment out or change the value of the next line
    NUM_ROWS_TO_PROCESS = 50

    try:
        df = pd.read_csv(input_file, low_memory=False)
        
        # If testing, take a slice of the dataframe
        if 'NUM_ROWS_TO_PROCESS' in locals() and NUM_ROWS_TO_PROCESS > 0:
            df = df.head(NUM_ROWS_TO_PROCESS)
            print(f"--- RUNNING IN TEST MODE: Processing first {NUM_ROWS_TO_PROCESS} rows. ---")

        all_takeover_events = []

        # Use tqdm for a progress bar
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
            prompt = generate_llm_prompt(row)
            if not prompt:
                continue

            llm_response = call_llm_api(prompt)
            if llm_response:
                events = parse_llm_response(llm_response)
                for event in events:
                    # Add original video and row index for reference
                    video = row.get("Input.Video", "")
                    event['Video'] = video
                    event['Original_Row_Index'] = index
                    all_takeover_events.append(event)
        
        # Create a new DataFrame from the results and save to CSV
        if all_takeover_events:
            results_df = pd.DataFrame(all_takeover_events)
            # Ensure columns are in a logical order
            results_df = results_df[['Video', 'Original_Row_Index', 'Time', 'Action', 'Reason']]
            results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"\nProcessing complete. Found {len(all_takeover_events)} takeover events.")
            print(f"Results saved to {output_file}")
        else:
            print("\nProcessing complete. No takeover events were identified.")

    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()