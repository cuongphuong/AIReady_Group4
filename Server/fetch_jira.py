import os
import requests
from dotenv import load_dotenv
import json
from requests.auth import HTTPBasicAuth

# Load environment variables
load_dotenv()

JIRA_BASE_URL = os.getenv("JIRA_BASE_URL")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_TOKEN = os.getenv("JIRA_TOKEN")

def fetch_jira_issues():
    if not JIRA_BASE_URL or not JIRA_EMAIL or not JIRA_TOKEN:
        print("Error: Missing JIRA configuration in .env file.")
        print("Please ensure JIRA_BASE_URL, JIRA_EMAIL, and JIRA_TOKEN are set.")
        return

    if JIRA_EMAIL == "your_email@example.com":
        print("Error: Please update JIRA_EMAIL in .env with your actual Jira email address.")
        return

    # Updated endpoint based on error message
    url = f"{JIRA_BASE_URL}/rest/api/3/search/jql"
    
    # JQL query provided by user
    jql = 'project = "My Software Team" and assignee = "5dd206cbaf96bc0efbe4ff34"'
    
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    
    # Note: The /search/jql endpoint might have different payload requirements
    # but usually it accepts 'jql' in the body.
    # If this is strictly the JQL search endpoint, it might just take the JQL string or a specific object.
    # Let's try keeping the payload structure first as it's standard for search.
    payload = {
        'jql': jql,
        'fields': ['summary', 'description', 'status', 'priority', 'assignee', 'created'],
        'maxResults': 50
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            auth=HTTPBasicAuth(JIRA_EMAIL, JIRA_TOKEN)
        )
        
        if response.status_code == 200:
            data = response.json()
            issues = data.get("issues", [])
            print(f"Found {len(issues)} issues:\n")
            
            for issue in issues:
                key = issue.get("key")
                fields = issue.get("fields", {})
                summary = fields.get("summary")
                status = fields.get("status", {}).get("name")
                
                print(f"[{key}] {summary} ({status})")
                # Uncomment to see description
                # description = fields.get("description")
                # print(f"Description: {description}\n")
                
        else:
            print(f"Failed to fetch issues. Status code: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    fetch_jira_issues()
