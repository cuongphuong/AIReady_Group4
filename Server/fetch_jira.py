import os
import requests
from dotenv import load_dotenv
import json
from requests.auth import HTTPBasicAuth

def extract_text_from_adf(adf_content):
    """
    Extract plain text from Atlassian Document Format (ADF).
    ADF is a JSON structure used by Jira for rich text fields.
    """
    if not isinstance(adf_content, dict):
        return str(adf_content)
    
    text_parts = []
    
    def extract_from_node(node):
        if isinstance(node, dict):
            node_type = node.get('type')
            
            # Text node
            if node_type == 'text':
                text_parts.append(node.get('text', ''))
            
            # Process content array
            if 'content' in node:
                for child in node['content']:
                    extract_from_node(child)
            
            # Add spacing for paragraphs
            if node_type in ['paragraph', 'heading']:
                text_parts.append(' ')
        elif isinstance(node, list):
            for item in node:
                extract_from_node(item)
    
    extract_from_node(adf_content)
    return ' '.join(text_parts).strip()

# Load environment variables
load_dotenv()

JIRA_BASE_URL = os.getenv("JIRA_BASE_URL")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_TOKEN = os.getenv("JIRA_TOKEN")

def update_jira_issue_fields(issue_key: str, label: str, team: str):
    """
    Update Jira issue fields: Labels and Team (custom field).
    - issue_key: Jira issue key (e.g., "PROJ-123")
    - label: value to add to Labels
    - team: value to set for Team field (custom field)
    """
    if not JIRA_BASE_URL or not JIRA_EMAIL or not JIRA_TOKEN:
        print("Error: Missing JIRA configuration in .env file.")
        return {"error": "Missing JIRA configuration"}

    url = f"{JIRA_BASE_URL}/rest/api/3/issue/{issue_key}"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    
    teamId = "b9bdfa6a-bf22-47a1-b7c7-a866020c4e72"
    if team == "Frontend":
        teamId = "f26bbd8b-f931-4be1-b92d-55d4c89ffc14"
    elif team == "Backend":
        teamId = "b9bdfa6a-bf22-47a1-b7c7-a866020c4e72"
    elif team == "Infra":
        teamId = "612606f6-7633-4283-884c-2d2f4e5d22ca"

    payload = {
        "fields": {
            "labels": [label],
            "customfield_10001": teamId
        }
    }

    try:
        response = requests.put(
            url,
            headers=headers,
            data=json.dumps(payload),
            auth=HTTPBasicAuth(JIRA_EMAIL, JIRA_TOKEN)
        )
        response.raise_for_status()
        print(f"Issue {issue_key} updated successfully with Label={label}, Team={team}")
        return {"success": True}
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return {"error": f"Failed to update issue {issue_key}", "details": http_err.response.text}
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return {"error": str(e)}

def update_jira_issues(issues: list):
    """
    Update nhiều Jira issues với Labels và Team.
    Input: list các dict có dạng:
      {"key": "PROJ-123", "label": "Functional", "team": "Backend"}
    """
    results = []
    for item in issues:
        issue_key = item.get("issue_key")
        label = item.get("label")
        team = item.get("team")

        res = update_jira_issue_fields(issue_key, label, team)
        results.append({"issue": issue_key, "result": res})
    return results

def fetch_jira_issues(jql: str):
    if not JIRA_BASE_URL or not JIRA_EMAIL or not JIRA_TOKEN:
        print("Error: Missing JIRA configuration in .env file.")
        print("Please ensure JIRA_BASE_URL, JIRA_EMAIL, and JIRA_TOKEN are set.")
        return {"error": "Missing JIRA configuration"}

    if JIRA_EMAIL == "your_email@example.com":
        print("Error: Please update JIRA_EMAIL in .env with your actual Jira email address.")
        return {"error": "JIRA_EMAIL not configured"}

    # Per Atlassian API documentation, we should use the /search/jql endpoint with POST
    url = f"{JIRA_BASE_URL}/rest/api/3/search/jql"
    
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    
    payload = json.dumps({
        'jql': jql,
        'fields': ['summary', 'description', 'status', 'priority', 'assignee', 'created'],
        'maxResults': 200
    })

    try:
        response = requests.post(
            url,
            headers=headers,
            data=payload,
            auth=HTTPBasicAuth(JIRA_EMAIL, JIRA_TOKEN)
        )
        
        response.raise_for_status() # Will raise an HTTPError for bad responses (4xx or 5xx)

        data = response.json()
        issues = data.get("issues", [])
        
        formatted_issues = []
        for issue in issues:
            key = issue.get("key")
            fields = issue.get("fields", {})
            summary = fields.get("summary", "")
            description = fields.get("description")
            status = fields.get("status", {}).get("name")
            
            # Extract plain text from description (Jira uses ADF format)
            description_text = ""
            if description:
                if isinstance(description, dict):
                    # ADF (Atlassian Document Format) - extract text from content
                    description_text = extract_text_from_adf(description)
                else:
                    description_text = str(description)
            
            formatted_issues.append({
                "key": key,
                "summary": summary,
                "description": description_text,
                "status": status
            })
        
        return formatted_issues
            
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        try:
            error_details = http_err.response.json()
            print(f"Error details: {error_details}")
            return {"error": f"Failed to fetch issues. Status: {http_err.response.status_code}", "details": error_details}
        except json.JSONDecodeError:
            print(f"Response text: {http_err.response.text}")
            return {"error": f"Failed to fetch issues. Status: {http_err.response.status_code}", "details": http_err.response.text}
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return {"error": f"An unexpected error occurred: {str(e)}"}

if __name__ == "__main__":
    # Example JQL for testing
    test_jql = 'project = "My Software Team" AND status = "To Do"'
    issues_result = fetch_jira_issues(test_jql)
    if "error" in issues_result:
        print(f"Error fetching issues: {issues_result['error']}")
    else:
        print(f"Found {len(issues_result)} issues:\n")
        for issue in issues_result:
            print(f"[{issue['key']}] {issue['summary']} ({issue['status']})")
