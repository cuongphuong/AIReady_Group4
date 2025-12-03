import React, { useState } from 'react';
import ReactDOM from 'react-dom';

export default function JiraImport({ onImport, onCancel }) {
  const [jql, setJql] = useState('project = "My Software Team" AND status = "To Do"');
  const [issues, setIssues] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedIssues, setSelectedIssues] = useState(new Set());

  async function handleFetchIssues() {
    if (!jql.trim()) {
      setError('JQL query cannot be empty.');
      return;
    }
    setLoading(true);
    setError(null);
    setIssues([]);

    try {
      const response = await fetch('http://localhost:8000/jira/fetch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jql }),
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setIssues(data.issues || []);
      if (!data.issues || data.issues.length === 0) {
        setError('No issues found for this query.');
      }
    } catch (e) {
      setError(`Failed to fetch issues: ${e.message}`);
    } finally {
      setLoading(false);
    }
  }

  function handleToggleIssue(issueKey) {
    setSelectedIssues(prev => {
      const newSelection = new Set(prev);
      if (newSelection.has(issueKey)) {
        newSelection.delete(issueKey);
      } else {
        newSelection.add(issueKey);
      }
      return newSelection;
    });
  }
  
  function handleToggleAll() {
    if (selectedIssues.size === issues.length) {
      setSelectedIssues(new Set());
    } else {
      setSelectedIssues(new Set(issues.map(i => i.key)));
    }
  }

  function handleConfirmImport() {
    const bugsToImport = issues
      .filter(issue => selectedIssues.has(issue.key))
      .map(issue => `${issue.key}: ${issue.summary}`);
    
    if (bugsToImport.length > 0) {
      onImport(bugsToImport.join('\n'));
    }
  }

  return ReactDOM.createPortal(
    <div className="modal-overlay" onClick={onCancel}>
      <div className="modal jira-import-modal" role="dialog" aria-modal="true" onClick={(e) => e.stopPropagation()}>
        <div className="modal-title">Import Bugs from Jira</div>
        <div className="modal-body">
          <p>Enter a JQL query to fetch issues from your Jira instance.</p>
          <div className="jira-query-form">
            <textarea
              className="jql-input"
              value={jql}
              onChange={(e) => setJql(e.target.value)}
              placeholder="e.g., project = ABC AND status = 'In Progress'"
              rows="3"
            />
            <button className="btn btn-primary" onClick={handleFetchIssues} disabled={loading}>
              {loading ? 'Fetching...' : 'Fetch Issues'}
            </button>
          </div>

          {error && <div className="jira-error">{error}</div>}

          {issues.length > 0 && (
            <div className="jira-results">
              <div className="jira-results-header">
                <input 
                  type="checkbox"
                  onChange={handleToggleAll}
                  checked={issues.length > 0 && selectedIssues.size === issues.length}
                  title="Select/Deselect All"
                />
                <span>{selectedIssues.size} of {issues.length} selected</span>
              </div>
              <ul className="jira-issue-list">
                {issues.map(issue => (
                  <li key={issue.key} className="jira-issue-item">
                    <input 
                      type="checkbox"
                      checked={selectedIssues.has(issue.key)}
                      onChange={() => handleToggleIssue(issue.key)}
                    />
                    <span className="issue-key">{issue.key}</span>
                    <span className="issue-summary">{issue.summary}</span>
                    <span className="issue-status">{issue.status}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
        <div className="modal-actions">
          <button className="btn btn-secondary" onClick={onCancel}>Cancel</button>
          <button 
            className="btn btn-primary" 
            onClick={handleConfirmImport}
            disabled={selectedIssues.size === 0}
          >
            Import Selected ({selectedIssues.size})
          </button>
        </div>
      </div>
    </div>,
    document.body
  );
}
