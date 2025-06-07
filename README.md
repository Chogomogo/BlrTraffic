# Slack & Google Calendar Productivity & Burnout Monitoring Agents

A modern, elegant backend agent system designed to monitor and analyze user activity across Slack and Google Calendar, aimed at detecting burnout and productivity bottlenecks. The system integrates Slack and Google Calendar data via dedicated agents, combines insights with advanced OpenAI language models (using LangChain), and can send real-time notifications via Pushover. Built with developer experience in mind, it emphasizes clarity, modularity, and cutting-edge technologies.

---

## Key Features

- **SlackAgent**  
  Fetches Slack user activity data, analyzes message volume and active hours, and computes burnout indicators.

- **GoogleCalendarAgent**  
  Fetches Google Calendar meetings, calculates total meeting time and frequency, and assesses potential overload or burnout risk.

- **BaseAgents**  
  Orchestrates Slack and Calendar agents, synthesizes combined insights using OpenAI GPT-4, and sends notifications to mobile devices via Pushover when burnout risk is detected.

- **Extensible & Modular**  
  Easily extend with additional agents or integrations. Supports MongoDB for data persistence and LangChain for sophisticated LLM orchestration.

- **Command-Line Interface (CLI)**  
  Intuitive CLI commands to fetch data, analyze, and receive alerts, facilitating automation and integration.

---

## Visual & Design Philosophy

While this project is backend-focused, it adheres to a clean, minimal, and elegant developer experience:

- Clear, semantic, and well-documented code
- Modular architecture for easy maintenance and extension
- Thoughtful naming and structured workflows
- Minimal external dependencies
- Easy CLI usability with consistent option naming and defaults

---

## Installation

Ensure you have Python 3.8+ installed.

Install required dependencies:

```bash
pip install slack_sdk langchain openai pymongo google-api-python-client google-auth google-auth-oauthlib google-auth-httplib2 requests

```

## Usage 
Slack Agent : Analyze Slack user activity by Slack User ID:
```bash
python slack_agent.py --user-id U12345678 --days 7
```

Google Calendar Agent :Analyze calendar meetings:

```bash
python google_calendar_agent.py --days 7
```


Combined Base Agents :Run both agents for a Slack user, generate a combined summary, and optionally send Pushover alerts:

```bash
python base_agents.py --slack-user-id U12345678 --pushover-user-key YOUR_USER_KEY --pushover-app-token YOUR_APP_TOKEN
```


