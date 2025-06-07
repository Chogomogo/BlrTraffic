"""
BaseAgents module integrating SlackAgent and GoogleCalendarAgent with unified summary,
and sends Pushover notifications for burnout alerts.

Features:
- Runs both Slack and Calendar agents.
- Produces a combined summary with raw metrics and LLM-generated analysis.
- Uses LangChain with OpenAI for synthesizing insights from both Slack and Calendar data.
- Sends notification via Pushover if burnout risk detected.
- CLI interface for quick testing of combined analysis and notifications.

Dependencies:
- slack_agent.py
- google_calendar_agent.py
- langchain
- openai
- requests
"""

import os
from datetime import datetime
from typing import Optional, Dict

import requests
from dotenv import load_dotenv

from agents.slack_agent import SlackAgent
from agents.google_calendar_agent import GoogleCalendarAgent

from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


class PushoverNotifier:
    """
    Sends notifications to a device using the Pushover API.
    Docs: https://pushover.net/api
    """

    API_URL = "https://api.pushover.net/1/messages.json"

    def __init__(self, user_key: str, app_token: str):
        """
        Initialize notifier with user key and application token.
        """
        self.user_key = user_key
        self.app_token = app_token

    def send_notification(self, title: str, message: str) -> bool:
        """
        Send a notification message.
        Returns True if success, False otherwise.
        """
        data = {
            "token": self.app_token,
            "user": self.user_key,
            "title": title,
            "message": message,
            "priority": 1,
            "sound": "magic",  # subtle sound
        }
        try:
            resp = requests.post(self.API_URL, data=data, timeout=10)
            resp.raise_for_status()
            return True
        except Exception as e:
            print(f"[PushoverNotifier] Error sending notification: {e}")
            return False


class BaseAgents:
    def __init__(
        self,
        slack_token: Optional[str] = None,
        google_credentials_path: str = "credentials.json",
        google_token_path: str = "token.pickle",
        openai_api_key: Optional[str] = None,
        mongo_uri: Optional[str] = None,
        pushover_user_key: Optional[str] = None,
        pushover_app_token: Optional[str] = None,
        slack_days: int = 7,
        calendar_days: int = 7,
        llm_model: str = "gpt-4",
        llm_temperature: float = 0.7,
    ):
        self.slack_token = slack_token or os.environ.get("SLACK_BOT_TOKEN")
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")

        if not self.slack_token:
            raise ValueError("Slack token must be provided or set in SLACK_BOT_TOKEN env variable")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY env variable")

        self.slack_agent = SlackAgent(
            slack_token=self.slack_token,
            openai_api_key=self.openai_api_key,
            mongo_uri=mongo_uri,
            llm_model=llm_model,
            llm_temperature=llm_temperature,
        )
        self.calendar_agent = GoogleCalendarAgent(
            credentials_path=google_credentials_path,
            token_path=google_token_path,
            openai_api_key=self.openai_api_key,
            mongo_uri=mongo_uri,
            llm_model=llm_model,
            llm_temperature=llm_temperature,
        )

        self.slack_days = slack_days
        self.calendar_days = calendar_days

        self.pushover_notifier = None
        if pushover_user_key and pushover_app_token:
            self.pushover_notifier = PushoverNotifier(pushover_user_key, pushover_app_token)

        # Combined LLM prompt template for unified summary / insight generation
        self.combined_prompt_template = PromptTemplate(
            input_variables=["slack_summary", "calendar_summary"],
            template=(
                "You are a senior productivity analyst tasked with synthesizing workspace data.\n\n"
                "Slack user activity data summary:\n{slack_summary}\n\n"
                "Google Calendar meetings data summary:\n{calendar_summary}\n\n"
                "Based on these datasets, please provide a comprehensive analysis looking for signs of "
                "user burnout, productivity bottlenecks, and suggest actionable recommendations to improve balance and efficiency."
            )
        )
        self.llm = ChatOpenAI(
            model_name=llm_model,
            temperature=llm_temperature,
            openai_api_key=self.openai_api_key,
        )
        self.combined_llm_chain = LLMChain(llm=self.llm, prompt=self.combined_prompt_template)

    def run_all_and_summarize(self, slack_user_id: str) -> Dict[str, any]:
        """
        Run Slack and Calendar agents to fetch, analyze and synthesize data for a given Slack user.

        Returns dict:
          {
            "slack_analysis": dict,
            "calendar_analysis": dict,
            "combined_summary": str
          }
        """
        # Fetch Slack data and analysis
        slack_result = self.slack_agent.fetch_and_analyze_user(slack_user_id, days=self.slack_days)
        raw_slack = slack_result.get("raw_analysis", {})
        llm_slack = slack_result.get("llm_analysis", "")

        # Prepare slack summary text for combined LLM
        slack_summary_text = (
            f"Raw metrics:\n"
            f" - Total messages: {raw_slack.get('total_messages', 0)}\n"
            f" - Avg messages/day: {raw_slack.get('avg_messages_per_day', 0):.2f}\n"
            f" - Burnout score: {raw_slack.get('burnout_score', 0)}\n"
            f"LLM analysis:\n{llm_slack}"
        )

        # Fetch Google Calendar data and analysis
        calendar_result = self.calendar_agent.fetch_and_analyze_calendar(days=self.calendar_days)
        meeting_stats = calendar_result.get("meeting_stats", {})
        llm_calendar = calendar_result.get("llm_summary", "")

        # Prepare calendar summary text for combined LLM
        calendar_summary_text = (
            f"Meeting stats:\n"
            f" - Total meetings: {meeting_stats.get('meeting_count', 0)}\n"
            f" - Total meeting hours: {meeting_stats.get('total_duration_seconds', 0)/3600:.2f}\n"
            f" - Average meeting duration (minutes): {meeting_stats.get('average_duration_seconds', 0)/60:.2f}\n"
            f"LLM analysis:\n{llm_calendar}"
        )

        # Generate combined summary with LLM chain
        combined_summary = self.combined_llm_chain.run(
            slack_summary=slack_summary_text,
            calendar_summary=calendar_summary_text,
        )

        # Check for burnout or alert conditions (simple heuristic)
        burnout_score = raw_slack.get('burnout_score', 0)
        burnout_threshold = 70  # customizable threshold
        alert_needed = False
        alert_reasons = []

        if burnout_score >= burnout_threshold:
            alert_needed = True
            alert_reasons.append(f"High Slack burnout score: {burnout_score}")

        # Additional possible calendar based checks (e.g., excessive meeting hours)
        total_meeting_hours = meeting_stats.get('total_duration_seconds', 0) / 3600
        meeting_hours_threshold = 30  # example threshold for weekly meeting overload
        if total_meeting_hours >= meeting_hours_threshold:
            alert_needed = True
            alert_reasons.append(f"High weekly meeting hours: {total_meeting_hours:.1f}h")

        # Send pushover notification if needed
        if alert_needed and self.pushover_notifier:
            notification_title = "Burnout Alert: User Activity Monitor"
            notification_message = (
                "Potential burnout detected for user.\n"
                + "\n".join(alert_reasons)
                + "\n\nSummary:\n" + combined_summary[:600]  # truncate for message limits
            )
            success = self.pushover_notifier.send_notification(notification_title, notification_message)
            if success:
                print("[BaseAgents] Burnout alert notification sent via Pushover.")
            else:
                print("[BaseAgents] Failed to send Pushover notification.")

        return {
            "slack_analysis": slack_result,
            "calendar_analysis": calendar_result,
            "combined_summary": combined_summary,
            "alert_sent": alert_needed and self.pushover_notifier is not None,
            "alert_reasons": alert_reasons,
        }


if __name__ == "__main__":
    import argparse
    load_dotenv()
    parser = argparse.ArgumentParser(description="BaseAgents CLI to run Slack and Google Calendar agents together with pushover notifications.")
    parser.add_argument("--slack-token", type=str, help="Slack Bot Token")
    parser.add_argument("--openai-key", type=str, help="OpenAI API Key")
    parser.add_argument("--mongo-uri", type=str, default=None, help="MongoDB connection URI (optional)")
    parser.add_argument("--pushover-user-key", type=str, default=None, help="Pushover User Key (optional)")
    parser.add_argument("--pushover-app-token", type=str, default=None, help="Pushover Application Token (optional)")
    parser.add_argument("--slack-user-id", type=str, required=True, help="Slack User ID to analyze")
    parser.add_argument("--slack-days", type=int, default=7, help="Days to analyze Slack activity")
    parser.add_argument("--calendar-credentials", type=str, default="credentials.json", help="Path to Google Calendar credentials.json")
    parser.add_argument("--calendar-token", type=str, default="token.pickle", help="Path to Google Calendar OAuth token.pickle")
    parser.add_argument("--calendar-days", type=int, default=7, help="Days to analyze calendar activity")

    args = parser.parse_args()

    base_agents = BaseAgents(
        slack_token=args.slack_token,
        openai_api_key=args.openai_key,
        mongo_uri=args.mongo_uri,
        pushover_user_key=args.pushover_user_key,
        pushover_app_token=args.pushover_app_token,
        slack_days=args.slack_days,
        calendar_days=args.calendar_days,
        google_credentials_path=args.calendar_credentials,
        google_token_path=args.calendar_token,
    )

    result = base_agents.run_all_and_summarize(slack_user_id=args.slack_user_id)

    print("\n===== SLACK RAW ANALYSIS =====")
    for k, v in result["slack_analysis"]["raw_analysis"].items():
        print(f"{k}: {v}")

    print("\n===== SLACK LLM ANALYSIS =====")
    print(result["slack_analysis"]["llm_analysis"])

    print("\n===== GOOGLE CALENDAR RAW STATS =====")
    for k, v in result["calendar_analysis"]["meeting_stats"].items():
        if isinstance(v, float):
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v}")

    print("\n===== GOOGLE CALENDAR LLM ANALYSIS =====")
    print(result["calendar_analysis"]["llm_summary"])

    print("\n===== COMBINED AGENT SUMMARY =====")
    print(result["combined_summary"])

    if result["alert_sent"]:
        print("\n[Pushover notification sent for burnout alert!]")
    elif result["alert_reasons"]:
        print("\n[Pushover available but notification not sent (check config)]")
    else:
        print("\n[No burnout alert triggered.]")

