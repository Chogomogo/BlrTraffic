"""
Google Calendar Agent integrating CalendarTool with LangChain OpenAI and MongoDB.

Features:
- Fetches calendar events from Google Calendar via CalendarTool.
- Calculates meeting stats like total meeting time, meeting count, average duration.
- Uses OpenAI via LangChain to analyze and summarize calendar activity (e.g., potential burnout).
- Stores analysis results in MongoDB optionally.
- Designed to be used as one agent in a multi-agent system.

Dependencies:
- calendar_tool.py (your CalendarTool implementation)
- langchain
- openai
- pymongo

Set environment variables or pass parameters for Google credentials and OpenAI API keys.

Example usage:
  python google_calendar_agent.py --list-events
  python google_calendar_agent.py --days 14

"""

import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from dotenv import load_dotenv

from tools.calendar_tool import CalendarTool # your previously created CalendarTool class

from langchain_community.chat_models import ChatOpenAI

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from pymongo import MongoClient


class GoogleCalendarAgent:
    def __init__(
        self,
        credentials_path: str = "credentials.json",
        token_path: str = "token.pickle",
        openai_api_key: Optional[str] = None,
        mongo_uri: Optional[str] = None,
        mongo_db_name: str = "calendar_agent_db",
        mongo_collection_name: str = "calendar_activity",
        llm_model: str = "gpt-4",
        llm_temperature: float = 0.7,
        calendar_id: str = "primary",
    ):
        # Initialize Google Calendar Tool
        self.calendar_tool = CalendarTool(credentials_path=credentials_path, token_path=token_path)
        self.calendar_id = calendar_id

        # OpenAI API key (required)
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environ variable")

        # Initialize LangChain OpenAI LLM client
        self.llm = ChatOpenAI(
            model_name=llm_model,
            temperature=llm_temperature,
            openai_api_key=self.openai_api_key,
        )

        # Setup MongoDB if URI provided
        self.mongo_client = None
        self.db = None
        self.collection = None
        if mongo_uri:
            self.mongo_client = MongoClient(mongo_uri)
            self.db = self.mongo_client[mongo_db_name]
            self.collection = self.db[mongo_collection_name]

        # Prepare prompt template for calendar activity summary via LLM
        self.prompt_template = PromptTemplate(
            input_variables=["meeting_summary"],
            template=(
                "You are an expert workplace productivity analyst. "
                "Analyze the following Google Calendar meeting summary data:\n\n"
                "{meeting_summary}\n\n"
                "Please provide a concise analysis of the user's calendar activity "
                "including total meeting time, meeting count, average meeting length, "
                "potential risks like burnout or meeting overload, and suggest recommendations."
            )
        )
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def fetch_and_analyze_calendar(
        self,
        days: int = 7,
        calendar_id: Optional[str] = None,
    ) -> Dict:
        """
        Fetch calendar events, analyze meeting stats and run OpenAI language model for summary.

        Returns:
          {
            "meeting_stats": {...},
            "llm_summary": "<text>",
          }
        """
        calendar_id = calendar_id or self.calendar_id
        now = datetime.utcnow()
        time_min = now - timedelta(days=days)

        # Fetch events
        events = self.calendar_tool.list_calendar_events(
            calendar_id=calendar_id,
            time_min=time_min,
            time_max=now,
        )

        # Calculate meeting statistics
        stats = self.calendar_tool.calculate_meeting_stats(events)

        # Prepare summary string for LLM input
        summary_lines = [
            f"Total Meetings: {stats.get('meeting_count', 0)}",
            f"Total meeting time (hours): {stats.get('total_duration_seconds', 0) / 3600:.2f}",
            f"Average meeting duration (minutes): {stats.get('average_duration_seconds', 0) / 60:.2f}",
        ]
        summary_text = "\n".join(summary_lines)

        # Call LLM for analysis
        llm_response = self.llm_chain.run(meeting_summary=summary_text)

        # Store to MongoDB if configured
        if self.collection:
            record = {
                "calendar_id": calendar_id,
                "days_analyzed": days,
                "meeting_stats": stats,
                "llm_summary": llm_response,
                "timestamp": datetime.utcnow(),
            }
            self.collection.update_one(
                {"calendar_id": calendar_id},
                {"$set": record},
                upsert=True,
            )

        return {
            "meeting_stats": stats,
            "llm_summary": llm_response,
        }


if __name__ == "__main__":
    import argparse
    load_dotenv()
    parser = argparse.ArgumentParser(description="Google Calendar Agent CLI")
    parser.add_argument(
        "--credentials",
        type=str,
        default="credentials.json",
        help="Path to Google API credentials.json",
    )
    parser.add_argument(
        "--token",
        type=str,
        default="token.pickle",
        help="Path to OAuth token.pickle",
    )
    parser.add_argument(
        "--openai-key",
        type=str,
        default=None,
        help="OpenAI API key",
    )
    parser.add_argument(
        "--mongo-uri",
        type=str,
        default=None,
        help="MongoDB connection URI (optional)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of past days to analyze (default 7)",
    )
    parser.add_argument(
        "--calendar-id",
        type=str,
        default="primary",
        help="Google Calendar ID to analyze (default 'primary')",
    )
    args = parser.parse_args()

    agent = GoogleCalendarAgent(
        credentials_path=args.credentials,
        token_path=args.token,
        openai_api_key=args.openai_key,
        mongo_uri=args.mongo_uri,
        calendar_id=args.calendar_id,
    )

    result = agent.fetch_and_analyze_calendar(days=args.days)
    stats = result["meeting_stats"]
    summary = result["llm_summary"]

    print("=== Meeting Statistics ===")
    print(f"Total Meetings: {stats.get('meeting_count', 0)}")
    print(f"Total Meeting Time (hours): {stats.get('total_duration_seconds', 0) / 3600:.2f}")
    print(f"Average Meeting Duration (minutes): {stats.get('average_duration_seconds', 0) / 60:.2f}")
    print()
    print("=== LLM Analysis Summary ===")
    print(summary)
