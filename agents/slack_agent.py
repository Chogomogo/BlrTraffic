"""
SlackAgent for integrating Slack data fetching and analysis with LangChain and OpenAI.

Features:
- Uses SlackTool to fetch Slack user activity data and analyze burnout.
- Uses OpenAI (LangChain) to further analyze or summarize Slack data.
- Optionally stores and retrieves Slack data & analysis results in MongoDB.
- Designed to be one agent within a multi-agent system.

Dependencies:
- slack_sdk (for Slack API)
- langchain
- openai
- pymongo

Setup:
- Requires Slack Bot Token with needed scopes.
- Requires OpenAI API key.
- MongoDB URI if persistence is used.

Usage example in __main__ section.

"""

import os
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv

from tools.slack_tool import SlackTool
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pymongo import MongoClient


class SlackAgent:
    def __init__(
            self,
            slack_token: Optional[str] = None,
            openai_api_key: Optional[str] = None,
            mongo_uri: Optional[str] = None,
            mongo_db_name: str = "slack_agent_db",
            mongo_collection_name: str = "user_activity",
            llm_model: str = "gpt-4",
            llm_temperature: float = 0.7,
    ):
        """Initialize SlackAgent with Slack token, OpenAI key, MongoDB URI."""
        self.slack_token = slack_token or os.environ.get("SLACK_BOT_TOKEN")
        if not self.slack_token:
            raise ValueError("Slack Bot Token must be provided or set as SLACK_BOT_TOKEN.")
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY.")

        self.slack_tool = SlackTool(token=self.slack_token)

        # Setup LLM with LangChain OpenAI wrapper
        self.llm = ChatOpenAI(
            model_name=llm_model,
            temperature=llm_temperature,
            openai_api_key=self.openai_api_key,
        )

        # MongoDB client, if URI provided
        self.mongo_client = None
        self.db = None
        self.collection = None
        if mongo_uri:
            self.mongo_client = MongoClient(mongo_uri)
            self.db = self.mongo_client[mongo_db_name]
            self.collection = self.db[mongo_collection_name]

        # Prompt template for summary/analyses from Slack messages
        self.prompt_template = PromptTemplate(
            input_variables=["messages_summary"],
            template=(
                "You are a helpful assistant evaluating Slack user activity data to detect potential burnout.\n"
                "Given the following summary of user messages over the last week:\n\n{messages_summary}\n\n"
                "Please provide a concise analysis of the user's activity patterns and signs of burnout, "
                "and suggest any recommendations."
            )
        )
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def fetch_and_analyze_user(self, user_id: str, days: int = 7) -> Dict[str, Any]:
        """Fetch messages for user and analyze using SlackTool + LLM."""
        # Fetch messages from Slack
        messages = self.slack_tool.fetch_user_messages(user_id, days=days)
        # Analyze raw metrics using SlackTool builtin
        raw_analysis = self.slack_tool.analyze_activity(messages)

        # Prepare text summary for LLM input
        summary_text = f"Total messages: {raw_analysis['total_messages']}\n"
        summary_text += f"Average messages per day: {raw_analysis['avg_messages_per_day']}\n"
        summary_text += f"Burnout score: {raw_analysis['burnout_score']}\n"
        summary_text += f"Activity hours distribution: {raw_analysis['active_hours_distribution']}\n"

        # Optionally add a short snippet of messages as example (limited)
        example_msgs = messages[:5]
        if example_msgs:
            summary_text += "Example messages:\n"
            for msg in example_msgs:
                text = msg.get("text", "").replace('\n', ' ')
                ts = msg.get("ts")
                dt = "unknown time"
                try:
                    from datetime import datetime
                    dt = datetime.utcfromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M UTC")
                except Exception:
                    pass
                summary_text += f"- [{dt}] {text}\n"

        # Run LLM analysis
        llm_result = self.llm_chain.run(messages_summary=summary_text)

        # Optionally store results in MongoDB
        if self.collection:
            record = {
                "user_id": user_id,
                "days_analyzed": days,
                "raw_analysis": raw_analysis,
                "llm_analysis": llm_result,
                "timestamp": datetime.utcnow(),
            }
            self.collection.update_one(
                {"user_id": user_id},
                {"$set": record},
                upsert=True
            )

        return {
            "raw_analysis": raw_analysis,
            "llm_analysis": llm_result,
        }

    def list_users(self) -> List[Dict[str, Any]]:
        """Fetch and return list of Slack users."""
        return self.slack_tool.fetch_users()


if __name__ == "__main__":
    import argparse
    from datetime import datetime
    load_dotenv()
    parser = argparse.ArgumentParser(description="SlackAgent CLI for Slack user activity and burnout analysis.")
    parser.add_argument("--slack-token", type=str, help="Slack Bot Token")
    parser.add_argument("--openai-key", type=str, help="OpenAI API Key")
    parser.add_argument("--mongo-uri", type=str, default=None, help="MongoDB connection URI")
    parser.add_argument("--user-id", type=str, help="Slack user ID to analyze")
    parser.add_argument("--days", type=int, default=7, help="Days to look back for analysis")
    parser.add_argument("--list-users", action="store_true", help="List users in Slack workspace")
    args = parser.parse_args()

    slack_token = args.slack_token or os.environ.get("SLACK_BOT_TOKEN")
    openai_key = args.openai_key or os.environ.get("OPENAI_API_KEY")

    agent = SlackAgent(
        slack_token=slack_token,
        openai_api_key=openai_key,
        mongo_uri=args.mongo_uri,
    )

    if args.list_users:
        users = agent.list_users()
        for u in users:
            print(f"- {u['id']}: {u['real_name']} (@{u['name']})")
    elif args.user_id:
        result = agent.fetch_and_analyze_user(args.user_id, days=args.days)
        print("Raw Analysis:")
        for k, v in result["raw_analysis"].items():
            print(f"{k}: {v}")
        print("\nLLM Analysis (summary):")
        print(result["llm_analysis"])
    else:
        print("Specify either --list-users or --user-id to analyze a user.")

