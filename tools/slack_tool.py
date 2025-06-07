"""
Slack Tool for fetching Slack data to monitor user activity and calculate burnout indicators.

Features:
- Fetch users in the workspace
- Fetch messages sent by users over a time period
- Analyze user activity metrics such as message counts and active hours
- Simple burnout calculation heuristics based on activity levels

Usage:
- Requires a Slack Bot Token with these scopes:
  * users:read
  * conversations:history
  * channels:read
  * groups:read
  * im:read
  * mpim:read
- You can set the token using the environment variable `SLACK_BOT_TOKEN`
  or pass it directly when creating SlackTool instance.

Example:
  tool = SlackTool(token="xoxb-...")
  users = tool.fetch_users()
  for user in users:
      messages = tool.fetch_user_messages(user['id'], days=7)
      analysis = tool.analyze_activity(messages)
      print(f"User: {user['name']}, Burnout score: {analysis['burnout_score']}")

"""

import os
import time
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


class SlackTool:
    def __init__(self, token=None):
        """
        Initialize SlackTool with Slack Bot Token.
        If token is None, it tries to get from environment variable SLACK_BOT_TOKEN.
        """
        if token is None:
            token = os.environ.get('SLACK_BOT_TOKEN')
        if not token:
            raise ValueError("Slack Bot Token must be provided or set in environment variable SLACK_BOT_TOKEN")
        self.client = WebClient(token=token)

    def fetch_users(self):
        """
        Fetch the list of active users in the workspace.
        Returns a list of dicts with user info: id, name, real_name, is_bot, deleted.
        """
        users = []
        cursor = None
        try:
            while True:
                response = self.client.users_list(cursor=cursor)
                members = response.get("members", [])
                users.extend(members)
                cursor = response["response_metadata"].get("next_cursor") if response.get("response_metadata") else None
                if not cursor:
                    break
            # Filter out bots and deleted users
            filtered_users = [
                {
                    "id": u["id"],
                    "name": u.get("name"),
                    "real_name": u.get("real_name"),
                    "is_bot": u.get("is_bot", False),
                    "deleted": u.get("deleted", False)
                }
                for u in users if not u.get("deleted", False) and not u.get("is_bot", False)
            ]
            return filtered_users
        except SlackApiError as e:
            print(f"Error fetching users: {e.response['error']}")
            return []

    def _fetch_conversations(self):
        """
        Fetch the list of conversations (channels, groups, ims, mpims) where user messages can exist.
        Returns list of conversation dicts.
        """
        conversations = []
        types = "public_channel,private_channel,im,mpim"
        cursor = None
        try:
            while True:
                response = self.client.conversations_list(types=types, cursor=cursor, limit=200)
                conversations.extend(response.get("channels", []))
                cursor = response["response_metadata"].get("next_cursor") if response.get("response_metadata") else None
                if not cursor:
                    break
            return conversations
        except SlackApiError as e:
            print(f"Error fetching conversations: {e.response['error']}")
            return []

    def fetch_user_messages(self, user_id, days=7):
        """
        Fetch messages sent by the given user in the last `days` days.
        Searches in all conversations returned by conversations_list.

        Returns a list of message dicts (with timestamp and text).
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        oldest_ts = cutoff.timestamp()
        user_messages = []

        conversations = self._fetch_conversations()
        for conv in conversations:
            channel_id = conv.get("id")
            # We'll paginate messages in this channel
            cursor = None
            while True:
                try:
                    response = self.client.conversations_history(
                        channel=channel_id,
                        cursor=cursor,
                        limit=200,
                        oldest=oldest_ts
                    )
                    messages = response.get("messages", [])
                    # Filter messages by user_id
                    user_msgs = [m for m in messages if m.get("user") == user_id]
                    user_messages.extend(user_msgs)

                    cursor = response["response_metadata"].get("next_cursor") if response.get(
                        "response_metadata") else None
                    if not cursor:
                        break
                    # Slack rate limit precautions
                    time.sleep(0.2)
                except SlackApiError as e:
                    # If channel is not accessible, ignore
                    if e.response['error'] in ('channel_not_found', 'not_in_channel', 'is_archived',
                                               'restricted_action'):
                        break
                    print(f"Error fetching messages from channel {channel_id}: {e.response['error']}")
                    break
        return user_messages

    def analyze_activity(self, messages):
        """
        Analyze user message data to calculate burnout indicators.
        Currently metrics:
        - total messages
        - average messages per day
        - active hours spread
        - burnout_score (simple heuristic)

        Returns a dictionary with analysis results.
        """
        if not messages:
            return {
                "total_messages": 0,
                "avg_messages_per_day": 0,
                "active_hours_distribution": {},
                "burnout_score": 0,
                "notice": "No messages found for user in timeframe"
            }

        # Group message timestamps by hour of day
        hour_counts = Counter()
        timestamps = []
        for m in messages:
            ts = float(m.get("ts"))
            timestamps.append(ts)
            dt = datetime.utcfromtimestamp(ts)
            hour_counts[dt.hour] += 1

        total_messages = len(messages)

        timespan_seconds = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 1
        timespan_days = timespan_seconds / (3600 * 24)

        avg_messages_per_day = total_messages / max(timespan_days, 1)

        # Active hours spread: count how many different hours user is active in (% active hours over 24 hours)
        active_hours = len([h for h, c in hour_counts.items() if c > 0])
        active_hours_percent = active_hours / 24

        # Burnout score heuristic:
        # - High message volume (> avg 50 messages per day) increases burnout
        # - Spread out activity (active many hours a day) decreases burnout
        # Score normalized 0 to 100
        burnout_score = 0
        if avg_messages_per_day > 50:
            burnout_score += min(50, (avg_messages_per_day - 50) * 2)  # penalize heavily for very high volume
        burnout_score += (1 - active_hours_percent) * 50  # penalize for concentrated activity hours
        burnout_score = min(100, round(burnout_score))

        return {
            "total_messages": total_messages,
            "avg_messages_per_day": round(avg_messages_per_day, 2),
            "active_hours_distribution": dict(hour_counts),
            "burnout_score": burnout_score,
            "notice": "Burnout score 0-100 based on message volume and activity hour spread"
        }


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    import argparse

    parser = argparse.ArgumentParser(description="Slack Tool for User Activity and Burnout Analysis")
    parser.add_argument("--token", type=str, help="Slack Bot Token (or set SLACK_BOT_TOKEN env variable)")
    parser.add_argument("--days", type=int, default=7, help="Number of days of activity to fetch (default: 7)")
    parser.add_argument("--list-users", action="store_true", help="List Slack workspace users")
    parser.add_argument("--user", type=str, help="User ID to fetch and analyze messages for")
    args = parser.parse_args()

    token = args.token or os.environ.get('SLACK_BOT_TOKEN')
    if not token:
        print("ERROR: Slack Bot Token must be provided via --token or SLACK_BOT_TOKEN environment variable")
        exit(1)

    slack_tool = SlackTool(token)

    if args.list_users:
        print("Fetching users...")
        users = slack_tool.fetch_users()
        for u in users:
            print(f"- {u['id']}: {u['real_name']} (@{u['name']})")
        exit(0)

    if args.user:
        print(f"Fetching messages for user ID: {args.user} (last {args.days} days)...")
        messages = slack_tool.fetch_user_messages(args.user, days=args.days)
        print(f"Fetched {len(messages)} messages")

        print("Analyzing activity...")
        analysis = slack_tool.analyze_activity(messages)
        print("Activity analysis:")
        for k, v in analysis.items():
            print(f"{k}: {v}")
        exit(0)

    print("No action specified. Use --list-users to list users or --user <USER_ID> to analyze user activity.")

