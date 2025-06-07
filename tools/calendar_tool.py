"""
Calendar Tool for fetching Google Calendar meetings and analyzing meeting durations.

Features:
- Fetch calendar events for a user over a specified timeframe.
- Calculate total time spent in meetings.
- Provide meeting counts and durations.
- CLI support for listing meetings and summary.

Requirements:
- google-api-python-client
- google-auth
- google-auth-oauthlib
- google-auth-httplib2

Setup:
- Obtain OAuth 2.0 credentials from Google Cloud Console:
  - Enable Google Calendar API.
  - Create OAuth client credentials.
  - Download credentials.json.
- This tool uses local OAuth flow and stores token.pickle for reuse.

Usage:
- Run `python calendar_tool.py --help` for CLI options.
"""

import os
import datetime
import pickle
import argparse
from typing import List, Dict

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If modifying these scopes, delete token.pickle.
SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']

class CalendarTool:
    def __init__(self, credentials_path: str = 'credentials.json', token_path: str = 'token.pickle'):
        self.creds = None
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.service = None
        self.authenticate()

    def authenticate(self):
        """Authenticate with Google API using OAuth 2.0."""
        if os.path.exists(self.token_path):
            with open(self.token_path, 'rb') as token:
                self.creds = pickle.load(token)

        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, SCOPES)
                self.creds = flow.run_local_server(port=0)
            with open(self.token_path, 'wb') as token:
                pickle.dump(self.creds, token)

        self.service = build('calendar', 'v3', credentials=self.creds)

    def list_calendar_events(
        self,
        calendar_id: str = 'primary',
        time_min: datetime.datetime = None,
        time_max: datetime.datetime = None,
        max_results: int = 2500
    ) -> List[Dict]:
        """
        List events from calendar within time range.

        Args:
          calendar_id: calendar to query ('primary' means main user calendar)
          time_min: RFC3339 timestamp, inclusive start time
          time_max: RFC3339 timestamp, exclusive end time
          max_results: maximum events to return (default 2500 max per Google API)

        Returns:
          List of event dicts.
        """
        if not self.service:
            raise Exception("Google Calendar API service not initialized.")

        time_min_rfc = time_min.isoformat() + 'Z' if time_min else None
        time_max_rfc = time_max.isoformat() + 'Z' if time_max else None

        events = []
        page_token = None

        while True:
            try:
                events_result = self.service.events().list(
                    calendarId=calendar_id,
                    timeMin=time_min_rfc,
                    timeMax=time_max_rfc,
                    maxResults=max_results,
                    singleEvents=True,
                    orderBy='startTime',
                    pageToken=page_token
                ).execute()
                fetched_events = events_result.get('items', [])
                events.extend(fetched_events)
                page_token = events_result.get('nextPageToken')
                if not page_token:
                    break
            except HttpError as error:
                print(f"An error occurred: {error}")
                break
        return events

    def calculate_meeting_stats(self, events: List[Dict]) -> Dict:
        """
        Calculate total meeting duration, count, average duration.

        Args:
          events: list of event dicts from calendar.

        Returns:
          Dict with counts and duration (seconds).
        """
        total_duration_sec = 0
        meeting_count = 0
        durations = []

        for event in events:
            start = event.get('start')
            end = event.get('end')

            if not start or not end:
                continue

            start_dt = self._parse_event_datetime(start)
            end_dt = self._parse_event_datetime(end)

            if start_dt and end_dt:
                duration = (end_dt - start_dt).total_seconds()
                if duration > 0:
                    total_duration_sec += duration
                    meeting_count += 1
                    durations.append(duration)

        avg_duration_sec = total_duration_sec / meeting_count if meeting_count > 0 else 0
        return {
            'meeting_count': meeting_count,
            'total_duration_seconds': total_duration_sec,
            'average_duration_seconds': avg_duration_sec,
            'durations': durations,
        }

    def _parse_event_datetime(self, dt_dict: Dict) -> datetime.datetime:
        """
        Parse event date/time dict; can be dateTime or date (all day).

        Returns datetime in UTC.
        """
        try:
            if 'dateTime' in dt_dict:
                return datetime.datetime.fromisoformat(dt_dict['dateTime'].replace('Z', '+00:00')).astimezone(datetime.timezone.utc).replace(tzinfo=None)
            if 'date' in dt_dict:
                # all-day event, parse date only
                return datetime.datetime.fromisoformat(dt_dict['date']).replace(tzinfo=None)
        except Exception as e:
            pass
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Google Calendar Tool - fetch and analyze meetings")
    parser.add_argument('--calendar-id', type=str, default='primary', help='Calendar ID to fetch')
    parser.add_argument('--days', type=int, default=7, help='Number of days in past to fetch events')
    parser.add_argument('--credentials', type=str, default='credentials.json', help='Path to Google API credentials.json')
    parser.add_argument('--token', type=str, default='token.pickle', help='Path to OAuth token pickle file')
    parser.add_argument('--list-events', action='store_true', help='List detailed events')
    parser.add_argument('--summary', action='store_true', help='Show meeting summary stats')
    args = parser.parse_args()

    tool = CalendarTool(credentials_path=args.credentials, token_path=args.token)
    now = datetime.datetime.utcnow()
    time_min = now - datetime.timedelta(days=args.days)

    events = tool.list_calendar_events(
        calendar_id=args.calendar_id,
        time_min=time_min,
        time_max=now
    )

    if args.list_events:
        print(f"Listing {len(events)} events:")
        for e in events:
            start = e.get('start')
            end = e.get('end')
            summary = e.get('summary', 'No title')
            start_str = start.get('dateTime', start.get('date')) if start else "N/A"
            end_str = end.get('dateTime', end.get('date')) if end else "N/A"
            print(f"- {summary} from {start_str} to {end_str}")
    if args.summary:
        stats = tool.calculate_meeting_stats(events)
        print("Meeting summary statistics:")
        print(f"Total meetings: {stats['meeting_count']}")
        print(f"Total meeting time (hours): {stats['total_duration_seconds'] / 3600:.2f}")
        print(f"Average meeting duration (minutes): {stats['average_duration_seconds'] / 60:.2f}")


