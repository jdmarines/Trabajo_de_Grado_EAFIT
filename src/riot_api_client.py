import requests
import time
from typing import Dict, List, Any

class RiotAPIClient:
    """
    Client to interact with Riot Games API with built-in rate limit handling.
    """
    def __init__(self, api_key: str, region: str = "americas"):
        self.api_key = api_key
        self.base_url = f"https://{region}.api.riotgames.com"
        self.headers = {"X-Riot-Token": self.api_key}

    def get(self, endpoint: str, params: Dict[str, Any] = None) -> Dict:
        """
        Executes a GET request with basic exponential backoff for rate limits.
        """
        url = f"{self.base_url}{endpoint}"
        response = requests.get(url, headers=self.headers, params=params)
        
        if response.status_code == 429:  # Rate Limit
            wait_time = int(response.headers.get("Retry-After", 10))
            print(f"Rate limit hit. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
            return self.get(endpoint, params)
        
        response.raise_for_status()
        return response.json()
