import os
import json
from src.riot_api_client import RiotAPIClient

class MatchExtractor:
    """
    Extracts match data for a specific patch and queue type.
    """
    def __init__(self, api_client: RiotAPIClient):
        self.client = api_client

    def get_match_ids(self, puuid: str, count: int = 100) -> List[str]:
        """
        Retrieves a list of match IDs for a given player.
        """
        endpoint = f"/lol/match/v5/matches/by-puuid/{puuid}/ids"
        params = {"queue": 420, "count": count} # 420 is Ranked Solo
        return self.client.get(endpoint, params=params)

    def get_match_details(self, match_id: str) -> Dict:
        """
        Retrieves detailed stats for a specific match.
        """
        endpoint = f"/lol/match/v5/matches/{match_id}"
        return self.client.get(endpoint)

    def save_match(self, match_data: Dict, patch_prefix: str):
        """
        Saves match data if it matches the target patch version.
        """
        game_version = match_data['info']['gameVersion']
        if game_version.startswith(patch_prefix):
            match_id = match_data['metadata']['matchId']
            path = f"data/raw/matches/{patch_prefix}"
            os.makedirs(path, exist_ok=True)
            
            with open(f"{path}/{match_id}.json", "w") as f:
                json.dump(match_data, f)
