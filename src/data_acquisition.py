import requests
import json
import os
from typing import Dict, Optional

class DataFetcher:
    """
    Handles the retrieval of static game data from Riot Games' Data Dragon.
    This ensures the model uses the correct base stats for a specific patch.
    """

    BASE_URL = "https://ddragon.leagueoflegends.com/cdn"

    def __init__(self, patch_version: str, language: str = "en_US"):
        """
        Args:
            patch_version (str): The specific game patch (e.g., '14.10.1').
            language (str): Localization string.
        """
        self.patch_version = patch_version
        self.language = language
        self.cache_dir = "data/raw"
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def fetch_champion_data(self) -> Optional[Dict]:
        """
        Fetches detailed champion data including base stats and spell metadata.
        
        Returns:
            Dict: A dictionary containing champion data or None if the request fails.
        """
        url = f"{self.BASE_URL}/{self.patch_version}/data/{self.language}/champion.json"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Save raw data for reproducibility
            file_path = os.path.join(self.cache_dir, f"champions_{self.patch_version}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
                
            return data['data']
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from Data Dragon: {e}")
            return None

    def get_champion_details(self, champion_id: str) -> Optional[Dict]:
        """
        Fetches specific details for a single champion (needed for spell data).
        """
        url = f"{self.BASE_URL}/{self.patch_version}/data/{self.language}/champion/{champion_id}.json"
        try:
            response = requests.get(url)
            return response.json()['data'][champion_id]
        except Exception as e:
            print(f"Error fetching details for {champion_id}: {e}")
            return None
