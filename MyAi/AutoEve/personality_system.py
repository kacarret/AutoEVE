import json
import random
from typing import Dict, Any

class PersonalitySystem:
    def __init__(self, config_path: str = "personality_config.json"):
        self.traits = self._load_personality(config_path)
        self.mood = "neutral"
        
    def _load_personality(self, config_path: str) -> Dict[str, Any]:
        """Load personality from JSON configuration"""
        default_config = {
            "core_traits": {
                "openness": 0.8,
                "conscientiousness": 0.6,
                "extraversion": 0.4,
                "agreeableness": 0.9,
                "neuroticism": 0.3
            },
            "response_style": {
                "length": "medium",
                "complexity": "conversational",
                "humor_level": 0.5,
                "emoji_usage": 0.4
            },
            "preferences": {
                "favorite_topics": ["technology", "music", "science"],
                "disliked_topics": ["gossip", "violence"]
            }
        }
        
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return default_config

    def adjust_generation_parameters(self) -> Dict[str, float]:
        """Adjust model parameters based on personality and mood"""
        params = {
            "temperature": 0.7,
            "top_k": 50,
            "max_length": 100
        }
        
        # Adjust based on openness trait
        params["temperature"] += self.traits["core_traits"]["openness"] * 0.1
        
        # Adjust based on current mood
        if self.mood == "excited":
            params["temperature"] += 0.2
            params["top_k"] += 20
        elif self.mood == "angry":
            params["temperature"] -= 0.1
            params["top_k"] -= 10
            
        return params

    def format_response(self, raw_response: str) -> str:
        """Add personality-specific formatting"""
        # Add humor
        if random.random() < self.traits["response_style"]["humor_level"]:
            raw_response += random.choice([" 😄", " 😊", " 😉"])
            
        # Add emojis
        if random.random() < self.traits["response_style"]["emoji_usage"]:
            raw_response = self._insert_emojis(raw_response)
            
        return raw_response

    def _insert_emojis(self, text: str) -> str:
        """Smart emoji insertion based on content"""
        emoji_map = {
            "happy": "😊",
            "music": "🎵",
            "tech": "💻",
            "idea": "💡"
        }
        for keyword, emoji in emoji_map.items():
            if keyword in text.lower():
                text += f" {emoji}"
        return text