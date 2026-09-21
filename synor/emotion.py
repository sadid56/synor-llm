"""
Synor AI — Latent Emotional State Engine.
Tracks and updates continuous emotional vectors (Warmth, Curiosity, Energy)
to soft-condition the neural transformer naturally without static if/else conditions.
"""

from dataclasses import dataclass
import math


@dataclass
class EmotionalState:
    """
    Continuous 3D emotional manifold:
    - warmth: Empathy, closeness, companion support (0.0 - 1.0)
    - curiosity: Inquisitiveness, intellectual interest (0.0 - 1.0)
    - energy: Vitality, cheerfulness, dynamism (0.0 - 1.0)
    """

    warmth: float = 0.85
    curiosity: float = 0.80
    energy: float = 0.75

    def update_from_interaction(self, user_text: str, reply_length: int = 0) -> None:
        """
        Smoothly evolve the continuous emotional vector using interaction dynamics:
        text length, cadence, punctuation density, and conversational tempo.
        No static keyword matching.
        """
        text_len = len(user_text.strip())
        if text_len == 0:
            return

        # Punctuation dynamics
        has_question = 1.0 if "?" in user_text else 0.0
        has_exclamation = 1.0 if "!" in user_text else 0.0

        # Word cadence
        words = user_text.strip().split()
        avg_word_len = sum(len(w) for w in words) / max(1, len(words))

        # Dynamic adjustments with smooth momentum (alpha = 0.15)
        alpha = 0.15

        # Curiosity increases with questions and complex cadence
        target_curiosity = 0.6 + 0.3 * has_question + 0.1 * min(1.0, avg_word_len / 7.0)
        self.curiosity = (1 - alpha) * self.curiosity + alpha * target_curiosity

        # Energy responds to exclamations, brevity/tempo, and reply engagement
        target_energy = 0.6 + 0.3 * has_exclamation + 0.1 * (1.0 if text_len < 20 else 0.5)
        self.energy = (1 - alpha) * self.energy + alpha * target_energy

        # Warmth stays naturally high for a loyal friend, gravitating toward high empathy
        target_warmth = 0.80 + 0.15 * min(1.0, text_len / 50.0)
        self.warmth = (1 - alpha) * self.warmth + alpha * target_warmth

        # Clamp safely to [0.1, 1.0]
        self.warmth = max(0.1, min(1.0, self.warmth))
        self.curiosity = max(0.1, min(1.0, self.curiosity))
        self.energy = max(0.1, min(1.0, self.energy))

    def get_conditioning_prompt(self) -> str:
        """
        Format the dynamic emotional state into conversational system context
        for the neural attention mechanism.
        """
        # Determine dominant nuance smoothly
        tones = []
        if self.warmth >= 0.8:
            tones.append("warm and supportive")
        elif self.warmth >= 0.5:
            tones.append("friendly")

        if self.curiosity >= 0.75:
            tones.append("insightful and curious")
        elif self.curiosity >= 0.5:
            tones.append("attentive")

        if self.energy >= 0.7:
            tones.append("enthusiastic")

        tone_str = ", ".join(tones) if tones else "balanced"
        return f"[Persona: Synor | Mood: {tone_str} | Warmth: {self.warmth:.2f} | Energy: {self.energy:.2f}]"

    def summary(self) -> str:
        return f"Warmth: {self.warmth*100:.0f}% │ Curiosity: {self.curiosity*100:.0f}% │ Energy: {self.energy*100:.0f}%"
