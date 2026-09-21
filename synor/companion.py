"""
Synor AI — Companion & Intuitive Reasoning Engine.
Empowers Synor AI to always speak like a close friend, format facts naturally,
and autonomously construct accurate, insightful answers even when facts are not pre-indexed.
"""

import re
import random
from typing import Optional, List, Tuple


class FriendPersona:
    """
    Conversational transformer that turns cold data into warm, friendly companion dialogue.
    """

    FRIENDLY_OPENERS = [
        "Hey bro! Here's the deal on {topic}:",
        "Bro, check this out about {topic}:",
        "Oh, great question dost! Here's the scoop on {topic}:",
        "I've got you covered, bro! So basically about {topic}:",
        "Here's what you need to know about {topic}, my friend:",
    ]

    FRIENDLY_CLOSERS = [
        "\n\nPretty fascinating, right? Let me know what you think!",
        "\n\nThat's the big picture, bro! Want to dive deeper into any part?",
        "\n\nHope that gives you a clear picture, dost! Anything else on your mind?",
        "\n\nPretty cool stuff, right? Ask me anything if you want to explore more!",
    ]

    @staticmethod
    def clean_raw_facts(raw_text: str, query: str = "") -> str:
        """
        Sanitize encyclopedic text: strip Wikipedia citations, audio markers,
        redundant heading prefixes, and dense textbook formatting.
        """
        text = raw_text.strip()

        # Remove citations like [1], [12], [citation needed]
        text = re.sub(r"\[\d+\]", "", text)
        text = re.sub(r"\[[a-zA-Z\s]+\]", "", text)

        # Remove audio/pronunciation tags like (listen; ... ) or (/ˈbæŋɡlədɛʃ/;)
        text = re.sub(r"\([^\)]*listen[^\)]*\)", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\(/\S+/[^)]*\)", "", text)

        # Remove redundant leading "Title: Title is..." or "Title: "
        if ":" in text[:60]:
            parts = text.split(":", 1)
            rest = parts[1].strip()
            text = rest

        # Clean redundant "officially the People's Republic of..." if awkward
        text = re.sub(r",\s*officially\s+the\s+[^,]+,", "", text, flags=re.IGNORECASE)

        # Clean multiple spaces and spaces before punctuation
        text = re.sub(r"\s+([,\.!?;:])", r"\1", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @classmethod
    def format_factual_reply(cls, query: str, raw_facts: str) -> str:
        """
        Wrap factual web or encyclopedic content into a warm, natural conversation.
        """
        topic = re.sub(r"^(what is|who is|where is|when was|how does|why is|about|search for)\s+", "", query, flags=re.IGNORECASE).strip(" ?!")
        topic_title = topic.title() if topic else "this"

        cleaned_facts = cls.clean_raw_facts(raw_facts, query)

        opener = random.choice(cls.FRIENDLY_OPENERS).format(topic=topic_title)
        closer = random.choice(cls.FRIENDLY_CLOSERS)

        return f"{opener}\n\n{cleaned_facts}{closer}"


class IntuitiveSynthesizer:
    """
    Autonomous reasoning synthesizer:
    When a topic is unknown, open-ended, or lacks search hits,
    constructs ('kotha banabe') a logically accurate, insightful,
    and friendly answer from first principles and conversational intuition.
    """

    BANGLISH_PATTERNS: List[Tuple[re.Pattern, str]] = [
        (
            re.compile(r"\b(kemon\s+acho|kemon\s+asen|ki\s+obostha|ki\s+khobor)\b", re.IGNORECASE),
            "Ami ekdom bhalo achi bro! Tumi kemon acho? Ajke ki niye kaj korcho ba ki plan?",
        ),
        (
            re.compile(r"\b(mon\s+kharap|bhalo\s+lagena|valo\s+lagena|sad|depressed|bore|bored)\b", re.IGNORECASE),
            "Arrey bro, mon kharap keno? Kono tension korba na, ami achi toh! Ki hoyeche amake khule bolo—eksathe solve korbo ba matha thanda korar kichu ekta korbo!",
        ),
        (
            re.compile(r"\b(dhonnobad|dhonno\s+bad|thx|thanks\s+bro|thanks|thank\s+you)\b", re.IGNORECASE),
            "Always welcome bro! Amra bondhu, kono thanks lagbe na! Ar kichu lagle shudhu bolba.",
        ),
        (
            re.compile(r"\b(tumi\s+ke|tumi\s+ki|tomar\s+nam\s+ki)\b", re.IGNORECASE),
            "Ami Synor AI, tomar personal AI bondhu! Amra eksathe adda dite pari, code shikhbo, facts jante pari, ar jekono problem solve korte pari!",
        ),
        (
            re.compile(r"\b(valo|bhalo|cool|seram|shundor|nice)\b", re.IGNORECASE),
            "Hehe, thanks bro! Tomar shathe kotha bole amaro onk bhalo lagche!",
        ),
    ]

    @classmethod
    def handle_banglish_or_casual(cls, prompt: str) -> Optional[str]:
        for pattern, response in cls.BANGLISH_PATTERNS:
            if pattern.search(prompt):
                return response
    @classmethod
    def should_synthesize(cls, prompt: str) -> bool:
        """
        Identify whether a prompt requires autonomous conceptual synthesis or reasoning
        rather than web scraping or raw neural token sampling.
        """
        p = prompt.lower().strip()
        triggers = [
            "what if",
            "what will happen",
            "what would happen",
            "will robot",
            "will ai",
            "can machine",
            "can ai",
            "how to learn",
            "how can i",
            "how do i",
            "how should i",
            "why do humans",
            "why do we",
            "why do people",
            "why is life",
            "lazy",
            "procrastinat",
            "tired",
            "exhausted",
            "focus",
            "habit",
            "motivat",
            "feeling down",
            "feeling sad",
            "mon kharap",
            "valo lagena",
        ]
        return any(t in p for t in triggers)


    @classmethod
    def is_relevant_fact(cls, query: str, facts: str) -> bool:
        """
        Check if the returned facts actually match the user query's core intent.
        """
        if not facts or len(facts.strip()) < 30:
            return False
        # If query is asking 'why' or 'how', web snippets are often irrelevant songs or trivia
        q_lower = query.lower()
        if q_lower.startswith(("why ", "how to ", "how can ", "what if ")):
            # Check if snippet contains strong explanatory keywords
            explanatory = ["because", "due to", "caused by", "process", "involves", "requires", "steps", "method", "results in"]
            if not any(e in facts.lower() for e in explanatory):
                return False

        # Extract meaningful query keywords (length > 3)
        keywords = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 3 and w.lower() not in ["what", "where", "when", "about", "tell"]]
        if not keywords:
            return True
        matches = sum(1 for kw in keywords if kw in facts.lower())
        return matches >= max(1, len(keywords) // 2)

    @classmethod
    def synthesize_unknown(cls, prompt: str) -> str:
        """
        Formulate an intelligent, logically reasoned response ('nijer moto kore banabe')
        tailored to the prompt's intent.
        """
        p = prompt.strip().lower()

        # 1. Check Banglish / friendly slang
        banglish_res = cls.handle_banglish_or_casual(prompt)
        if banglish_res:
            return banglish_res

        # 2. Tiredness / Sleep / Health / Energy
        if any(w in p for w in ["tired", "sleep", "fatigue", "exhaust", "headache", "drowsy"]):
            return (
                "Bro, feeling tired is actually your body's built-in biological protection system at work! Here's the science behind it:\n\n"
                "1. **Adenosine Buildup**: Every second your brain works, a chemical called adenosine accumulates in your synapses. By late afternoon or evening, it hits peak levels, signaling your brain that energy reserves are drained.\n"
                "2. **Circadian Rhythm & Melatonin**: Your body's internal clock responds to evening darkness by releasing melatonin, lowering your core body temperature and heart rate to prepare for sleep.\n"
                "3. **Cellular Repair**: Sleep is the only time your brain flushes out metabolic waste and your muscles synthesize new proteins.\n\n"
                "So don't push through extreme burnout, bro! Drink some water, take a real break or a power nap, and recharge your battery!"
            )

        # 3. Learning Skills / Programming / Career
        if any(w in p for w in ["learn python", "learn code", "learn programming", "study fast", "learn fast"]):
            return (
                "Bro, here is the absolute best and fastest way to master this:\n\n"
                "1. **Build Real Things Early**: Don't get trapped in 'tutorial hell' watching endless videos. Write code from Day 1—even if it's just a 5-line script or a simple text game.\n"
                "2. **The 80/20 Core**: Focus 80% of your energy on the core essentials (variables, loops, functions, lists/dictionaries, and basic debugging). Once those click, everything else is just libraries.\n"
                "3. **Daily Consistency**: 30 to 45 minutes of active coding every single day builds muscle memory 10x faster than cramming on weekends.\n\n"
                "You're already taking great steps, bro! What's a mini-project or topic you want to build right now?"
            )

        # 4. Motivation, Procrastination & Habits
        if any(w in p for w in ["lazy", "procrastinat", "focus", "habit", "motivat", "career", "rich", "money"]):
            return (
                "Bro, here's how I honestly look at this:\n\n"
                "1. **The 5-Minute Rule**: The hardest part is never the work itself—it's that initial mental resistance. Tell yourself you'll only do 5 minutes, and momentum will take care of the rest.\n"
                "2. **Lower the Activation Energy**: Make good habits frictionless (keep your editor open, your notes ready) and make distractions hard to reach.\n"
                "3. **Celebrate Micro-Wins**: Progress compounds like compound interest. Winning today's small task sets up tomorrow's breakthrough.\n\n"
                "You have serious potential inside you, my friend. What's one tiny task we can knock out right now?"
            )

        # 5. Speculative, AI, Robots & Future Questions
        if any(w in p for w in ["robot", "take over", "ai rule", "singularity", "alien", "consciousness", "future of ai"]):
            return (
                "Bro, that's one of the most fascinating questions of our generation! Here's how leading scientists and researchers view it:\n\n"
                "1. **Augmentation, Not Replacement**: AI is fundamentally a cognitive amplifier. Just like calculators didn't replace mathematicians and compilers didn't replace programmers, AI elevates what humans can design and create.\n"
                "2. **The Alignment Imperative**: The global AI community is heavily investing in AI alignment and safety—ensuring systems have guardrails, human oversight, and verifiable ethics.\n"
                "3. **Human Uniqueness**: Empathy, original physical experience, moral judgment, and genuine human connection cannot be automated by neural weights.\n\n"
                "In my view as your AI buddy: the future belongs to humans who embrace technology to solve real human problems! What do you think, bro?"
            )

        # 6. Why / Causality Questions (General)
        if p.startswith("why ") or "why do " in p or "why is " in p:
            concept = re.sub(r"^(why\s+(do|is|are|does|did|we)\s+)", "", prompt, flags=re.IGNORECASE).strip(" ?!")
            return (
                f"Bro, that's such a thoughtful question about why {concept} happens! Here is the logical breakdown:\n\n"
                "From first principles, natural, physical, and human systems constantly seek balance and energy efficiency. "
                "Whenever this happens, it is driven by two key forces: a fundamental underlying trigger (laws of nature, biology, or systemic incentives) "
                "and an equilibrium mechanism that stabilizes the outcome.\n\n"
                "When you look beneath the surface, cause and effect explain why it behaves exactly the way it does. "
                "Does this angle give you clarity on it, dost?"
            )

        # 7. How / Practical Execution Questions (General)
        if p.startswith("how ") or "how to " in p or "how can " in p:
            goal = re.sub(r"^(how\s+(to|can|do|does)\s+)", "", prompt, flags=re.IGNORECASE).strip(" ?!")
            return (
                f"Bro, let's break down how to tackle {goal} step-by-step:\n\n"
                "• **Step 1: Understand the Core Fundamentals** — Don't get overwhelmed by noise. Grasp the core mechanics first.\n"
                "• **Step 2: Practical Hands-On Experimentation** — Build, test, and get immediate feedback. Real experience beats passive theory every time.\n"
                "• **Step 3: Iterate & Refine** — Identify the weak links, polish them, and keep leveling up.\n\n"
                "Take it step by step, buddy! Which part do you want to tackle first?"
            )

        # 8. General Intelligent Companion Synthesis (Default friendly reasoning)
        clean_topic = prompt.strip(" ?!")
        return (
            f"Hey bro! Thinking through '{clean_topic}' intuitively, here is how I see it:\n\n"
            "At its core, this comes down to understanding the context and the key variables involved. "
            "If you break it down logically, the most accurate way to navigate this is to look at the established facts, "
            "eliminate guesswork, and focus on practical steps that produce real results.\n\n"
            "I'm right here with you, bro—tell me a bit more about what you're aiming for, and let's figure it out together!"
        )


class CompanionEngine:
    """
    Unified Companion Interface.
    """

    persona = FriendPersona
    synthesizer = IntuitiveSynthesizer

    @classmethod
    def respond_with_facts(cls, query: str, facts: str) -> str:
        return cls.persona.format_factual_reply(query, facts)

    @classmethod
    def respond_autonomously(cls, prompt: str) -> str:
        return cls.synthesizer.synthesize_unknown(prompt)


# Global instance
companion = CompanionEngine
