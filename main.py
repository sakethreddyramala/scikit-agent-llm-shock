"""
scikit-agent: an LLM as a random shock inside a DBlock.
(Free-tier: Gemini Flash-Lite)

In an economic model, a *shock* is something random that happens TO the agent
(like the weather, or news). The agent can't control it, but it can react to it.

Our shock here answers one question: "is today's headline good for the market?"
An LLM (Google Gemini) reads the headline and answers:
    True  = Yes, good news
    False = No, bad news
The AGENT only ever sees that yes/no. It never sees the headline or Gemini's
explanation, and it buys or sells based on the yes/no alone.

Gemini also writes a fresh one-line explanation every time. That line is only
printed for us humans to read; it is NOT passed to the agent.

Setup:
    pip install scikit-agent openai
    export GEMINI_API_KEY="your_key_here"    # free key from https://aistudio.google.com/apikey
    (Windows PowerShell:  $env:GEMINI_API_KEY="your_key_here")
"""

import os
import sys
import time

import skagent as ska
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam


client = OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

MODELS = [
    "gemini-flash-lite-latest",   # best for free usage: lightweight + high limits
    "gemini-flash-latest",        # backup: a bit smarter, tighter free limits
]


# ---------------------------------------------------------------------------
# The LLM function: headline -> (yes/no, one-line explanation)
# ---------------------------------------------------------------------------
def analyze_headline(headline: str):
    """Ask Gemini if the headline is good for stocks.

    Returns a pair: (is_good: bool, reason: str)
    The reason is freshly generated each call, so its wording changes every time.
    """
    # We ask for a strict two-line format so the reply is easy to parse:
    #   Reason: <one fresh sentence>
    #   Verdict: Yes   (or No)
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": (
                "You are a market analyst. Is this headline good for stocks? "
                "Reply in exactly two lines:\n"
                "Reason: <one short sentence in your own words>\n"
                "Verdict: <Yes or No>"
            ),
        },
        {"role": "user", "content": headline},
    ]

    text = ""
    for model in MODELS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.9,  # higher = the one-liner is worded differently each time
                max_tokens=150,   # room for one sentence + the verdict
            )
            text = (response.choices[0].message.content or "").strip()
            break
        except Exception as error:
            # Most common failure on a free tier: "429 too many requests".
            print(f"  [note] {model} failed ({type(error).__name__}); trying next model")
            time.sleep(2)

    # Split the reply into the reason and the verdict.
    reason = "(no reason given)"
    verdict = ""
    for line in text.splitlines():
        lowered = line.strip().lower()
        if lowered.startswith("reason:"):
            reason = line.split(":", 1)[1].strip()
        elif lowered.startswith("verdict:"):
            verdict = line.split(":", 1)[1].strip().lower()

    # Only the verdict line decides True/False, so a word like "no" inside
    # the reason can't fool us.
    if verdict.startswith("yes"):
        is_good = True
    elif verdict.startswith("no"):
        is_good = False
    else:
        print(f"  [warning] Unexpected LLM answer {text!r}; treating as No")
        is_good = False

    return is_good, reason


# ---------------------------------------------------------------------------
# The shock: wraps the LLM so scikit-agent can "draw" from it like a distribution
# ---------------------------------------------------------------------------
class LLMGoodNewsShock:
    def __init__(self):
        self.headline = ""        # the "news of the day"; we update this each run
        self.last_reason = ""     # Gemini's one-liner, saved only so we can print it
        self.last_value = False   # the latest yes/no (True/False) the agent received

    def draw(self, *_args, **_kwargs):
        # "Drawing" from this shock = asking Gemini about today's headline.
        # We return ONLY the yes/no (True/False). The reason stays on the side.
        is_good, self.last_reason = analyze_headline(self.headline)
        self.last_value = is_good     # saved so the printout can show what the agent got
        return is_good


news_shock = LLMGoodNewsShock()


# ---------------------------------------------------------------------------
# The agent: it receives only good_news (True/False) and acts on that alone
# ---------------------------------------------------------------------------
def decision_rule(good_news):
    """Yes (True) -> Buy. No (False) -> Sell."""
    return "Buy" if good_news else "Sell"


SHARE_CHANGE = {"Buy": +1, "Sell": -1}


# ---------------------------------------------------------------------------
# The DBlock
# ---------------------------------------------------------------------------
def make_block(use_shocks: bool):
    """Build the DBlock.

    use_shocks=True  -> the LLM is hooked in through the `shocks` parameter
                        (the intended, official way).
    use_shocks=False -> backup: the LLM is called as the first dynamics step,
                        in case your scikit-agent version rejects a custom shock.
    """
    agent_steps = {
        # CONTROL: the agent chooses "action". ["good_news"] is its information
        # set = the ONLY thing it is allowed to look at when deciding.
        "action": ska.Control(["good_news"]),

        # Update how many shares the agent holds.
        "shares": lambda shares, action: shares + SHARE_CHANGE[action],
    }

    if use_shocks:
        return ska.DBlock(
            name="news_trading_stage",
            shocks={"good_news": news_shock},
            dynamics=agent_steps,
        )

    return ska.DBlock(
        name="news_trading_stage",
        dynamics={"good_news": lambda: news_shock.draw(), **agent_steps},
    )


def run_transition(pre_state):
    last_error = None
    for use_shocks in (True, False):
        try:
            block = make_block(use_shocks)
            state = dict(pre_state)
            if use_shocks:
                # scikit-agent expects shock values to already be in the state when
                # `transition` runs (its simulator draws them first). So we do the
                # same: "draw" every shock in the block's `shocks` (this is where
                # Gemini gets asked) and add the results to the starting state.
                for shock_name, shock in block.shocks.items():
                    state[shock_name] = shock.draw()
            post_state = block.transition(state, dr={"action": decision_rule})
            return post_state, ("shocks" if use_shocks else "dynamics (backup)")
        except Exception as error:
            last_error = error
            if use_shocks:
                print(f"  [note] `shocks` hook failed ({type(error).__name__}: {error}); "
                      "trying the backup")
    raise last_error


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------
def run_headline(headline: str):
    print(f"\nHeadline: {headline}")

    # Tell the shock which headline to read this round.
    news_shock.headline = headline

    # The starting state: the agent owns 10 shares.
    pre_state = {"shares": 10}

    post_state, hook = run_transition(pre_state)

    answer = "Yes" if news_shock.last_value else "No"
    print(f"Gemini's one-liner: {news_shock.last_reason}")
    print(f"Agent received:     {answer}  (good_news={news_shock.last_value})")
    print(f"Agent's action:     {post_state['action']}")
    print(f"State before: {pre_state}")
    print(f"State after:  {post_state}")
    print(f"(LLM hooked in via: {hook})")


if __name__ == "__main__":
    # No built-in headlines: you decide what news the agent reacts to.
    #
    #   Option 1 (type them in):    python main.py
    #   Option 2 (pass them in):    python main.py "Fed lowers rates" "Bank collapses"
    #
    # In PyCharm, Option 2 works through Run -> Edit Configurations -> Parameters.
    if len(sys.argv) > 1:
        for text in sys.argv[1:]:
            run_headline(text)
    else:
        print("Type a news headline and press Enter. Press Enter on an empty line to quit.")
        while True:
            text = input("\nHeadline> ").strip()
            if not text:
                break
            run_headline(text)