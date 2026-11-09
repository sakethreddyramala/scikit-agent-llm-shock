# Scikit-Agent LLM Chance Variable Prototype

A research prototype integrating Google Gemini (via free-tier cloud API) as an exogenous "chance variable" within a `scikit-agent` `DBlock` transition function.

## Overview
This script demonstrates how unstructured text data (such as financial news headlines) can be processed by a cloud LLM to generate stochastic environment shocks, driving dynamic agent control decisions (`Buy`/`Sell`) and state updates within economic simulation loops.

## Tech Stack
* **Framework:** `scikit-agent`
* **LLM Provider:** Google Gemini Flash (via `openai` Python client configured for Google AI Studio)
* **Environment:** Python / PyCharm
