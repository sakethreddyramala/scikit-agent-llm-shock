# Scikit-Agent LLM Chance Variable Prototype

A research prototype integrating Google Gemini (via free-tier cloud API) as an exogenous "chance variable" within a `scikit-agent` `DBlock` transition function.

## Overview
This script demonstrates how unstructured news text can be processed by a cloud LLM to generate stochastic environment shocks. The LLM evaluates headlines, outputs a deterministic market signal (`Yes`/`No`), and generates dynamic text explanations. The agent acts solely on the boolean market signal to adjust portfolio shares (`Buy`/`Sell`).

## Features
* **Exogenous Shock Integration:** Converts unstructured LLM output into `scikit-agent` `DBlock` shocks.
* **Interactive Mode:** Supports real-time user headline input via terminal or CLI arguments.
* **Fallback Handling:** Implements dual-hook routing (`shocks` and `dynamics` backup) to handle version discrepancies smoothly.

## Tech Stack
* **Framework:** `scikit-agent`
* **LLM API:** Google Gemini Flash / Flash-Lite (via `openai` Python client)
* **Environment:** Python / PyCharm
