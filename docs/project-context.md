# OrderPilot: Project Context

## Name
- **Project:** OrderPilot: an evaluated e-commerce support agent
- **Repo:** `orderpilot-agent`
- **Python package:** `orderpilot`
- **GitHub description:** OrderPilot: an evaluated e-commerce support agent built with LangGraph — order tracking, returns and cancellations, with a τ-bench-style evaluation suite.
- **GitHub topics:** `langgraph`, `langchain`, `llm-agents`, `agent-evaluation`, `tool-calling`, `ecommerce`, `customer-support`

## Goal
Build an e-commerce agent that helps customers with tasks such as tracking an order and returning an order. Then evaluate the agent.

## Framework
- LangChain and LangGraph
- Architecture: a tool-based agent, or possibly a multi-agent system with a supervisor/worker setup

## Data
- Looker theLook E-commerce BigQuery dataset
- Uploaded to the project: `users.csv`, `orders.csv`, `order_items.csv`, `products.csv`, `distribution_centers.csv`
- Kept locally because of the upload size limit: `inventory_items.csv`, `events.csv`

## Purpose
A portfolio project for GitHub, and possibly a LinkedIn post, to show skills in building agents and especially in evaluating them.
