# CLAUDE.md

## Project Overview

This project is a conversational sitcom episode recommender system.

The system helps users discover sitcom episodes to watch based on:
- current mood
- emotional preferences
- novelty preferences
- conversational refinement
- critique-based feedback

The system is NOT a general chatbot.
The architecture should remain recommendation-focused and deterministic where possible.

---

# Core Architecture

The system is divided into 3 major modules:

## 1. UI Layer
Responsibilities:
- render questions
- display episode options
- display recommendations
- collect user input

The UI should NOT:
- update vectors
- manage recommendation logic
- generate questions dynamically

---

## 2. Conversation Strategy Module (CSM)
Responsibilities:
- determine next interaction type
- decide whether to:
  - ask attribute question
  - ask item question
  - recommend episode
- manage exploration vs exploitation
- select question intent
- select question template

The CSM should remain deterministic.

The CSM should NOT:
- update vectors directly
- perform embedding math
- rank episodes
- query databases directly

---

## 3. Recommendation Engine
Responsibilities:
- maintain/update user vector
- retrieve candidate episodes
- rerank candidate episodes
- compute recommendation confidence
- return candidate summaries and rankings

The recommendation engine is the ONLY module allowed to update:
- intent vectors
- user vectors

The recommendation engine may contain:
- ML models
- reranking models
- embedding models

---

# Vector Philosophy

There are 3 important vector types:

## Intent Vector
Represents:
- current conversational preference state
- volatile preferences
- active mood signals

The intent vector uses dimensions in the range:
[-1, 1]

Interpretation:
- negative = avoidance
- positive = preference
- near zero = indifference/unknown

---

## User Vector
Represents:
- stabilized recommendation-facing state
- accumulated conversational evidence
- metadata preferences

The user vector is derived from:
- intent vector
- session refinement
- critique updates

---

## Episode Vector
Represents:
- episode mood/style profile
- semantic embedding
- structured trait representation

Episode vectors and user vectors must share the same core feature space.

---

# Metadata Philosophy

Metadata is NOT part of the core mood vector.

Examples:
- runtime
- release date
- season
- ratings
- show identity

Metadata is:
- stored separately
- deterministic
- used for filtering and reranking

Metadata preferences should remain stable unless explicitly changed by the user.

---

# Questioning Philosophy

Questions are generated using:
- intent graph relationships
- uncertainty-aware selection
- deterministic templates

Questions should:
- maximize information gain
- minimize conversational weirdness

Questions should NOT:
- be random
- over-narrow too early
- force rigid decision trees

---

# Recommendation Philosophy

Recommendation occurs in stages:

1. Retrieval
- vector search retrieves broad candidate set

2. Filtering
- hard constraints remove invalid candidates

3. Reranking
- reranker refines candidate ordering

4. Conversational Selection
- CSM decides whether to:
  - ask more questions
  - recommend now

---

# Exploration vs Exploitation

The system uses:
- epsilon-greedy exploration initially

Exploration should:
- remain constrained
- ask meaningful alternative questions
- avoid random irrelevant prompts

---

# Current MVP Constraints

DO NOT:
- implement RL
- implement Bayesian inference
- implement Thompson Sampling
- implement graph neural networks
- implement LangChain/LangGraph
- introduce microservices
- introduce distributed infrastructure

Prefer:
- deterministic systems
- modular architecture
- inspectable logic
- simple retrieval/reranking pipelines

---

# Coding Philosophy

Prefer:
- explicit interfaces
- thin modules
- composable components
- deterministic behavior

Avoid:
- premature abstraction
- unnecessary frameworks
- hidden magic behavior

All major architecture decisions should remain explainable and debuggable.