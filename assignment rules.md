# ESILV Smart Assistant

This project aims to design and implement an intelligent chatbot dedicated to the ESILV engineering school. The system should be capable of:
• Answering questions about programs, admissions, and courses using ESILV website and 
internal documentation.
• Interacting with users to collect contact details for follow-up or registration.
• Coordinating multiple specialized agents (retrieval, form-filling, orchestration) to handle 
complex user queries.

The chatbot must integrate both retrieval-augmented generation (RAG) for factual answers and 
multi-agent coordination for structured interactions.
Students can deploy it locally with the Google AI platform (GCP).
The app still runs locally or on a server, but the LLM calls go to Google
A Streamlit front-end will serve as the user interface for chatting, document uploads, and admin 
visualization.

Source Code (Github, or any VC): Full implementation with version control, clear 
README. Should include: app/, agents/, ui/, ingestion/, and notebooks/