# 🤖 Akash Chatbot

Akash Chatbot is a **smart AI assistant** built using **LangGraph**, **OpenAI**, and **Streamlit**. It supports natural language queries, tool-based tasks like calculations and stock price retrieval, and allows multiple chat sessions. The chatbot is fully interactive and can be accessed online for demonstration.

---

## 🌐 Live Demo

Try the chatbot in your browser:

[**Access Akash Chatbot**](https://akash-chatbot-fvalyflxnqmkpc3pq3qbbh.streamlit.app/)

You can also scan the QR code below to open it on mobile:

![QR Code](https://akash-chatbot-fvalyflxnqmkpc3pq3qbbh.streamlit.app/)

---

## ⚡ Features

- **Conversational AI** using OpenAI's GPT-4o-mini model.
- **Multi-tool integration**:
  - **Calculator**: Perform basic arithmetic operations.
  - **Stock Price Fetcher**: Get real-time stock prices using Alpha Vantage API.
  - **DuckDuckGo Search**: Quickly fetch search results via API.
- **Persistent chat threads**: Each conversation is saved using SQLite.
- **Status container**: Shows which tool is being used in real-time.
- **Rich Streamlit UI**: User-friendly and visually appealing interface.
- **Multi-thread management**: Switch between multiple chat sessions seamlessly.

---

## 🏗️ Tech Stack

- **Backend**:
  - [LangGraph](https://github.com/langgraph/langgraph)
  - [OpenAI GPT-4o-mini](https://platform.openai.com/)
  - Python 3.11
  - SQLite for chat history

- **Frontend**:
  - [Streamlit](https://streamlit.io/)
  - HTML & CSS customizations for enhanced UI

- **APIs & Tools**:
  - Alpha Vantage (Stock prices)
  - DuckDuckGo Search API

---

## 📁 Project Structure

