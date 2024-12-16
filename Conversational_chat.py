from langchain_google_genai import GoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from langchain.memory import ConversationBufferWindowMemory
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
import os
import traceback

# Load environment variables
load_dotenv()

# Configuration from environment variables
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_NAME = os.getenv('DB_NAME')

# Create SQLAlchemy engine
db = SQLDatabase.from_uri(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")

# Initialize Google Generative AI model
try:
    llm = GoogleGenerativeAI(model="gemini-pro", temperature=0, google_api_key=GOOGLE_API_KEY)
except Exception as e:
    print(f"Error initializing Google Generative AI model: {str(e)}")
    llm = None

# Prompt templates
history_check_prompt = PromptTemplate.from_template(
    "Given the following conversation history and a new question, determine if the question can be answered using the information in the history. If so, provide the answer. If not, respond with 'NEED_QUERY'.\n\n"
    "Conversation history:\n{history}\n\n"
    "New question: {question}\n\n"
    "Can this be answered from the history? If yes, provide the answer. If no, just respond with 'NEED_QUERY'."
)

sql_prompt_template = """
Create a MySQL query for the following question. Provide only the raw SQL query without formatting:
Question: {input}

Schema:
{table_info}

Relevant tables (top {top_k}):
{table_names}
"""

response_prompt = PromptTemplate.from_template(
    "SQL query: {query}\n"
    "Result: {result}\n"
    "Context: {history}\n"
    "Answer the question concisely: {question}"
)

# Create SQL query chain
try:
    sql_chain = create_sql_query_chain(
        llm,
        db,
        prompt=PromptTemplate(
            input_variables=["input", "top_k", "table_info", "table_names"],
            template=sql_prompt_template
        )
    )
except Exception as e:
    print(f"Error creating SQL query chain: {str(e)}")
    sql_chain = None

# Function definitions

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def check_history_for_answer(question, history):
    """
    Check if the question can be answered from the conversation history.
    
    Args:
    question (str): The user's input question.
    history (str): The conversation history.
    
    Returns:
    str: The answer from history or 'NEED_QUERY' if a new query is needed.
    """
    history_check_chain = LLMChain(llm=llm, prompt=history_check_prompt, verbose=True)
    return history_check_chain.run(question=question, history=history)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_sql_query(question):
    """
    Generate an SQL query based on the given question.
    
    Args:
    question (str): The input question.
    
    Returns:
    str: The generated SQL query or None if an error occurs.
    """
    if sql_chain is None:
        print("SQL query chain is not initialized.")
        return None
    
    try:
        return sql_chain.invoke({
            "question": question,  # Changed from "input" to "question"
            "top_k": 5,
            "table_info": db.get_table_info(),
            "table_names": ", ".join(db.get_usable_table_names())
        })
    except Exception as e:
        print(f"Error generating SQL query: {str(e)}")
        traceback.print_exc()
        return None

def clean_sql_query(query):
    """
    Clean the SQL query by removing markdown formatting and unnecessary characters.
    
    Args:
    query (str): The SQL query to clean.
    
    Returns:
    str: The cleaned SQL query.
    """
    cleaned_query = query.strip()
    if cleaned_query.startswith('```sql'):
        cleaned_query = cleaned_query[6:]
    if cleaned_query.endswith('```'):
        cleaned_query = cleaned_query[:-3]
    return cleaned_query.strip()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_response(query, result, question, history):
    """
    Generate a natural language response based on the SQL query, its result, and conversation history.
    
    Args:
    query (str): The SQL query.
    result (str): The result of the SQL query.
    question (str): The original input question.
    history (str): The conversation history.
    
    Returns:
    str: The generated natural language response.
    """
    response_chain = LLMChain(llm=llm, prompt=response_prompt, verbose=True)
    return response_chain.run(query=query, result=result, question=question, history=history)

def get_response(question, memory):
    """
    Process the user's question and return a response using conversation memory.
    
    Args:
    question (str): The user's input question.
    memory (ConversationBufferWindowMemory): The conversation memory object.
    
    Returns:
    str: The chatbot's response.
    """
    history = memory.load_memory_variables({})["history"]
    
    try:
        # First, check if the question can be answered from history
        history_response = check_history_for_answer(question, history)
        
        if history_response != "NEED_QUERY":
            # The question was answered from history
            memory.save_context({"input": question}, {"output": history_response})
            return history_response

        # If we need a new query, proceed with the existing flow
        if llm is None or sql_chain is None:
            return "Sorry, I can't process your request due to a configuration issue. Please try again later or contact support."

        sql_query = generate_sql_query(question)
        
        if sql_query is None:
            return "I couldn't generate a valid SQL query. Could you rephrase your question?"

        cleaned_sql_query = clean_sql_query(sql_query)
        result = db.run(cleaned_sql_query)
        response = generate_response(cleaned_sql_query, result, question, history)
        
        memory.save_context({"input": question}, {"output": response})
        
        return response
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in get_response: {str(e)}\n{error_trace}")
        return f"An error occurred: {str(e)}"

def main():
    """Main function to run the chatbot."""
    print("Welcome to the T-Shirt Inventory Q&A Chatbot powered by Ideal Management Group")
    print("Ask questions about t-shirts and discounts. Type 'exit' to quit.")

    memory = ConversationBufferWindowMemory(k=5)  # Remember last 5 exchanges

    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() == 'exit':
            print("Thank you for using the T-Shirt Inventory Q&A Chatbot. Goodbye!")
            break
        
        response = get_response(user_input, memory)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    main()