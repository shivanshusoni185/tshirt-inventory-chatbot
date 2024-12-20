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

# Custom prompt template for SQL query generation
# Custom prompt template for SQL query generation
sql_prompt_template = """
Given an input question, create a syntactically correct MySQL query to run. Do not include any markdown formatting or backticks in your response. Only provide the raw SQL query.

Here is the question: {input}

Given the following database schema:
{table_info}

Please write a SQL query to answer the question. If you need to, you can use the top {top_k} most similar tables:
{table_names}

"""

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

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_sql_query(question):
    """
    Generate an SQL query based on the given question.
    
    Args:
    question (str): The input question.
    
    Returns:
    str: The generated SQL query.
    """
    if sql_chain is None:
        print("SQL query chain is not initialized.")
        return None
    
    try:
        return sql_chain.invoke({
            "input": question,  # Changed from "question" to "input"
            "top_k": 5,
            "table_info": db.get_table_info(),
            "table_names": ", ".join(db.get_usable_table_names())
        })
    except Exception as e:
        print(f"Error generating SQL query: {str(e)}")
        print(f"Question: {question}")
        print(f"Table info: {db.get_table_info()}")
        print(f"Table names: {db.get_usable_table_names()}")
        traceback.print_exc()
        return None



class SQLOutputParser(BaseOutputParser):
    """Custom output parser for SQL queries."""
    def parse(self, text):
        """Parse the output text."""
        return text



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
def generate_sql_query(question):
    """
    Generate an SQL query based on the given question.
    
    Args:
    question (str): The input question.
    
    Returns:
    str: The generated SQL query.
    """
    if sql_chain is None:
        print("SQL query chain is not initialized.")
        return None
    
    try:
        return sql_chain.invoke({
            "question": question,
            "top_k": 5,
            "table_info": db.get_table_info(),
            "table_names": ", ".join(db.get_usable_table_names())
        })
    except Exception as e:
        print(f"Error generating SQL query: {str(e)}")
        print(f"Question: {question}")
        print(f"Table info: {db.get_table_info()}")
        print(f"Table names: {db.get_usable_table_names()}")
        traceback.print_exc()
        return None

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
    response_prompt = PromptTemplate.from_template(
        "Based on the SQL query: {query}\n"
        "And its result: {result}\n"
        "Previous conversation:\n{history}\n"
        "Please provide a natural language answer to the question: {question}"
    )
    response_chain = LLMChain(llm=llm, prompt=response_prompt,verbose=True)
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
        if llm is None or sql_chain is None:
            return "I'm sorry, but I'm currently unable to process your request due to a configuration issue. Please try again later or contact support."

        print(f"Generating SQL query for question: {question}")  # Debug print
        
        # Generate SQL query with retry
        sql_query = generate_sql_query(question)
        
        if sql_query is None:
            return "I'm sorry, but I couldn't generate a valid SQL query for your question. Could you please rephrase or ask a different question?"

        # Clean the SQL query
        cleaned_sql_query = clean_sql_query(sql_query)
        
        print(f"Generated SQL query: {cleaned_sql_query}")  # Debug print
        
        # Execute the cleaned query
        result = db.run(cleaned_sql_query)
        
        print(f"Query result: {result}")  # Debug print
        
        # Generate response based on the query result with retry
        response = generate_response(cleaned_sql_query, result, question, history)
        
        # Save the exchange to memory
        memory.save_context({"input": question}, {"output": response})
        
        return response
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in get_response: {str(e)}\n{error_trace}")
        return f"I apologize, but an error occurred while processing your request. Here's some technical information that might help diagnose the issue: {str(e)}"

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