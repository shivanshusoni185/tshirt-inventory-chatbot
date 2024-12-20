﻿# tshirt-inventory-chatbot
# T-Shirt Inventory Q&A Chatbot Flask API

## Overview

This project implements a Flask-based API for a T-Shirt Inventory Q&A Chatbot. It uses Google's Generative AI model to generate SQL queries and natural language responses based on user questions about t-shirt inventory. The chatbot interacts with a MySQL database to retrieve and process inventory information.

## Features

- Natural language processing of user queries about t-shirt inventory
- Automatic generation of SQL queries based on user questions
- Integration with Google's Generative AI for enhanced response generation
- RESTful API endpoints for easy integration with front-end applications
- Error handling and retry mechanisms for improved reliability

## Requirements

- Python 3.7+
- MySQL database
- Google API key for Generative AI

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/shivanshusoni185/tshirt-inventory-chatbot.git
   cd tshirt-inventory-chatbot
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the project root directory with the following content:
   ```
   GOOGLE_API_KEY=your_google_api_key
   DB_USER=your_database_user
   DB_PASSWORD=your_database_password
   DB_HOST=your_database_host
   DB_NAME=your_database_name
   ```

## Usage

1. Start the Flask server:
   ```
   python app.py
   ```

2. The API will be available at `http://localhost:5000`

## API Endpoints

### 1. Ask a Question

- **URL:** `/ask`
- **Method:** `POST`
- **Data Params:** 
  ```json
  {
    "question": "Your question about t-shirt inventory"
  }
  ```
- **Success Response:**
  - **Code:** 200
  - **Content:** 
    ```json
    {
      "response": "Natural language answer to the question"
    }
    ```
- **Error Response:**
  - **Code:** 400
  - **Content:** 
    ```json
    {
      "error": "No question provided"
    }
    ```

### 2. Health Check

- **URL:** `/health`
- **Method:** `GET`
- **Success Response:**
  - **Code:** 200
  - **Content:** 
    ```json
    {
      "status": "healthy"
    }
    ```

## Configuration

The application uses environment variables for configuration. Ensure that all required variables are set in the `.env` file:

- `GOOGLE_API_KEY`: Your Google API key for accessing Generative AI services
- `DB_USER`: MySQL database username
- `DB_PASSWORD`: MySQL database password
- `DB_HOST`: MySQL database host address
- `DB_NAME`: Name of the MySQL database

## Database Schema

Ensure your MySQL database has a table structure compatible with the t-shirt inventory queries. A sample schema might include:

- `tshirts` table:
  - `id` (INT, Primary Key)
  - `color` (VARCHAR)
  - `size` (VARCHAR)
  - `quantity` (INT)
  - `price` (DECIMAL)

## Error Handling

The application includes error handling mechanisms:
- Invalid or missing questions return a 400 Bad Request response.
- Internal server errors are caught and return appropriate error messages.
- The `tenacity` library is used for retrying API calls in case of temporary failures.

## Security Considerations

- The API key and database credentials are stored as environment variables to prevent exposure.
- Ensure to implement proper authentication and authorization before deploying in a production environment.
- Use HTTPS in production to encrypt data in transit.

## Deployment

For production deployment:
1. Use a production-grade WSGI server like Gunicorn.
2. Set `debug=False` in the Flask app configuration.
3. Implement proper logging and monitoring.
4. Consider using a reverse proxy like Nginx for better security and performance.

## Contributing

Contributions to improve the chatbot or extend its functionality are welcome. Please follow these steps:
1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes and commit (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a new Pull Request

