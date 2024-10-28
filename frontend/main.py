import streamlit as st
import requests

# Set up the Streamlit app title
st.title("API Search Application")

# Create a text input for search queries
query = st.text_input("Enter your search query")

# Define a function to fetch results from the API
def fetch_results(search_query):
    # Placeholder API URL (replace with your actual API endpoint)
    api_url = "https://api.example.com/search"
    # Send a GET request with the search query as a parameter
    response = requests.get(api_url, params={"q": search_query})
    
    # Check if the request was successful
    if response.status_code == 200:
        return response.json().get("results", [])
    else:
        st.error("Failed to fetch results. Please try again.")
        return []

# Display results when there is a search query
if query:
    st.write(f"Showing results for: **{query}**")
    
    # Fetch results from the API
    results = fetch_results(query)
    
    # Display results in a list format
    if results:
        for idx, result in enumerate(results):
            st.write(f"{idx + 1}. {result['title']}")
            st.write(result['description'])
    else:
        st.write("No results found.")
