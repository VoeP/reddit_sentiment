import streamlit as st
import requests

# Api constants
URL = "http://localhost:8000/predict"

# Setup page
st.set_page_config(
    page_title="Comment Sentiment Analysis",
    page_icon="📝"
)

st.markdown("# Comment Sentiment Analysis")

st.markdown("## Enter your comment below for analysis:")

# Main form
with st.form(key='comment_form'):
    comment = st.text_input("Comment")
    submit_button = st.form_submit_button("Predict Sentiment")

    # Handle form submission
    if submit_button:
        # Check there is a comment
        if comment == "":
            st.error("Please enter a comment for analysis")
        else:
            # Make an API request
            params = {"message": comment}
            response = requests.get(URL, params=params)
            # Check the response is valid
            if response.status_code != 200:
                st.error("There was an error with the API request.")
            else:
                # Display the results
                results = response.json()
                st.markdown(f"""### Comment
                            {results['comment']}""")
                st.markdown(f"""### Sentiment Score
                            {results['sentiment']}""")
