import streamlit as st

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
            st.success(f"Comment: {comment}")
