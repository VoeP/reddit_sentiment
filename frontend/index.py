import streamlit as st
import streamlit.components.v1 as components
import requests, html
import pandas as pd
import matplotlib.pyplot as plt
import pathlib

# Api constants
deployment = st.secrets.deployment.deployment
if deployment == "LOCAL":
    URL = "http://localhost:8000/"
elif deployment == "DOCKER_LOCAL":
    URL = "http://localhost:8080/"
elif deployment == "DOCKER_GCP":
    URL = "https://test-cont-bcmupioatq-nw.a.run.app/"

# Get the endpoint from secrets
endpoint = st.secrets.deployment.endpoint
# Add it to the url
URL += endpoint

# Setup page
st.set_page_config(
    page_title="Comment Sentiment Analysis",
    page_icon="📝"
)

# Add external CSS to style the Streamlit app
css = pathlib.Path('frontend/style.css').read_text()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

st.markdown("<div class='my-markdown'>Comment Sentiment Analysis</div>", unsafe_allow_html=True)

st.markdown("## Enter your comment below for analysis:")

def clear_text():
    st.session_state["comment_input"] = ""

# Main form
import pathlib
with st.form(key='comment_form'):
    comment = st.text_area("Comment", key="comment_input")
    col1, col2 = st.columns(2)
    with col1:
        submit_button = st.form_submit_button("Predict Sentiment")
    with col2:
        clear_button = st.form_submit_button("Reset", on_click=clear_text)

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
                st.markdown("### Comment")

                # Incredibly jank way of displaying the comment without any markdown or latex
                comment_with_line_breaks = html.escape(results['comment']).replace('\n', '<br>')
                # Set height and stuff (mega jank)
                # I would like this to be more dynamic but it's a pain because of line breaks and text wrapping
                scroll = True
                height = 150

                # Write the html itself with styling for font
                components.html(f"<p style=\"font-family: Sans-Serif\">{comment_with_line_breaks}</p>", height=height, scrolling=scroll)

                # Now the rest of the stuff (easy)
                st.markdown("### Sentiment")
                st.write(results['sentiment'])
                st.markdown("### Confidence")
                st.write(results['confidence'])


st.markdown("<div class='my-markdown2'>Currently happening on WSB:</div>", unsafe_allow_html=True)

st.markdown("### Sentiment in comments")

st.markdown("### Composition of emotions:")

df = pd.read_csv("data_for_plotting/comment_data.csv")
if df is not None:

    #st.write(df)
    sum_joy = df["joy"].sum()
    sum_optimism = df["optimism"].sum()
    sum_anger = df["anger"].sum()
    sum_sadness = df["sadness"].sum()

    keys=["joy", "optimism", "anger", "sadness"]
    data=[sum_joy,
        sum_optimism,
        sum_anger,
        sum_sadness]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Plotting the pie chart
    ax1.pie(data, labels=keys, autopct='%1.1f%%')
    ax1.set_title('Emotional pie')

    # Plotting the bar chart
    ax2.bar(keys, data)
    ax2.set_xlabel('Categories')
    ax2.set_ylabel('Values')
    ax2.set_title('Emotion bars')

    plt.tight_layout()

    st.pyplot(fig)


st.markdown("### Composition of sentiment and upvotes:")

if df is not None:
    sents=df.groupby("sentiment").count()["text"]
    grouped_df = df.groupby("sentiment")["score"].sum().reset_index()
    plt.figure(figsize=(18, 6))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Plotting the bar chart for scores
    ax1.bar(grouped_df["sentiment"], grouped_df["score"])
    ax1.set_title('Score of each sentiment by upvotes')

    # Plotting the bar chart for comment counts
    ax2.bar(grouped_df["sentiment"], sents)
    ax2.set_xlabel('Categories')
    ax2.set_ylabel('Values')
    ax2.set_title('Number of comments in each sentiment class')
    plt.tight_layout()

    st.pyplot(fig)


st.markdown("### Sentiment in posts:")

df_posts = pd.read_csv("data_for_plotting/post_data.csv", index_col="ids")

if df_posts is not None:
    df = df_posts
    fig, ax1 = plt.subplots(figsize=(10, 6))

    colors = ['purple', 'green', 'orange', 'red', 'black']
    ax1.set_xlabel('Post ids', fontsize=12)
    ax1.set_ylabel('emotion strength', color='black', fontsize=12)
    ax1 = df[["joy","optimism","anger","sadness"]].plot(kind='bar', ax=ax1, color=colors, width=0.6, position=0)
    ax1.tick_params(axis='y', labelcolor='black')

    ax2 = ax1.twinx()
    color = 'blue'
    ax2.set_ylabel('sentiment', color=color, fontsize=12)
    ax2 = df['sentiment'].plot(kind='bar', ax=ax2, color=color, width=0.15, position=1)
    ax2.tick_params(axis='y', labelcolor=color)

    # Create custom legend labels with colors
    legend_labels = ['Joy', 'Optimism', 'Anger', 'Sadness', 'Average Sentiment']

    # Plotting dummy lines to create legend with custom labels and colors
    #for i, label in enumerate(legend_labels):
    #    ax1.plot([], label=label, color=colors[i])
    plt.title('Post sentiment breakdown')
    ax1.legend(loc='upper left', title='Legend', bbox_to_anchor=(1.05, 1))
    ax2.legend(loc='upper right', title='Legend', labels=['Average sentiment'], bbox_to_anchor=(1.25, 1.15))

    plt.tight_layout()
    st.pyplot(fig)


st.dataframe(df_posts["titles"])
