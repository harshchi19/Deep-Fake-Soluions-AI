import streamlit as st
import pandas as pd
import pickle
from sklearn.impute import SimpleImputer
import os
import joblib
import sklearn

# Load the trained model for Twitter Fake News Detection
model_path = "LogisticTwiiterss.pkl"
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Get feature names used during training
feature_names = [
    'UserID',
    'No Of Abuse Report',
    'No Of Rejected Friend Requests',
    'No Of Freind Requests Thar Are Not Accepted',
    'No Of Friends',
    'No Of Followers',
    'No Of Likes To Unknown Account',
    'No Of Comments Per Day'
]

# Main function to run the Streamlit app
def main():
    st.title("VeriGuard - Text Classification and Twitter Fake News Detection")
    st.sidebar.title('Navigation')

    # Sidebar options
    page = st.sidebar.selectbox('Select a page', ['Home', 'Text Classification', 'Twitter Fake Account', 'Chatbot'])

    # Home page
    if page == 'Home':
        st.header('**Welcome to VeriGuard, your personalized Text Classification and Twitter Fake Account App!**')
        st.image('1.jpeg', width=200)
        st.write('Explore VeriGuard')
        st.write('VeriGuard is your one-stop solution for text classification and Twitter fake news detection. Navigate through the sections below to discover its features:')
        st.write('***Home:*** Get started with an overview of VeriGuard.')
        st.write('Welcome to the world of VeriGuard!')
        st.write("Here, you'll find an intuitive interface designed to make text classification and Twitter fake news detection easy and efficient.")
        st.write('Use the navigation on the left to explore different sections and unlock the full potential of VeriGuard.')
        st.write('***Text Classification:*** Analyze text with precision.')
        st.write('In the Text Classification section, you can enter any text and let VeriGuard analyze it for you.')
        st.write('Discover whether the text is relevant or not, along with detailed probabilities.')
        st.write("VeriGuard's advanced algorithms provide accurate results at your fingertips.")
        st.write('***Twitter Fake News Detection:*** Detect fake news in Twitter data.')
        st.write('In the Twitter Fake News Detection section, you can input user data to predict whether a tweet is fake or not.')
        st.write("VeriGuard's trained model provides quick insights into the authenticity of Twitter data.")
        st.write('***Chatbot:*** Interact with our intelligent chatbot.')
        st.write('Engage in meaningful conversations with VeriGuard\'s chatbot.')

    # Text Classification page
    elif page == 'Text Classification':
        st.header('Text Classification')
        # Get user input
        text = st.text_area('Enter text to classify', '', height=200)

        # Make prediction and display results
        if st.button('Classify') and text:
            prediction, probabilities = predict_label(text)

            # Display prediction
            st.subheader('Prediction:')
            if prediction == 1:
                st.write('Relevant')
            else:
                st.write('Not Relevant')

            # Display probabilities
            st.subheader('Probabilities:')
            relevant_prob = probabilities[1]
            not_relevant_prob = probabilities[0]
            st.write(f'Relevant: {relevant_prob:.2f}')
            st.write(f'Not Relevant: {not_relevant_prob:.2f}')

    # Twitter Fake News Detection page
    elif page == 'Twitter Fake Account':
        st.header('Twitter Fake Account')
        st.write("Enter the following information from a Twitter user's profile to predict if it's fake or not:")

        # User inputs for Twitter Fake News Detection
        user_data = {
            'UserID': st.text_input("User ID"),
            'No Of Abuse Report': st.number_input("No Of Abuse Report", min_value=0, step=1),
            'No Of Rejected Friend Requests': st.number_input("No Of Rejected Friend Requests", min_value=0, step=1),
            'No Of Freind Requests Thar Are Not Accepted': st.number_input("No Of Freind Requests Thar Are Not Accepted", min_value=0, step=1),
            'No Of Friends': st.number_input("No Of Friends", min_value=0, step=1),
            'No Of Followers': st.number_input("No Of Followers", min_value=0, step=1),
            'No Of Likes To Unknown Account': st.number_input("No Of Likes To Unknown Account", min_value=0, step=1),
            'No Of Comments Per Day': st.number_input("No Of Comments Per Day", min_value=0, step=1)
        }

        # Convert user input to DataFrame
        user_df = pd.DataFrame(user_data, index=[0])

        # Ensure all required columns are present
        for feature in feature_names:
            if feature not in user_df.columns:
                user_df[feature] = 0

        # Ensure only relevant features are used for prediction
        user_df = user_df[feature_names]

        if st.button("Predict"):
            # Make prediction
            prediction = model.predict(user_df)[0]

            # Get the predicted class label
            if prediction == 0:
                st.success("Prediction: Not Fake")
            else:
                st.success("Prediction: Fake")

    # Chatbot page
    elif page == 'Chatbot':
        st.header('Chatbot')
        st.write('Click the link below to open the chatbot:')
        st.markdown("[Open Chatbot](https://mediafiles.botpress.cloud/d9a12241-981b-4e8e-8848-281ac1d564fb/webchat/bot.html)")

# Function to predict label and probabilities for text classification
def predict_label(text):
    # Load the saved model and vectorizer for text classification
    text_model = joblib.load('logistic_regression_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')

    # Vectorize the text
    text_vectorized = vectorizer.transform([text])

    # Make prediction and return results
    prediction = text_model.predict(text_vectorized)[0]
    probabilities = text_model.predict_proba(text_vectorized)[0]
    return prediction, probabilities

# Run the app
if __name__ == '__main__':
    main()
