import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import pandas as pd
import os

# --- App Configuration ---
st.set_page_config(
    page_title="LSTM Prediction Models",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Caching for Model and Data Loading ---
# This decorator ensures that the models and data are loaded only once,
# making the app much faster on subsequent interactions.

@st.cache_resource
def load_word_prediction_artifacts():
    """Loads the pre-trained model, tokenizer, and max_seq_len for word prediction."""
    model_path = 'next_word_model.h5'
    tokenizer_path = 'tokenizer.pkl'
    max_seq_len_path = 'max_seq_len.pkl'

    if not all(os.path.exists(p) for p in [model_path, tokenizer_path, max_seq_len_path]):
        return None, None, None

    model = tf.keras.models.load_model(model_path)
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open(max_seq_len_path, 'rb') as handle:
        max_seq_len = pickle.load(handle)
    return model, tokenizer, max_seq_len

@st.cache_resource
def load_song_prediction_artifacts():
    """Loads the pre-trained model and test data for song prediction."""
    model_path = 'hit_song_model.h5'
    test_data_path = 'hit_song_test_samples.pkl'

    if not all(os.path.exists(p) for p in [model_path, test_data_path]):
        return None, None

    model = tf.keras.models.load_model(model_path)
    with open(test_data_path, 'rb') as handle:
        test_data = pickle.load(handle)
    return model, test_data

# --- Prediction Functions ---

def predict_next_words(model, tokenizer, max_seq_len, seed_text, next_words):
    """Predicts the next words in a sequence using the trained LSTM model."""
    output_text = seed_text
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([output_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len - 1, padding='pre')
        
        # Predict probabilities for the next word
        predicted_probs = model.predict(token_list, verbose=0)
        # Get the index of the word with the highest probability
        predicted_index = np.argmax(predicted_probs, axis=-1)[0]
        
        output_word = ""
        # Find the word corresponding to the predicted index
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        output_text += " " + output_word
    return output_text

# --- Main Application UI ---

st.title("LSTM-Powered Prediction Suite 🚀")
st.markdown("An interactive web application to demonstrate the predictive power of LSTM models for two distinct tasks.")
st.markdown("---")

# --- Sidebar for Navigation ---
st.sidebar.title("Navigation")
st.sidebar.markdown("Select a model below to get started.")
app_mode = st.sidebar.radio(
    "Choose a Prediction Model",
    ["Next Word Prediction", "Hit Song Prediction"]
)
st.sidebar.markdown("---")
st.sidebar.info("This app uses pre-trained Keras models. Ensure all `.h5` and `.pkl` files from the notebook are in the same directory as this script.")

# --- Program 1: Next Word Prediction ---
if app_mode == "Next Word Prediction":
    st.header("✍️ NLP Sequence Prediction using LSTM")
    st.markdown("This model, trained on an SMS dataset, predicts the next word(s) in a sentence. It helps in understanding how LSTMs learn sequential text patterns.")

    word_model, tokenizer, max_seq_len = load_word_prediction_artifacts()
    
    if word_model is None:
        st.error(
            "**Error: Model files for Next Word Prediction not found!**\n\n"
            "Please ensure `next_word_model.h5`, `tokenizer.pkl`, and `max_seq_len.pkl` are in the same directory as `app.py`."
        )
    else:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Enter Your Text")
            default_sentences = ["i am not able", "please help me", "how can i", "my account is"]
            seed_text = st.text_input("Start typing a sentence:", default_sentences[0])
            num_words_to_predict = st.slider("Number of words to predict:", min_value=1, max_value=10, value=3, step=1)

            if st.button("Predict Next Words", type="primary"):
                if seed_text:
                    with st.spinner("🧠 Predicting..."):
                        result = predict_next_words(word_model, tokenizer, max_seq_len, seed_text, num_words_to_predict)
                        st.success("Prediction Complete!")
                        st.subheader("Result:")
                        st.markdown(f"> **Original:** `{seed_text}`")
                        st.markdown(f"> **Full Prediction:** `{result}`")
                else:
                    st.warning("Please enter some text to start.")
        
        with col2:
            st.subheader("Example Prompts")
            st.markdown("Try these sentences from the original notebook:")
            for sentence in default_sentences:
                st.code(sentence, language=None)


# --- Program 2: Hit Song Prediction ---
elif app_mode == "Hit Song Prediction":
    st.header("🎵 Predicting Hit Songs using LSTM")
    st.markdown("Can a song's future success be predicted from its first week of data? This model attempts to do just that by analyzing the initial 7-day streaming and chart rank patterns to predict if a song will become a **Top 50 hit** in the following month.")

    song_model, test_data = load_song_prediction_artifacts()

    if song_model is None:
        st.error(
            "**Error: Model files for Hit Song Prediction not found!**\n\n"
            "Please ensure `hit_song_model.h5` and `hit_song_test_samples.pkl` are in the same directory as `app.py`."
        )
    else:
        X_seq_test = test_data['X_seq_test']
        y_test = test_data['y_test']
        titles = test_data['titles']
        artists = test_data['artists']

        st.subheader("Select a Song to Analyze")
        
        # Create a more descriptive list for the selectbox
        song_options = [f"{title} - by {artist}" for title, artist in zip(titles, artists)]
        selected_song_display = st.selectbox(
            "Choose from a random sample of test songs:",
            options=song_options,
            index=10  # A default selection
        )
        
        selected_index = song_options.index(selected_song_display)

        # Get the data for the selected song
        song_sequence_data = X_seq_test[selected_index]
        actual_result = "Hit" if y_test[selected_index] == 1 else "Non-Hit"

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Initial 7-Day Performance")
            st.write("This is the (normalized) input data the model uses for its prediction.")
            df_display = pd.DataFrame(
                song_sequence_data,
                columns=['Streams (normalized)', 'Rank (normalized)'],
                index=[f'Day {i+1}' for i in range(7)]
            )
            st.dataframe(df_display, use_container_width=True)
            st.caption("Note: 'Rank' is normalized; a lower value means a better chart position.")

        with col2:
            st.markdown(f"#### Song Details")
            st.info(f"**Artist(s):** {artists[selected_index]}")
            st.warning(f"**Actual Outcome:** This song was a **{actual_result}**.")

            if st.button("Predict Hit Potential", type="primary"):
                with st.spinner("🤖 Analyzing song data..."):
                    # Reshape for single prediction: (1, timesteps, features)
                    input_data = np.expand_dims(song_sequence_data, axis=0)
                    
                    prediction_proba = song_model.predict(input_data, verbose=0)[0][0]
                    prediction_result = "Likely to be a Hit" if prediction_proba > 0.5 else "Not likely to be a Hit"
                    
                    st.subheader("Prediction Result:")
                    
                    if prediction_result == "Likely to be a Hit":
                        st.success(f"**{prediction_result}**")
                        st.progress(prediction_proba)
                        st.metric(label="Hit Confidence", value=f"{prediction_proba:.2%}")
                    else:
                        st.error(f"**{prediction_result}**")
                        st.progress(prediction_proba)
                        st.metric(label="Hit Confidence", value=f"{prediction_proba:.2%}")
