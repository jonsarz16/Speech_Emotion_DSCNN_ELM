from streamlit_option_menu import option_menu 
from setup import *
import tensorflow as tf
import pandas as pd
import IPython.display as ipd
import librosa, librosa.display
import os

pd.set_option('precision', 2)
# pd.reset_option('display.float_format')

# st.header("Speech Emotion Recognition")
local_css("style.css")
remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

# with st.sidebar:
#     st.write("Speech Emotion")
#     agree = st.checkbox('Use Test file')
# if agree:
#     test_audio = "YAF_back_angry.wav"
#     data_visual_improved(test_audio)

# else:
# st.header("Speech Emotion Recognition")
st.markdown("<h4 style='text-align: center; margin-top: -70px'>Speech Emotion Recognizer</h4>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center; margin-top: -40px'>John Ferrie Sarigumba | Raven Joy Tudtud | Steven Badang </h6>", unsafe_allow_html=True)
container = st.empty()
file_audio = container.file_uploader("", type=['wav'])
selected = option_menu(
    None,
    options=["Improved Algorithm", "Baseline Algorithm", "Performance Comparison"],
    icons=["arrow-up-right-circle","arrow-repeat","window-stack"],
    menu_icon="cast",
    default_index=1,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#3CAEA3", "display": "inline"},
        "icon": {"color": "black", "font-size": "25px"}, 
        "nav-link": {"font-size": "12px", "text-align": "left", "margin":"0px", "--hover-color": "#ffe100"},
        "nav-link-selected": {"background-color": "#0F2557"},}
)
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
emotions1 = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
x = [[ '😡','angry'], [ '🤮','disgust'], ['🥶','fear'],['😀','happy'], ['🤴','neutral'], ['😢','sad'],['😱','surprise']]
                
hehe = ['😡','🤮','🥶','😀','🤴','😢','😱']
if selected == "Improved Algorithm":
    # st.warning("to be continued")
    col1, col2 = st.columns(2)
    container1 = st.empty()
    
    with col1:
        # file_audio1 = container1.file_uploader("", type=['mp3','wav'])
        
        if file_audio is not None:
            predict1 = container1.button("Predict", key = 'improved')
            data_visual(file_audio, 0)

    with col2:
        try:
            if predict1:
                
                # # st.title("Prediction Results")
                # st.markdown("<h4 style='text-align: center;'>Prediction Results</h4>", unsafe_allow_html=True)
                # st.warning("Processing ...")
                       
                # container1.empty()
                st.markdown("<h4 style='text-align: center;'>Prediction Results</h4>", unsafe_allow_html=True)
             
                st.write("Predicted Emotion:  **{}** " 
                .format(modified_predicted_emotion('mel_spectrogram_0.png').upper()))

                st.write("Actual Emotion:  **{}** " 
                .format(get_actual_emotion(file_audio.name).upper()))
                
                var_mod = classify_modified('mel_spectrogram_0.png') * 100
                
                df_mod = pd.DataFrame(x, columns=["","Predicted emotion"])
                df_mod['Percentage'] = var_mod[0]
                df_mod['Percentage'] = df_mod['Percentage'].apply(lambda x: float("{:,.2f}".format(x)))
                df_mod['.'] = "%"
                
                df_mod = df_mod.style.background_gradient()
                st.table(df_mod)
        except:
            pass
        

if selected == "Baseline Algorithm":
    col1, col2 = st.columns(2)
    container2 = st.empty()
    
    with col1:
        if file_audio is not None:
            predict2 = container2.button("Predict", key = 'improved')
            data_visual(file_audio, 1)

    with col2:
        try:
            if predict2:

                st.markdown("<h4 style='text-align: center;'>Prediction Results</h4>", unsafe_allow_html=True)
             
                st.write("Predicted Emotion:  **{}** " 
                .format(baseline_predicted_emotion('mel_spectrogram_1.png').upper()))

                st.write("Actual Emotion:  **{}** " 
                .format(get_actual_emotion(file_audio.name).upper()))
    
                # st.write("Probability")
                # st.write(probabilities('mel_spectrogram.png'))
                
                var = classify('mel_spectrogram_1.png') * 100
                
                df = pd.DataFrame(x, columns=["","Predicted emotion"])
                df['Percentage'] = var[0]
                df['Percentage'] = df['Percentage'].apply(lambda x: float("{:,.2f}".format(x)))
                df['.'] = "%"
                
                df = df.style.background_gradient()
                st.table(df)
            
        except:
            pass

if selected == "Performance Comparison":
    col1, col2 = st.columns(2)
    with col1:
        st.success("Baseline Algorithm")
        if file_audio is not None:
            st.write("Predicted Emotion:  **{}** " 
            .format(baseline_predicted_emotion('mel_spectrogram_1.png').upper()))

            st.write("Actual Emotion:  **{}** " 
            .format(get_actual_emotion(file_audio.name).upper()))

            # st.write("Probability")
            # st.write(probabilities('mel_spectrogram.png'))
            
            var = classify('mel_spectrogram_1.png') * 100
            
            df = pd.DataFrame(x, columns=["","Predicted emotion"])
            df['Percentage'] = var[0]
            df['Percentage'] = df['Percentage'].apply(lambda x: float("{:,.2f}".format(x)))
            df['.'] = "%"
            
            df = df.style.background_gradient()
            st.table(df)
    
    with col2:
        st.warning("Improved Algorithm")
        if file_audio is not None:
            # st.warning("Processing ...")
            st.write("Predicted Emotion:  **{}** " 
            .format(modified_predicted_emotion('mel_spectrogram_0.png').upper()))

            st.write("Actual Emotion:  **{}** " 
            .format(get_actual_emotion(file_audio.name).upper()))

            # st.write("Probability")
            # st.write(probabilities('mel_spectrogram.png'))
            
            var = classify_modified('mel_spectrogram_0.png') * 100
            
            df = pd.DataFrame(x, columns=["","Predicted emotion"])
            df['Percentage'] = var[0]
            df['Percentage'] = df['Percentage'].apply(lambda x: float("{:,.2f}".format(x)))
            df['.'] = "%"
            
            df = df.style.background_gradient()
            st.table(df)





