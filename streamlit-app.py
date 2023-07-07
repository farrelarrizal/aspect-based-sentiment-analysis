import re
import streamlit as st
import time
from PIL import Image
import requests
from streamlit_modal import Modal
import streamlit.components.v1 as components
from stqdm import stqdm
import matplotlib.pyplot as plt
import pandas as pd
from langdetect import detect
import random 
# from fpdf import FPDF
# import base64
# im = Image.open(requests.get(url, stream=True).raw)
HEADER_URL = 'assets/header.jpg'
image = Image.open(HEADER_URL)
st.image(image)
st.markdown('# ReviewPro')



st.markdown('**_Boost your business with Aspect Based Sentiment Analysis!_**')
st.markdown('**_Analyze, Enhance, and Succeed Your Business!_**')
st.warning('This project is still under research and development', icon="⚠️")
modal = Modal(key="Demo Key", title="Aspect Based Sentiment Analaysis")
open_modal = st.button(":information_source: About This Project", use_container_width=True)

if open_modal:
    modal.open()

if modal.is_open():
    with modal.container():
        st.markdown('## :innocent: About This Project')
        st.markdown('This project is a part of my final project in my bachelor degree. This project is still under research and development. This project is a web-based application that can be used to analyze reviews from customers. This project is built using Python and Streamlit. The model is built using PyABSA library. The model is trained using Indonesian and English dataset. The model is trained using BERT and DistilBERT. The model is trained using 3 classes of sentiment (positive, neutral, negative).')

        st.markdown('## :warning: Limitations')
        st.markdown('Model trained using Indonesian and English dataset. The model is trained using BERT base (YangHeng Deberta & Indobert by Indobenchmark). The model has training on Attraction Domain in Bali Province. So the model may not be accurate if you use it for other domains.')

        #special thanks
        st.markdown('## :heart: Special Thanks')
        st.markdown('Renny Pradina, S.T., M.T. as my supervisor, Radityo Prasetianto Wibowo, S.Kom, M.Kom. as my co-supervisor, Ali Arham as Lead ML Engineer in Tech Company, and all my friends who support me to finish this project.')
st.divider()

with st.form(key="single_text_form"):
    st.markdown('#### Just enter your text and get a great analysis! :sunglasses:')

    example_review = [
        "Beautiful temple at a beautiful location. I couldn't go for the sunset but I could imagine its beauty here. When you come up from the beach make a right turn to the cafes where you will have the best view of the sunset. However you must give them your custom to have nothing blocking your view.",
        "Very nice environment and we so happened joined the Art & Food Festival.  Full of celebration mode. The natural landscape is excellent, with holy spring water and temple. Good venue for watching denser.",
        "[B-ASP]pantai kuta[E-ASP] adalah tempat paling terpopuler. tetapi pantai ini terlalu kotor banyak [B-ASP]sampah[E-ASP] plastik berserakan. [B-ASP]Pemandangan[E-ASP] pantai sangat indah. Walaupun sebagian membawa [B-ASP]anjing[E-ASP] namun sangat menyenangkan.",
        "The beach is very nice, but there are many plastic waste scattered. ",
        "tempat waterboom terbaik dengan segala wahana yang sangat beragam. Tempat makan yang cukup banyak sehingga orang tua tidak terlalu bosan. Jangan lupa untuk membawa handuk agar tidak basah."
    ]

    text = st.text_area(label='input text with aspect to get more accurate analysis',label_visibility="visible", value=random.choice(example_review))
    language = st.radio('Select Text Language:', ('auto-detect', 'english', 'indonesia')) 
    single_predict = st.form_submit_button("Let's Go!")

if single_predict:
    aspect_tag = False
    # find [B-ASP] and [E-ASP]
    if '[B-ASP]' in text and '[E-ASP]' in text:
        aspect_tag = True
    
    st.snow()
    with st.spinner('analyzing your text:'):
        if language == 'auto-detect':
            language = detect(text)
        elif language == 'english':
            language = 'en'
        elif language == 'indonesia':
            language = 'id'

        if language == 'id' or language == 'en':
            header = {
                'Content-Type': 'application/json',
            }
            
            data = {
                'text': text,
                'lang': language,
                'aspect_tag': aspect_tag
            }

            if not aspect_tag:
                st.warning('Extraction aspect may be not accurate, add manual aspect to get more accurate polarity  :sweat_smile:', icon="⚠️")

            response = requests.post('http://localhost:8080/predict', headers=header, json=data).json()

            if response['status_code'] == 200:
                df = pd.DataFrame.from_dict(response).drop(columns=['status_code', 'message'])
                vc = df.sentiment.value_counts().to_dict()

                # text = df.text[0]
                # temp_text = text.split()
                # st.write(text[7:23].replace(text[7:23],f':green[{text[7:23]}]'))
                st.dataframe(df, use_container_width=True, hide_index=True)

                # Pie chart, where the slices will be ordered and plotted counter-clockwise:
                aspect_labels = list(vc.keys())
                sizes = list(vc.values())
                # labels = ["Postive", "Neutral", "Negative"]
                # sizes = [pos_count, neu_count, neg_count]
                # explode = (0, 0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

                fig1, ax1 = plt.subplots(figsize=(5,3))
                ax1.pie(sizes, labels=aspect_labels, autopct='%1.1f%%',
                        shadow=False, startangle=90)
                ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

                st.pyplot(fig1)
                # st.write(response)
        
                def convert_df(df):
                # IMPORTANT: Cache the conversion to prevent computation on every rerun
                    return df.to_csv().encode('utf-8')

                csv = convert_df(df)

            
                download_csv = st.download_button(
                                    label="Download Insight as CSV",
                                    data=csv,
                                    file_name='aspect-analysis-review.csv',
                                    mime='text/csv',
                                    use_container_width=True
                                    )
            else:
                st.write('error connection.. please try again later')
        else:
            st.write(f'we detect your language is: {language}, we still develope to support your need soon.. sorry for inconvenience.')

# author information
st.divider()
st.markdown('## :computer: Author Information')
st.markdown('This project is built by [Farrel Arrizal](https://www.linkedin.com/in/farrel-arrizal/)')



# with st.form(key="file_upload_form"):
#     st.markdown('#### Want to analyze with large reviews? sure we can do it!')
#     # st.file_uploader('upload', label_visibility="hidden", help="Only file with format .txt is received!", type=['txt'])
#     uploaded_file = st.file_uploader("Choose a TXT file", accept_multiple_files=True, type=['txt'])
#     for file in uploaded_file:
#         bytes_data = file.read()
#         st.write("filename:", file.name)
#         # st.write(bytes_data)
#     files_language = st.radio('Select Review Language from File:', ('auto-detect', 'english', 'indonesia')) 
#     file_predict = st.form_submit_button('predict')
    
#     if file_predict:
#         for _ in stqdm(range(10)):
#             time.sleep(0.5)
