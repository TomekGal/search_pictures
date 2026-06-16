import streamlit as st
from openai import OpenAI
from dotenv import dotenv_values
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import base64
from pathlib import Path
import os
from qdrant_client.http import models


env=dotenv_values(".env")
EMBEDDING_DIM=3072
EMBEDING_MODEL="text-embedding-3-large"
QDRANT_COLLECTION_NAME="images_descriptions"
input_path=""
EXTENSIONS={".jpg",".jpeg",".png",".gif"}

## OpenAi get Api_Key

@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=st.session_state["openai_api_key"])

def get_qdrant_client():
    return QdrantClient(
        url=st.secrets["QDRANT_URL"],
        api_key=st.secrets["QDRANT_API_KEY"],
        check_compatibility=False
    )

@st.cache_data
def get_embeddings(text):
    if text:
        openai_client=get_openai_client()
        response=openai_client.embeddings.create(
            input=[text],
            model=EMBEDING_MODEL,
            dimensions=EMBEDDING_DIM,
        )
        return response.data[0].embedding
    
#function for preparing images for OpenAI

def prepare_image_for_openai(uploaded_file):
        return base64.b64encode(
        uploaded_file.getvalue()).decode("utf-8")

def image_to_base64(uploaded_file):
      with open(uploaded_file, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
     
def get_image_description(uploaded_file):
        base64_image=prepare_image_for_openai(uploaded_file)
        openai_client=get_openai_client()
        response=openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                {
                    "type": "text",
                    "text": """Podaj bardzo szczegółowy,
                               wyczerpujący opis tego obrazu w języku polskim.
                               Skoncentruj się na obiektach, kolorach, akcjach, tekście i ogólnym kontekście sceny.
                               Ten opis zostanie wykorzystany do wygenerowania osadzania wyszukiwania semantycznego.""",
                },
                {
                    "type": "image_url",
                    "image_url": { 
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail":"high"
                    }
                }
                ],
            }

            ],
        )
        return response.choices[0].message.content

# MAIN 
# 
# OpenAI API key protection
if not st.session_state.get("openai_api_key"):
    if "OPENAI_API_KEY" in env:
        st.session_state["openai_api_key"] = env["OPENAI_API_KEY"]

    else:
        st.info("Dodaj swój klucz API OpenAI aby móc korzystać z tej aplikacji")
        st.session_state["openai_api_key"] = st.text_input("Klucz API", type="password")
        if st.session_state["openai_api_key"]:
            st.rerun()

if not st.session_state.get("openai_api_key"):
    st.stop()

# Create QDrant collection
qdrant_client=get_qdrant_client()
if not qdrant_client.collection_exists(collection_name=QDRANT_COLLECTION_NAME):
    qdrant_client.create_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
    )
    
st.title("Aplikacja do przeszukiwania zdjęć")
st.subheader("W tej aplikacji na podstawie wpisanej sentencji możesz wyszukać pasujące zdjęcia")

if "files_lib" not in st.session_state:
    st.session_state.files_lib=[]

if "image_path" not in st.session_state:
    st.session_state["image_path"] = []

# if "uploaded_to_qdrant" not in st.session_state:
#     st.session_state.uploaded_to_qdrant = False

uploaded_files = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"], accept_multiple_files=True)     
st.session_state["input_path"] = input_path

if uploaded_files:
    st.session_state.files_lib = []
    for uploaded_file in uploaded_files:
            #    os.makedirs("images", exist_ok=True)
            #    file_path=os.path.join("images", uploaded_file.name)
            #    with open(file_path,"wb") as f:
            #          f.write(uploaded_file.getbuffer())
               st.session_state.files_lib.append(
                {"name": uploaded_file.name,
                "description": get_image_description(uploaded_file),
                "image_base64": image_to_base64(uploaded_file)
                })
         
# st.session_state.uploaded_to_qdrant = True       
# Add pictures to QDerant on server       
info = qdrant_client.get_collection(QDRANT_COLLECTION_NAME)      
for idx, file in enumerate(st.session_state.files_lib):
                          
                    qdrant_client.upsert(
                    collection_name=QDRANT_COLLECTION_NAME,
                    points=[
                        PointStruct(
                            id=info.points_count+idx,
                            vector=get_embeddings(f'{file["name"]} {file["description"]}'),
                            payload=file
                        )])
          
# ## Get sentence from user
st.session_state["input_sentence"]=st.text_input("Wpisz czego szukasz")
       
if st.button("Szukaj"):
            
            sentence = st.session_state["input_sentence"]
            result=qdrant_client.query_points(
            collection_name=QDRANT_COLLECTION_NAME,
            query=get_embeddings(sentence),
            limit=3,
           # with_payload=True
            )
           
# show results of searching 
                        
            for point in result.points:
                    score = point.score        # Pobranie wartości score (float)
                    payload = point.payload    # Pobranie powiązanych metadanych (słownik)
                    point_id = point.id        # Pobranie ID punktu


                    if score>0.5:
                        st.write(f"**Procent dopasowania:** {score*100:.2f}% | **Nazwa zdjęcia:** {payload['name']}")
                        img_bytes = base64.b64decode(payload["image_base64"])
                        st.image(img_bytes)
                        st.session_state["image_path"].append(payload["image_base64"])
                        

            if len(st.session_state["image_path"])>0:          
                st.subheader(f"W wynikach wyszukiwania znajduje się: {len(st.session_state["image_path"])} plików")  
                
            else:
                    st.write("Niestety nie znaleziono pasujących zdjęć. Zmień sentencję do wyszukiwania")

#if len(st.session_state["image_path"])>0:   
if st.button("Przygotuj wyszukane pliki do pobrania"):
            
            for value in st.session_state["image_path"]:
                source = Path(value)

                if source.exists():
                   with open(source,"rb") as f:
                    st.download_button(
                         label=f"Pobierz{source.name}",
                         data=f,
                         file_name=source.name
                    )
                   
                            
            qdrant_client=get_qdrant_client() 
            qdrant_client.delete(
                    collection_name=QDRANT_COLLECTION_NAME,
                    points_selector=models.FilterSelector(
                        filter=models.Filter()
                    )
                )
         


info = qdrant_client.get_collection(QDRANT_COLLECTION_NAME)
if info.points_count == 0:
    st.toast("Kolekcja jest pusta", duration='long', icon='📙')
else:
    st.toast(f"Kolekcja zawiera {info.points_count} pozycji", duration='long', icon='📚')     