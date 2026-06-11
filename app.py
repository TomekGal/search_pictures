import streamlit as st
from openai import OpenAI
from dotenv import dotenv_values
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import base64
from pathlib import Path
from PIL import Image
from tkinter.filedialog import askdirectory, askopenfilename
from tkinter import Tk
import shutil
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
        url=env["QDRANT_URL"],
        api_key=env["QDRANT_API_KEY"],
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

def prepare_image_for_openai(file_name):
    with open(file_name, "rb") as f:
        image_data=base64.b64encode(f.read()).decode('utf-8')
        
    return image_data    
    
@st.cache_data
def get_image_description(file_name):
        base64_image=prepare_image_for_openai(file_name)
        openai_client=get_openai_client()
        response=openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                {
                    "type": "text",
                    "text": """Podaj bardzo szczegółowy,
                               wyczerpujący opis tego obrazu.
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

files_lib=[]
if "file_name" not in st.session_state:
    st.session_state["file_name"] = []
if "input_path" not in st.session_state:
    st.session_state["input_path"] = []

if st.button("Wybierz folder ze zdjęciami"):
       root = Tk()
       root.withdraw()
       root.attributes('-topmost', True)
       input_path=askdirectory(title='Wybierz folder', parent=root)
       
       DATA_PATH=Path(input_path)
       
       st.session_state["input_path"] = input_path
       status = st.empty()
       liczba_plikow = sum(1
        for p in DATA_PATH.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS)
       st.subheader(f'W folderze znajduje się {liczba_plikow} plików')
       licznik=1
       for file in DATA_PATH.iterdir():
               
                if file.suffix.lower() in EXTENSIONS:
                    status.info(f"Przetwarzanie pliku: {file.name}. Plik {licznik} z {liczba_plikow}")
                    files_lib.append(
                    {"name": str(file),
                    "description": get_image_description(file)
                    })
                    licznik+=1
                              
       if len(files_lib)==liczba_plikow:
          status.success("Wszystkie pliki zostały przetworzone ✅") 
        
# Add pictures to QDerant on server       
qdrant_client=get_qdrant_client()  
info = qdrant_client.get_collection(QDRANT_COLLECTION_NAME)            
for idx, file in enumerate(files_lib):       
                    qdrant_client.upsert(
                    collection_name=QDRANT_COLLECTION_NAME,
                    points=[
                        PointStruct(
                            id=info.points_count+idx,
                            vector=get_embeddings(f'{file["name"]} {file["description"]}'),
                            payload=file
                        )])
                          
## Get sentence from user
st.session_state["input_sentence"]=st.text_input("Wpisz czego szukasz")
         
if st.button("Szukaj"):
 @st.fragment()
 def search_picture():
            sentence = st.session_state["input_sentence"]
            result=qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=get_embeddings(sentence),
            limit=3,
            )
           
# show results of searching 
                        
            for point in result:
                    score = point.score        # Pobranie wartości score (float)
                    payload = point.payload    # Pobranie powiązanych metadanych (słownik)
                    point_id = point.id        # Pobranie ID punktu


                    if score>0.5:
                        st.write(f"**Procent dopasowania:** {score*100:.2f}% | **Nazwa zdjęcia:** {payload['name']}")
                        st.image(payload["name"])
                        st.write("Dodaję:", payload["name"])
                        st.session_state["file_name"].append(payload["name"])
                        

            if len(st.session_state["file_name"])>0:          
                st.subheader(f"W wynikacha wyszukiwania znajduje się: {len(st.session_state['file_name'])} plików")  
                
            else:
                    st.write("Niestety nie znaleziono pasujących zdjęć. Zmień sentencję do wyszukiwania")

            if len(st.session_state["file_name"])>0:   
                if st.button("Zapisz pliki"):
                        st.subheader(len(st.session_state["file_name"]))
                        folder=Path(st.session_state["file_name"][0])
                        subfolder=folder.parent / "Wyszukane obrazy"
                        subfolder.mkdir(parents=True, exist_ok=True)             
                        for key, value in enumerate(st.session_state["file_name"]):
                            src=Path(value)
                            shutil.copy2(src,subfolder)
                            st.write(f"Zapisany plik: {src}")

            st.session_state["file_name"]=[]    
 search_picture()                           

st.subheader("Jeżeli chcesz usunąć kolekcję z bazy danych QDrant")    
if st.button("Clear Qdrant collections"):
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