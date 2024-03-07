from fastapi import FastAPI,File,UploadFile,Request,Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import numpy as np
from io import BytesIO
from serpapi import GoogleSearch
import json
from PIL import Image
import tensorflow as tf

app=FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates= Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse("HomePage.html", {"request": request})


MODEL=tf.keras.models.load_model("../saved_model/2")

class_names=["Early Blight","Late Blight","Healthy"]

def read_file_as_image(data)->np.ndarray:
    image=np.array(Image.open(BytesIO(data)))
    return image

p_class=None

@app.post("/predict/")
async def predict(file:UploadFile=File(...)):

    image= read_file_as_image(await file.read())
    image_batch=np.expand_dims(image,0)
    predictions=MODEL.predict(image_batch)
    predicted_class=class_names[np.argmax(predictions[0])]
    confidence_level=np.max(predictions[0])
    return {'class':predicted_class,
            "confidence":float(confidence_level)}

# serpapi code
def relevant_result(predicted_class):
    if predicted_class=='healthy':
        return 'No remidies for healthy leaf'
    
    params = {
  'engine': "youtube",
  'search_query': f"treatment for {predicted_class} in potato leaf",
  'api_key': "dc926e2ee48ec0ad9c2a46794535447893ee9fa2f55d5589fd9e857467efea8d"
}

    search = GoogleSearch(params)
    results = search.get_dict()
    video_results=results["video_results"]
    sort_result=[]
    for data in video_results:
        if data['views']>10000 and( f"{predicted_class}" in data['description'].lower() or "potato blight" in data['description'].lower() and 'tomato' not in data['title'].lower()): 
            sort_result.append(data["link"])

    sort_result
    return sort_result


@app.post("/fetch_videos/")
async def fetch_videos(predicted_class:str=Depends(predict)):
    # Call the relevant_result function and return video data
    # predicted_class = p_class
    video_data = relevant_result(predicted_class)
    # print(video_data)
    return {"videos": video_data}
    
 
if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)