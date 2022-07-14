# TAP_assesment


# Quick overview
Submission for TAP programme

# How to Run 

### Prerequisites 
- Windows10 (OS used)
- Nvidia with GPU enabled. ([How-to](https://towardsdatascience.com/installing-tensorflow-with-cuda-cudnn-and-gpu-support-on-windows-10-60693e46e781))

Download data [here](https://drive.google.com/drive/folders/18IX5ywPLuNWkNnwosmPEHop672eFEvbq?usp=sharing).

Download models [here](https://drive.google.com/drive/folders/1fHSTbDKZq7RckDvOqEQ8uccUPDyAi8S8?usp=sharing).

# Repository Directiory
```
  ├── data            <-- Where all data is stored
  |   ├── error 
  |   └── train 
  │        
  ├── docker          <-- Contains Dockerfile for deployment
  |   └── Dockerfile
  |
  ├── images          <-- Visualisations stored
  |   
  ├── models          <-- Saved models
  |   
  ├── notebooks       <-- Jupyter notebooks
  |   ├── eda.ipynb   <-- Exploratory Data Analysis 
  |   └── boat_classification.ipynb <-- Main Jupyter notebook presentation
  |   
  ├── src
  |   ├── model_training.py    <-- Python Functions for model building
  |   └── pre_process_data.py  <-- Python Functions for pre-processing data
  |
  ├── .gitignore  <-- Removes unwanted files from being pushed to github
  |   
  ├── app.py      <-- FastAPI python implementation of model
  |
  ├── model_config.py <-- Model Configurations (Classes, class weights, early stopper, optimiser, learning rate scheduler)
  | 
  └── requirements.txt <-- required libraries for repository
```

# Quick-Start
```bash
  git clone https://github.com/Cawinchan/TAP_assesment.git
  cd TAP_assement 
  code . 
```

Note: boat_classification.ipynb has dependencies from src.model_training.py and src.pre_process_data.py

# How to run FastAPI locally
```bash
  uvicorn app:app --reload
```

# How to run build Docker Image and run the container
```bash
  docker build -t fastapiapp:latest -f docker/Dockerfile .
  docker run -p 80:80 fastapiapp:latest
```

