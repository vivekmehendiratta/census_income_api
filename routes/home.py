from fastapi import APIRouter
app_home = APIRouter()


@app_home.get('/', tags=["Welcome to Census Income Prediction Page"])
async def hello():
    return {"message": "Hello!"}


# @app_home.get('/bye', tags=["Intro"])
# async def bye():
#     return {"message": "Bye!"}