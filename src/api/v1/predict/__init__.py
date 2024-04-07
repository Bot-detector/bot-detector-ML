from fastapi import APIRouter

router = APIRouter()


@router.get("/predict")
async def get_prediction():
    pass
