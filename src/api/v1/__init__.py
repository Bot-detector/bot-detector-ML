from fastapi import APIRouter
from . import predict

router = APIRouter()
router.include_router(predict.router)
