from fastapi import APIRouter, HTTPException
from services import delete_all_docs

delete_all_docs_router = APIRouter()

@delete_all_docs_router.delete("/delete_all_documents")
async def delete_all_documents_endpoint():
    """
    Delete all documents from FAISS indexes and remove related files.
    """
    try:
        deleted = await delete_all_docs()
        if not deleted:
            raise HTTPException(status_code=404, detail="No documents found to delete.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting all documents: {e}")

    return {
        "status": "success",
        "message": "All documents deleted successfully."
    }