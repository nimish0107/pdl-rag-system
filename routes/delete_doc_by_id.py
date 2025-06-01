from fastapi import APIRouter, HTTPException
from services import delete_doc_by_id

delete_document_router = APIRouter()

@delete_document_router.delete("/delete_document/{doc_id}")
async def delete_document_endpoint(doc_id: str):
    """
    Delete a document and its chunks by document ID.
    """
    try:
        deleted = await delete_doc_by_id(doc_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"No document found with ID: {doc_id}")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {e}")

    return {
        "status": "success",
        "message": f"Document with ID '{doc_id}' deleted successfully."
    }