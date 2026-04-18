from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import io
from PIL import Image
from upscale_logic import get_upscaler
import time

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/upscale")
async def upscale_image(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        start_time = time.time()
        
        # Get upscaler instance
        upscaler = get_upscaler()
        
        # Perform upscale
        upscaled_image = upscaler.upscale(image)
        
        process_time = time.time() - start_time
        print(f"Upscaling completed in {process_time:.2f}s")
        
        # Save to buffer
        img_byte_arr = io.BytesIO()
        upscaled_image.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr = img_byte_arr.getvalue()
        
        return Response(content=img_byte_arr, media_type="image/jpeg", headers={
            "X-Process-Time": str(process_time)
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
