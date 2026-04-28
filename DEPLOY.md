# Deploying ISL.live

This project is a FastAPI web app. The browser handles the camera and MediaPipe hand landmarks, then the server runs TensorFlow inference over WebSocket.

## Recommended: Railway

1. Push the `2HMPTLISL` folder to GitHub.
2. Create a new Railway project from that GitHub repo.
3. Railway should detect the `Dockerfile`.
4. Deploy.
5. Open the generated `https://...railway.app` URL.

The camera works only on `https://` or `localhost`, so use the HTTPS cloud URL.

## Render

1. Push the `2HMPTLISL` folder to GitHub.
2. Create a new Render Web Service.
3. Choose Docker as the runtime.
4. Set the service port to `8000` if Render asks.
5. Deploy and open the HTTPS URL.

## Local test before cloud

```bash
pip install -r requirements.txt
uvicorn backend:app --host 0.0.0.0 --port 8000
```

Then open:

```text
http://localhost:8000
```

## Important files that must be deployed

- `backend.py`
- `requirements.txt`
- `static/index.html`
- `label_classes.json`
- `isl_mediapipe.weights.h5`

## Notes

- WebSockets are required. Use a host that supports WebSockets.
- Free hosting may be slow because TensorFlow is heavy.
- If the app crashes during install due to memory limits, use a paid/small instance with at least 1 GB RAM.
