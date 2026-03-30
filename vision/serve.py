from fastapi import FastAPI, Request

app = FastAPI(title="Vision Mock Service")


@app.post("/detect")
async def detect(_: Request):
	return {
		"labels": ["plastic_bottle", "cardboard"],
		"confidence": [0.98, 0.85],
	}
