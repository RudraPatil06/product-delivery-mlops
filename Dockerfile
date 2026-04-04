FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Fix Python path
ENV PYTHONPATH=/app

EXPOSE 8000
EXPOSE 8501

CMD sh -c "uvicorn app:app --host 0.0.0.0 --port 8000 & streamlit run ui/dashboard.py --server.address 0.0.0.0 --server.port 8501"