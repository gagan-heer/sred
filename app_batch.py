import os
from flask import Flask, request, jsonify
from openai import OpenAI, AsyncOpenAI
import time
import json
from celery.exceptions import CeleryError
from celery.result import AsyncResult
from dotenv import load_dotenv
from celery_config import make_celery
import numpy as np
import asyncio

load_dotenv()

app = Flask(__name__)

# Load OpenAI API keys from environment variables
api_keys = os.getenv('OPENAI_API_KEYS').split(',')

# Configure Redis URL
redis_url = os.getenv('REDIS_URL', 'redis://127.0.0.1:6379/0')

# Configure Celery
app.config.update(
    CELERY_BROKER_URL=redis_url,
    CELERY_RESULT_BACKEND=redis_url,
)
celery = make_celery(app)

BATCH_SIZE = 50000  # Number of requests to batch together
POLL_INTERVAL = 10  # Time in seconds to wait between polling
MAX_RETRY_ATTEMPTS = 1  # Maximum number of retry attempts for failed requests
MAX_API_KEY_RETRY = len(api_keys)  # Maximum retry attempts for different API keys
EMBEDDING_ENGINE = "text-embedding-3-small"

@celery.task(name='app_batch.process_batch')
def process_batch(batch_data, api_key_index=0, retry_count=0):
    current_api_key = api_keys[api_key_index]
    client = OpenAI(api_key=current_api_key)
    print("Using API Key: ", current_api_key)
    print("Task started with batch_data: ", batch_data)

    try:
        # Get the Celery task ID
        task_id = process_batch.request.id
        batch_input_file_id = upload_batch_file(client, batch_data, task_id)
        print("Batch input file ID: ", batch_input_file_id)
        batch_id = create_batch(client, batch_input_file_id)
        print("Batch ID: ", batch_id)

        # Poll for batch status
        while True:
            status_response = get_batch_status(client, batch_id)
            print("status_response: ", status_response)
            status = status_response.status
            if status == 'completed':
                break
            elif status in ['failed', 'expired']:
                error_code = status_response.errors.data[0].code if status_response.errors else 'unknown_error'
                if error_code == 'token_limit_exceeded' and api_key_index + 1 < MAX_API_KEY_RETRY:
                    print("Rate limit reached. Retrying with next API key.")
                    return process_batch(batch_data, api_key_index=api_key_index+1, retry_count=retry_count)
                process_batch.update_state(state='FAILURE', meta={'exc_type': 'CeleryError', 'exc_message': f'Batch processing failed with status: {status}'})
                raise CeleryError(f'Batch processing failed with status: {status}')
            time.sleep(POLL_INTERVAL)

        results, errors = get_batch_results(client, batch_id)
        print("Results: ", results)
        print("Errors: ", errors)

        # Retry failed requests
        if errors and retry_count < MAX_RETRY_ATTEMPTS:
            failed_requests = [item['custom_id'] for item in errors if item.get('response', {}).get('status_code') == 400]
            if failed_requests:
                print("Retrying failed requests: ", failed_requests)
                failed_data = [item for item in batch_data if item['custom_id'] in failed_requests]
                retry_results = process_batch(failed_data, api_key_index=api_key_index, retry_count=retry_count+1)
                results.extend(retry_results.get('results', []))

        return {'results': results}
    except Exception as e:
        print("Error in process_batch: ", str(e))
        raise e

def upload_batch_file(client, batch_data, task_id):
    print("Uploading batch file with data: ", batch_data)
    try:
        # Create a .jsonl file for the batch requests
        file_name = f'batch_input_{task_id}.jsonl'  # <- Modified this line
        with open(file_name, 'w') as f:
            for item in batch_data:
                f.write(json.dumps(item) + '\n')
        print(f"Batch file content written to {file_name}")
        
        # Upload the batch file to OpenAI
        with open(file_name, "rb") as f:
            batch_input_file = client.files.create(
                file=f,
                purpose="batch"
            )
        print("Batch file uploaded, file ID: ", batch_input_file.id)
        return batch_input_file.id
    except Exception as e:
        print("Error in upload_batch_file: ", str(e))
        raise e

def create_batch(client, batch_input_file_id):
    print("Creating batch with input file ID: ", batch_input_file_id)
    try:
        # Create the batch using the uploaded file ID
        batch = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": "batch processing job"}
        )
        print("Batch created with ID: ", batch.id)
        return batch.id
    except Exception as e:
        print("Error in create_batch: ", str(e))
        raise e

def get_batch_status(client, batch_id):
    print("Retrieving batch status for batch ID: ", batch_id)
    try:
        # Retrieve the status of the batch
        return client.batches.retrieve(batch_id)
    except Exception as e:
        print("Error in get_batch_status: ", str(e))
        raise e

def get_batch_results(client, batch_id):
    print("Retrieving batch results for batch ID: ", batch_id)
    try:
        # Retrieve the results of the batch
        batch = client.batches.retrieve(batch_id)
        output_file_id = batch.output_file_id
        error_file_id = batch.error_file_id

        results = []
        errors = []

        if output_file_id:
            output_response = client.files.content(output_file_id)
            output_content = output_response.read().decode('utf-8')
            results = [json.loads(line) for line in output_content.splitlines()]

        if error_file_id:
            error_response = client.files.content(error_file_id)
            error_content = error_response.read().decode('utf-8')
            errors = [json.loads(line) for line in error_content.splitlines()]

        return results, errors
    except Exception as e:
        print("Error in get_batch_results: ", str(e))
        raise e

 @app.route('/genai', methods=['POST'])
def genai():
    data = request.get_json()
    batch = data.get('batch', [])
    print("batch: ", batch)
    if not batch:
        return jsonify({'error': 'Batch data is required'}), 400
    
    # Check if batch size exceeds the limit
    if len(batch) > BATCH_SIZE:
        return jsonify({'error': f'Batch size exceeds the limit of {BATCH_SIZE}'}), 400

    try:
        task = process_batch.apply_async(args=[batch])
        print("Task ID: ", task.id)
        return jsonify({'job_id': task.id}), 202
    except Exception as e:
        print("Error: ", e)
        return jsonify({'error': str(e)}), 500

@app.route('/bstatus/<job_id>', methods=['GET'])
def job_status(job_id):
    task = AsyncResult(job_id, app=celery)
    
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'status': task.info.get('status', ''),
            'result': task.info.get('results', [])
        }
    else:
        response = {
            'state': task.state,
            'status': str(task.info)  # This is the exception raised
        }
    return jsonify(response)

async def get_text_embedding(client, text, model=EMBEDDING_ENGINE):
    text = text.replace("\n", " ")
    response = await client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

async def calculate_similarity(client, text1, text2, model=EMBEDDING_ENGINE):
    embedding1 = await get_text_embedding(client, text1, model=model)
    embedding2 = await get_text_embedding(client, text2, model=model)
    similarity = cosine_similarity(embedding1, embedding2)
    return similarity

@app.route('/find_best_match', methods=['POST'])
async def find_best_match():
    data = request.get_json()
    entry = data.get('entry')
    matches = data.get('matches')

    if not entry or not matches:
        return jsonify({"error": "Invalid input"}), 400

    current_api_key = api_keys[0]
    client = AsyncOpenAI(api_key=current_api_key)

    max_similarity = -1
    best_match_id = None

    tasks = [
        calculate_similarity(client, entry, match)
        for match in matches
    ]

    similarities = await asyncio.gather(*tasks)

    for idx, similarity in enumerate(similarities):
        if similarity > max_similarity:
            max_similarity = similarity
            best_match_id = idx + 1

    print(f"For entry: {entry}, best match is: {best_match_id} with similarity score: {max_similarity}")
    return jsonify({"best_match_id": best_match_id, "similarity_score": max_similarity})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
