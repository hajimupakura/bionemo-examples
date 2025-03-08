import requests

def predict_structure(sequence):
    url = "https://www.ebi.ac.uk/Tools/services/rest/alphafold/run"
    data = {
        "email": "alhajjm@yahoo.com",  # Required by EBI
        "sequence": sequence,
    }
    response = requests.post(url, data=data)
    if response.status_code == 200:
        job_id = response.text
        return job_id
    else:
        raise Exception("Failed to submit job")

def get_results(job_id):
    url = f"https://www.ebi.ac.uk/Tools/services/rest/alphafold/result/{job_id}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        raise Exception("Failed to retrieve results")

# Example usage
sequence = "MKTIIALSYIFCLVFA"
job_id = predict_structure(sequence)
results = get_results(job_id)
print(results)