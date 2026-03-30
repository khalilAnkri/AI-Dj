from google.cloud import aiplatform

# 1. Initialize
aiplatform.init(
    project="ai-dj-487610",
    location="europe-west4"
)

# 2. Define the Job
job = aiplatform.PipelineJob(
    display_name="ai-dj-manual-run",
    template_path="ai-dj-training-pipeline.json", 
    pipeline_root="gs://ai-dj-487610-bucket/pipeline_root",
    enable_caching=True 
)

# 3. Submit
job.submit()
print("Pipeline submitted! View it here: https://console.cloud.google.com/vertex-ai/pipelines")