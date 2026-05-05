from kfp.dsl import Input, Model, component

from src.pipelines.config import BASE_IMAGE


@component(
    base_image=BASE_IMAGE,
    packages_to_install=[
        "google-cloud-aiplatform",
    ]
)
def deploy_model(
    registered_model: Input[Model],
    project_id: str,
    location: str,
    endpoint_display_name: str,
    machine_type: str = "n1-standard-2",
    min_replicas: int = 1,
    max_replicas: int = 2,
):

    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=location)

    # 1. Retrieve the registered model using the resource name passed from register_model
    resource_name = registered_model.metadata.get("resource_name")
    if not resource_name:
        raise ValueError(
            "No 'resource_name' found in registered_model metadata. "
            "Make sure register_model ran successfully before this step."
        )

    print(f"Fetching model from registry: {resource_name}")
    vertex_model = aiplatform.Model(model_name=resource_name)

    # 2. Find or create the endpoint
    print(f"Looking for existing endpoint '{endpoint_display_name}'...")
    existing_endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{endpoint_display_name}"',
        project=project_id,
        location=location,
    )

    if existing_endpoints:
        endpoint = existing_endpoints[0]
        print(f"Reusing existing endpoint: {endpoint.resource_name}")
    else:
        print(f"No existing endpoint found. Creating '{endpoint_display_name}'...")
        endpoint = aiplatform.Endpoint.create(
            display_name=endpoint_display_name,
            project=project_id,
            location=location,
        )
        print(f"Endpoint created: {endpoint.resource_name}")

    # 3. Undeploy all previously deployed models to avoid stale versions
    for deployed_model in endpoint.list_models():
        print(f"Undeploying old model: {deployed_model.id}")
        endpoint.undeploy(deployed_model_id=deployed_model.id)

    # 4. Deploy the new model version
    print(f"Deploying model to endpoint. This takes ~10-15 minutes...")
    vertex_model.deploy(
        endpoint=endpoint,
        deployed_model_display_name="spotify-hit-predictor-live",
        machine_type=machine_type,
        min_replica_count=min_replicas,
        max_replica_count=max_replicas,
        traffic_percentage=100,
    )

    print("-" * 40)
    print("SUCCESS — model is live!")
    print(f"Endpoint resource name : {endpoint.resource_name}")
    print(f"Endpoint ID            : {endpoint.name}")
    print(
        f"Test it: gcloud ai endpoints predict {endpoint.name} "
        f"--region={location} --json-request=request.json"
    )
    print("-" * 40)