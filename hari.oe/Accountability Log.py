import datetime

def deploy_model(model_id, model_version, owner_email):
    """Logs the owner of a deployed AI model."""
    metadata = {
        "model_id": model_id"version": model_version"deployed_by": owner_email"deployment_time": datetime.datetime.now().isoformat()"status": "Production"
    }
    
    print(f"Accountability Log: {metadata}")
    return metadata
deploy_model("loan_classifier_v4""1.2.0""data_science_team_leads@company.com")
