import json
import boto3


def get_secret(secret_name: str, region: str = "eu-north-1b") -> dict:
    client = boto3.client(
        "secretsmanager",
        region_name=region,
    )

    response = client.get_secret_value(
        SecretId=secret_name,
    )

    return json.loads(response["SecretString"])