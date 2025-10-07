"""Model alias resolution and promotion logic for Bank Marketing project."""
import mlflow
from mlflow.client import MlflowClient
from loguru import logger
from ARISA_DSML.config import MODEL_NAME


def get_model_by_alias(client: MlflowClient, model_name: str = MODEL_NAME, alias: str = "champion"):
    """
    Retrieve model version by alias (champion/challenger).
    Returns None if alias not found.
    """
    try:
        return client.get_model_version_by_alias(model_name, alias)
    except Exception as e:
        if f"alias {alias} not found" in str(e).lower():
            logger.info(f"No alias '{alias}' found for model {model_name}.")
            return None
        logger.error(f"Error retrieving alias '{alias}': {e}")
        raise


def promote_newest_to_champion(client: MlflowClient, model_name: str = MODEL_NAME):
    """Promote the latest model version to champion if none exist."""
    latest = client.get_latest_versions(model_name)
    if not latest:
        logger.warning(f"No versions found for model {model_name}. Cannot promote.")
        return None
    newest = latest[0]
    client.set_registered_model_alias(model_name, "champion", newest.version)
    logger.info(f"Promoted newest model (v{newest.version}) to champion.")
    return newest


def compare_and_promote(client: MlflowClient, champ_mv, chall_mv, model_name: str = MODEL_NAME):
    """Compare f1 metrics and promote challenger if better."""
    champ_run = client.get_run(champ_mv.run_id)
    chall_run = client.get_run(chall_mv.run_id)

    f1_champ = champ_run.data.metrics.get("f1_cv_mean", 0)
    f1_chall = chall_run.data.metrics.get("f1_cv_mean", 0)

    logger.info(f"Champion f1_cv_mean={f1_champ}, Challenger f1_cv_mean={f1_chall}")

    if f1_chall >= f1_champ:
        logger.info("✅ Challenger outperforms Champion — promoting to Champion.")
        client.delete_registered_model_alias(model_name, "challenger")
        client.set_registered_model_alias(model_name, "champion", chall_mv.version)
    else:
        msg = "🚫 Challenger does not surpass current Champion. Keeping existing model."
        logger.warning(msg)
        raise Exception(msg)


if __name__ == "__main__":
    client = MlflowClient(mlflow.get_tracking_uri())

    champ_mv = get_model_by_alias(client, alias="champion")
    chall_mv = get_model_by_alias(client, alias="challenger")

    if champ_mv is None and chall_mv is None:
        promote_newest_to_champion(client)
    elif champ_mv is None and chall_mv:
        logger.info("Found Challenger without Champion — promoting Challenger.")
        client.set_registered_model_alias(MODEL_NAME, "champion", chall_mv.version)
    elif champ_mv and chall_mv:
        compare_and_promote(client, champ_mv, chall_mv, MODEL_NAME)
    elif champ_mv and chall_mv is None:
        logger.info("Champion exists, no Challenger found — nothing to promote.")
