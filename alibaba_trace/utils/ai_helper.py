import os
import logging
from dotenv import load_dotenv

root_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')
load_dotenv(dotenv_path=root_env_path)

logger = logging.getLogger("AIHelper")

def get_gpt_explanation(top_features, reason, scores):
    """
    Returns an RCA string using GenAIRCAEngine using the given anomaly characteristics.
    """
    try:
        from alibaba_trace.incident.rca_genai import GenAIRCAEngine
        key = os.getenv('OPENAI_API_KEY')
        if not key:
            return "No OpenAI key configured."
            
        model = os.getenv('OPENAI_MODEL', 'gpt-4o')
        engine = GenAIRCAEngine(openai_api_key=key, openai_model=model)
        
        # Build dummy dicts for the engine
        feature_errors = {feat: float(scores.get('top_score', 0.0)) for feat in top_features}
        import pandas as pd
        
        response = engine.run_rca(
            incident_type=reason,
            raw_data=pd.DataFrame(),
            feature_error_map=feature_errors,
            container_context="k8s_live_stream"
        )
        return response
    except Exception as e:
        logger.error(f"Failed to get GPT Explanation: {e}")
        return f"GenAI generation failed: {e}"

def get_configured_engine(temperature=0.10):
    from alibaba_trace.incident.rca_genai import GenAIRCAEngine
    key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    return GenAIRCAEngine(
        openai_api_key=key, 
        openai_model=model,
        temperature=temperature
    )
