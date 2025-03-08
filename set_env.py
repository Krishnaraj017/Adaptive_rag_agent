import os


def set_env_variables():
    """Sets required environment variables."""
    os.environ["GROQ_API_KEY"] = "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"
    os.environ["TAVILY_API_KEY"] = "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"
    
    print("Environment variables set successfully!")