from langchain_core.language_models import BaseLanguageModel

def get_llm(config: dict) -> BaseLanguageModel:
    """
    Initialize and return the LLM (Ollama, ChatGPT, Groq, Grok, DeepSeek) based on config.yaml.
    """
    llm_cfg = config.get("llm", {})
    provider = llm_cfg.get("provider", "").strip().lower()
    model = llm_cfg.get("model", "").strip()
    temp = llm_cfg.get("temperature", 1.0)  # Default temperature

    if not provider or not model:
        raise ValueError("❌ LLM provider or model not specified in config['llm'].")

    # ✅ Ollama
    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError("🦙 Install Ollama support with: pip install langchain-ollama")
        print(f"🦙 Using Ollama with model: '{model}', temperature: {temp}")
        return ChatOllama(model=model, temperature=temp)

    # ✅ OpenAI ChatGPT
    elif provider in ("chatgpt", "openai"):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError("🤖 Install OpenAI support with: pip install langchain-openai")
        api_key = llm_cfg.get("api_key", "").strip()
        if not api_key:
            raise ValueError("❌ Missing OpenAI API key in config['llm']['api_key'].")
        print(f"🤖 Using OpenAI ChatGPT with model: '{model}', temperature: {temp}")
        return ChatOpenAI(model=model, api_key=api_key, temperature=temp)

    # ✅ Groq LLM
    elif provider == "groq":
        try:
            from langchain_groq.chat_models import ChatGroq
        except ImportError:
            raise ImportError("⚡ Install Groq support with: pip install langchain-groq")
        api_key = llm_cfg.get("api_key", "").strip()
        if not api_key:
            raise ValueError("❌ Missing Groq API key in config['llm']['api_key'].")
        print(f"⚡ Using Groq LLM with model: '{model}', temperature: {temp}")
        return ChatGroq(model=model, api_key=api_key, temperature=temp)

    # ✅ DeepSeek LLM
    elif provider == "deepseek":
        try:
            from langchain_deepseek import ChatDeepSeek
        except ImportError:
            raise ImportError("🧠 Install DeepSeek support with: pip install langchain-deepseek")
        api_key = llm_cfg.get("api_key", "").strip()
        if not api_key:
            raise ValueError("❌ Missing DeepSeek API key in config['llm']['api_key'].")
        print(f"🧠 Using DeepSeek LLM with model: '{model}', temperature: {temp}")
        return ChatDeepSeek(model=model, api_key=api_key, temperature=temp)

    # ❌ Unsupported provider
    else:
        raise ValueError(f"❌ Unsupported LLM provider: '{provider}'. Supported: ollama, openai, groq, grok, deepseek")
