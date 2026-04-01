# Core settings for AuraDocs

# Embedding
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval
TOP_K = 5

# LLM
LLM_MODEL = "gemini-3-flash-preview"
MAX_CHAT_HISTORY = 6  # Messages to include in context

# File Processing
MAX_FILE_SIZE_MB = 10
SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".csv", ".xlsx", ".xls"]
