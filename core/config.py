# Core settings for AuraDocs

# Embedding
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval
TOP_K = 5
SIMILARITY_THRESHOLD = 0.3  # Minimum cosine similarity to include a result

# LLM
LLM_MODEL = "gemini-2.5-flash"
MAX_CHAT_HISTORY = 6  # Messages to include in context

# File Processing
MAX_FILE_SIZE_MB = 10
SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".csv", ".xlsx", ".xls"]
SUPPORTED_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg"]
