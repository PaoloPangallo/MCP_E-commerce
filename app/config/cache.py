"""
Configurazione centrale per le policy di caching del sistema.
Centralizza i TTL ed evita valori sparsi nel codice.
"""

# TTL in secondi
DEFAULT_CACHE_TTL = 300       # 5 minuti (default generale)
SHORT_CACHE_TTL = 120         # 2 minuti (per executor o dati volatili)
LONG_CACHE_TTL = 3600         # 1 ora
USER_QUERY_HISTORY_TTL = 86400  # 24 ore

# Mapping specifico per componente (facoltativo se si preferisce la granularità)
SEARCH_PIPELINE_TTL = DEFAULT_CACHE_TTL
TOOL_EXECUTOR_TTL = SHORT_CACHE_TTL
SELLER_FEEDBACK_TTL = DEFAULT_CACHE_TTL
