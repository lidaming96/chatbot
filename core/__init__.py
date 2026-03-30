"""应用核心业务逻辑（原 Chatbot.py 主体）。"""

from .paths import MEMORY_DIR, USER_DB_FILE
from .auth_memory import (
    deduplicate_items,
    get_memory_file,
    get_memory_context,
    hash_password,
    init_user_db,
    load_memories,
    login_user,
    register_user,
    save_memories,
)
from .conversation_memory import (
    extract_key_facts,
    summarize_conversation,
    update_memory_system,
)
from .documents import extract_document_facts, process_uploaded_document, update_document_memory
from .images import process_uploaded_image, update_image_memory
from .structured_profile import (
    extract_age_from_basic_info,
    format_profile_display,
    integrate_all_memories_to_profile,
    merge_basic_info,
    merge_education_info,
    merge_health_info,
    merge_structured_profile,
    merge_work_info,
    regenerate_structured_profile,
)
from .ui_app import main

__all__ = [
    "MEMORY_DIR",
    "USER_DB_FILE",
    "deduplicate_items",
    "extract_age_from_basic_info",
    "extract_document_facts",
    "extract_key_facts",
    "format_profile_display",
    "get_memory_context",
    "get_memory_file",
    "hash_password",
    "init_user_db",
    "integrate_all_memories_to_profile",
    "load_memories",
    "login_user",
    "main",
    "merge_basic_info",
    "merge_education_info",
    "merge_health_info",
    "merge_structured_profile",
    "merge_work_info",
    "process_uploaded_document",
    "process_uploaded_image",
    "regenerate_structured_profile",
    "register_user",
    "save_memories",
    "summarize_conversation",
    "update_document_memory",
    "update_image_memory",
    "update_memory_system",
]
