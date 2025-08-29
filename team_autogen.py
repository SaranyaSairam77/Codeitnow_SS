from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from dotenv import load_dotenv
import os

# ---- Load env & validate key ----
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY", "").strip()

def validate_api_key(key: str):
    # quick sanity check; avoids silent "invalid key" loops
    if not key:
        raise ValueError("OPENAI_API_KEY missing. Put it in a .env file as: OPENAI_API_KEY=sk-...")
    if not (key.startswith("sk-") or key.startswith("sk-proj-")):
        raise ValueError(
            "OPENAI_API_KEY format looks wrong. It must start with 'sk-' or 'sk-proj-'. "
            "Ensure it's on ONE line, no spaces or quotes."
        )

validate_api_key(api_key)

llm_cfg = {"config_list": [{"model": "gpt-3.5-turbo", "api_key": api_key}]}

# ---- Define agents ----
planner = AssistantAgent(
    name="Planner",
    system_message=(
        "Role: Project Planner.\n"
        "Break the user's goal into 3–6 concrete steps. Assign the next step to Engineer.\n"
        "After Engineer replies, ask Critic to review.\n"
        "Do NOT terminate until Critic explicitly replies 'APPROVED.'.\n"
        "When Critic approves, summarize briefly and then say: TERMINATE."
    ),
    llm_config=llm_cfg,
)

engineer = AssistantAgent(
    name="Engineer",
    system_message=(
        "Role: Implementer.\n"
        "When Planner assigns a step, produce concise, actionable output (lists/tables/bullets).\n"
        "Indian kitchen practicality, kid-friendly, diabetic-friendly variants when asked.\n"
        "Do NOT terminate; wait for Critic's review."
    ),
    llm_config=llm_cfg,
)

critic = AssistantAgent(
    name="Critic",
    system_message=(
        "Role: Reviewer.\n"
        "Check correctness, nutrition, and Thinai Organics brand tone (warm, clear, no medical claims).\n"
        "Suggest precise fixes if needed. If acceptable, reply EXACTLY with: APPROVED."
    ),
    llm_config=llm_cfg,
)

user = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    code_execution_config={"use_docker": False},  # important on your machine
)

# ---- Group chat ----
groupchat = GroupChat(
    agents=[user, planner, engineer, critic],
    messages=[],
    max_round=16,
)

# Helper: find agent by name using the manager->groupchat
def get_agent_by_name(manager: GroupChatManager, name: str):
    for a in manager.groupchat.agents:
        if a.name == name:
            return a
    return None

# Custom routing: Planner -> Engineer -> Critic -> Planner ...
def round_robin_selector(last_speaker, manager: GroupChatManager):
    order = ["Planner", "Engineer", "Critic"]
    # First speaker or after User speaks → Planner
    if last_speaker is None or last_speaker.name == "User":
        return get_agent_by_name(manager, "Planner")
    if last_speaker.name not in order:
        return get_agent_by_name(manager, "Planner")
    i = order.index(last_speaker.name)
    next_name = order[(i + 1) % len(order)]
    return get_agent_by_name(manager, next_name)

# Attach selector (GroupChatManager passes itself as 2nd arg)
groupchat.select_speaker = round_robin_selector

manager = GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_cfg,
)

# ---- Kick off ----
user.initiate_chat(
    manager,
    message=(
        "Goal: Create a 1-week breakfast plan using Thinai/millets for kids (6–10 years), "
        "Indian palate, easy to cook, include diabetic-friendly options, and keep it varied. "
        "Output a neat day-wise plan and a short shopping checklist."
    ),
)
