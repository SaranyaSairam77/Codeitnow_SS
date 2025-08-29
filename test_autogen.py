from autogen import AssistantAgent, UserProxyAgent

# Step 1: Add your OpenAI key here or use dotenv
config_list = [
    {
        "model": "gpt-3.5-turbo",
        "api_key": ""
 # Replace this with your real key
    }
]

# Step 2: Create the assistant agent
assistant = AssistantAgent(
    name="Assistant",
    llm_config={"config_list": config_list}
)

# Step 3: Create the user agent (disable Docker execution)
user = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    code_execution_config={"use_docker": False},
)

# Step 4: Start the conversation
user.initiate_chat(
    assistant,
    message="Suggest a healthy millet breakfast for kids",
)
