import openai
import json
import base64
from dotenv import load_dotenv

from image_embedding import image_embedding_store
from details.generate_tools_schema import generate_json_schema

load_dotenv()

# Initialize the OpenAI client
client = openai.OpenAI()

dataset_dir = 'dataset'
img_store = image_embedding_store(dataset_dir)

def find_image_path_based_on_description(description: str):
    """
    Finds one image path based on description

    @param description: image description
    """
    file_name = img_store.find_closest_image_by_linear_search(description)
    return file_name

def show_image(path: str):
    """
    Show the image based on the input path

    @param path: path to the image file
    """
    # TODO: implement this function here using. For example: https://stackoverflow.com/questions/35286540/how-to-display-an-image
    # TODO: add this function to the tools list
    # TODO: update the query and ask the LLM to show an image based on description
    pass

# TODO: add more functions as tools for completing the task of computing the total number of feet in all the images

# generate_json_schema is a convenient helper function to generate a JSON schema for the functions, so we can skip the tedious work of writing the schema manually.
tools = [generate_json_schema(f) for f in [
    find_image_path_based_on_description,
    ]]

def run_image_agent(query: str):

    assistant = client.beta.assistants.create(
        model='gpt-4o-2024-08-06',
        instructions="You are an image assistant. Your job is helping the user identify and understand the images",
        tools=tools,
        name="image-agent",
    )

    thread = client.beta.threads.create()
    client.beta.threads.messages.create(thread_id=thread.id, role="user", content=query)
    run = client.beta.threads.runs.create_and_poll(thread_id=thread.id, assistant_id=assistant.id,)

    max_turns = 50
    for _ in range(max_turns):
        messages = client.beta.threads.messages.list(
                    thread_id=thread.id,
                    run_id=run.id,
                    order="desc",
                    limit=1,
                )
        
        if run.status == "completed":
            return  next((
                    content.text.value
                    for content in messages.data[0].content
                    if content.type == "text"
                ),
                None,
            )

        elif run.status == "requires_action":
            func_tool_outputs = []

            for tool in run.required_action.submit_tool_outputs.tool_calls:

                if tool.function.name in globals() and callable(globals()[tool.function.name]):
                    print(f"Calling Function: {tool.function.name}")
                    func_output = globals()[tool.function.name](**json.loads(tool.function.arguments))
                    func_tool_outputs.append({"tool_call_id": tool.id, "output": str(func_output)})
                else:
                    raise Exception("Function not available")

            # Submit the function call outputs back to OpenAI
            run = client.beta.threads.runs.submit_tool_outputs_and_poll(thread_id=thread.id, run_id=run.id, tool_outputs=func_tool_outputs)

        elif run.status == "failed":
                print(f"Agent run failed for the reason: {run.last_error}")
                break
        else:
            print(f"Run status {run.status} not yet handled")
    else:
        print("Reached maximum reasoning turns, maybe increase the limit?")

if __name__ == "__main__":
    # Testing prompts
    query_find = "Find the image of a cat reading a book"
    result = run_image_agent(query_find)
    print(f"Response from LLM: {result}")

