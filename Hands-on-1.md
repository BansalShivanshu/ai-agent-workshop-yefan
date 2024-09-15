In this part, a basic reasoning loop is already implemented. You will be adding new tools to the agent and play with LLM reasoning loop.

# Warm-up:
Follow the demo, and make the show_image agent working.

# Exercise 1: Implement a new tool
This exercise is to get you familiar with adding new tools to the agent.

## Goal: 
Add a new function `find_animal_in_an_image` to the agent, and instruct the agent return the type of the animal in the image.

## Instructions:
- Implement a function `get_animal_category` which calls the OpenAI API to get the animal category of the image. [Example API usage](https://platform.openai.com/docs/guides/vision/uploading-and-processing-images)
- Use the query "Print out the animals in the image dog_a.png." and test the agent. Expected result: Dog.

# Exercise 2: Design a reasoning agent
This exercise is to get you familiar with designing your own agent for accomplishing a complex task.

## Goal: 
Ask the agent how many animal feet are there in the images and we'll discuss your answers in the end.

## Instructions:
Use the query "How many feet are there in total for all the animals in the images in the dataset folder"

- Add a tool that could return all the image paths from the image_embedding_store
- Build on top of the `find_animal_in_an_image` function
- Test the agent with the prompt "How many feet are there in total for all the animals in the images in the dataset folder"

# Exercise 3: [Optional]
- Can you ask LLM to find a set of images that the total animal feet in those images are equal to 38?
- Add your favorite animal to the dataset, and have fun with the agent.
