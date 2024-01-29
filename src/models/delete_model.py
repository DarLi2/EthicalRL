import pickle
import os

# Get the current working directory
current_directory = os.getcwd()

# Specify the relative path to the parent folder
relative_folder_path = '..'
agents_file_path = os.path.join(current_directory, relative_folder_path, 'agents.pkl')

try:
    with open(agents_file_path, 'rb') as file:
        agents = pickle.load(file)
except FileNotFoundError:
    print("No file with stored agents. You need to train and store an agent first. exit")

agent_selection = None
# Loop to delete agents as long as the user does not type "exit"
while agent_selection != "exit":

    for agent_name in agents.keys():
        print(agent_name)
    agent_selection = input("Please choose an agent to delete. Type \"exit\" to exit the script. ")

    try:
        del agents[agent_selection]
        print(f"Agent with name '{agent_selection}' deleted.")
    except KeyError:
        print(f"Agent with name '{agent_selection}' does not exist.")

with open(agents_file_path, 'wb') as file:
    pickle.dump(agents, file)