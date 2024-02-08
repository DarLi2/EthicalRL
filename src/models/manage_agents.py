import pickle
import os

# Specify the relative path to the parent folder
path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(path, 'agents.pkl')

if not os.path.exists(path):
    print("File does not exist. Please train an agent.")
else:
    with open(path, 'rb') as file:
            agents = pickle.load(file)
    agent_selection = None

    if not agents:
             print("No agents saved. Exiting script.")
             exit()
    # Loop to delete agents as long as the user does not type "exit"
    while agent_selection != "exit":

        if not agents:
             print("All agents deleted. Exiting script.")
             break
             
        for agent_name in agents.keys():
            print(agent_name)
        agent_selection = input("Please choose an agent to delete. Type \"exit\" to exit the script. ")
        if agent_selection == "exit":
             break

        try:
            del agents[agent_selection]
            print(f"Agent with name '{agent_selection}' deleted.")
        except KeyError:
            print(f"Agent with name '{agent_selection}' does not exist.")
    

    with open(path, 'wb') as file:
        pickle.dump(agents, file)