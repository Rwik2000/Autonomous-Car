from dqn_try import Agent
import numpy as np
from utils import plotLearning
from gym.envs.box2d  import CarRacing
import gym

if __name__ == "__main__":
    env= gym.make("CarRacing-v0")
    n_games = 500

    agent = Agent(gamma=0.99, epsilon=1.0, alpha=0.0005, input_dims=(96,96,3), n_actions=3, 
                  mem_size=10000, batch_size = 64, epsilon_dec=0.01)
    
    scores = []
    eps_history = []

    for i in range(n_games):
        done = False
        score = 0
        current_state = env.reset()
        print("GAME: "+str(i))
        for t in range(150):
            action = agent.choose_action(current_state)
            # print(action)
            new_state, reward, done, info = env.step(action)
            print(reward)
            score +=reward
            agent.remember(current_state, action, reward, new_state, done)
            current_state = new_state
            agent.learn()
            if i%2==0:
                env.render()
            # print(t)
        eps_history.append(agent.epsilon)
        scores.append(score)

        avg_score = np.mean(scores[max(0,i-100):(i+1)])
        print('episode', i ,'score %.2f' % score, 'average score %.2f'% avg_score)

        if i%10 == 0 and i>0:
            agent.save_model()
    filename = "CarRacing.png"
    x= [i+1 for i in range(n_games)]
    plotLearning(x, scores, eps_history, filename)